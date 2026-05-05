import argparse
import umbridge
import threading
import time
import copy
import logging
from logging.handlers import RotatingFileHandler
import os
import uuid


# Setup simple logger
def setup_logger(log_file="batcher.log", max_bytes=10 * 1024 * 1024, backup_count=3):
    logger = logging.getLogger("Batcher")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid duplicate logs if root logger is configured

    # Only add a file handler if one for this log_file is not already present
    log_file_path = os.path.abspath(log_file)
    for handler in logger.handlers:
        if isinstance(handler, (logging.FileHandler, RotatingFileHandler)):
            # Compare absolute paths to see if this handler already targets our log file
            if os.path.abspath(getattr(handler, "baseFilename", "")) == log_file_path:
                return logger

    try:
        # Use RotatingFileHandler to prevent unbounded log growth
        fh = RotatingFileHandler(
            log_file, mode="a", maxBytes=max_bytes, backupCount=backup_count
        )
    except OSError:
        # Fall back to stderr if file logging is not possible (e.g., unwritable directory)
        fh = logging.StreamHandler()

    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# Lazy initialization - logger will be set up on first use
_logger = None
_logger_lock = threading.Lock()


def get_logger():
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = setup_logger()
    return _logger


# Define a model that batches parameters per config before sending them to the simulator
class Batcher(umbridge.Model):

    class Batch:
        def __init__(self, config, simulator, cli_args, parent_batcher):
            self.parameters = []
            self.output = None
            self.error = None
            self.thread = None
            self.start_time = time.time()
            self.last_input_time = time.time()
            self.config = config
            self.order = self.config["order"]
            self.simulator = simulator
            self.cli_args = cli_args
            self.parent_batcher = parent_batcher
            self.batchLock = threading.Condition()
            # Use a UUID-based suffix for a practically unique batch ID under high load
            # (using full UUID to ensure uniqueness even under very high load)
            self.batch_id = f"{self.order}_{uuid.uuid4().hex}"
            self.real_param_count = 0  # Track count before padding
            print(
                f"batch instance created with id {self.batch_id} and config: {self.order} at {time.ctime()}"
            )
            self._batchsize = (
                self.cli_args.batchsize2
                if self.order == "4"
                else self.cli_args.batchsize
            )
            print(f"Batch Size for this batch is: {self._batchsize}")

        def is_full(self):
            return len(self.parameters) == self._batchsize

        def _start_timeout_exceeded(self):
            return time.time() - self.last_input_time > self.cli_args.timeout

        def is_computing(self):
            return self.thread is not None

        def _wait_for_active_computes(self, logger, waiting_logged):
            """Check if active computes are running and wait
            for them to finish. Returns True if we should
            continue the loop (i.e. we waited), along with
            updated waiting_logged flag."""
            with self.parent_batcher.active_computes_condition:
                if self.parent_batcher.active_computes <= 0:
                    return False, waiting_logged

                if not waiting_logged:
                    msg = (
                        f"Batch {self.batch_id}"
                        f" (Order {self.order})"
                        " paused timeout."
                        f" Waiting for {self.parent_batcher.active_computes}"
                        " active job(s) to finish..."
                    )
                    print(msg)
                    logger.info(msg)
                    waiting_logged = True

                # Release batchLock so other threads can add
                # samples, then wait on the condition that
                # gets notified when computes finish.
                self.batchLock.release()
                try:
                    with self.parent_batcher.active_computes_condition:
                        self.parent_batcher.active_computes_condition.wait(
                            timeout=self.cli_args.timeout
                        )
                finally:
                    self.batchLock.acquire()
                self.last_input_time = time.time()
                return True, waiting_logged

        def _wait_for_batch_and_submit(self):
            waiting_logged = False
            logger = get_logger()

            with self.batchLock:
                while not self.is_computing():
                    remaining_time = self.cli_args.timeout - (
                        time.time() - self.last_input_time
                    )

                    if self.is_full() or remaining_time <= 0:
                        # check global queue state before padding
                        if not self.is_full():
                            should_wait, waiting_logged = (
                                self._wait_for_active_computes(
                                    logger, waiting_logged
                                )
                            )
                            if should_wait:
                                continue

                        # Store real count before padding
                        self.real_param_count = len(self.parameters)

                        # Pad parameters in case the batch is not full
                        print(
                            f"The actual size of the parameters is {self.real_param_count}"
                        )
                        # Use the last parameter for padding to maintain valid input shapes/values
                        if self.real_param_count > 0:
                            padding_vector = self.parameters[-1]
                        else:
                            # This should not happen since we always add a sample before waiting
                            raise RuntimeError(
                                "Cannot pad an empty batch - no parameters available for shape inference"
                            )

                        while len(self.parameters) < self._batchsize:
                            self.parameters.append(
                                copy.deepcopy(padding_vector))
                        self._compute()
                        self.batchLock.notify_all()
                        break

                    self.batchLock.wait(max(0, remaining_time))

            if self.thread.is_alive():
                self.thread.join()

            if self.output is None and self.error is None:
                raise RuntimeError(
                    "Batch processing finished but no output or error set."
                )

        def add_sample(self, parameter):
            with self.batchLock:
                if self.is_computing() or self.is_full():
                    return -1

                own_entry_index = len(self.parameters)
                self.parameters.append(parameter)
                self.last_input_time = time.time()
                self.batchLock.notify_all()
                return own_entry_index

        def wait_for_result(self, own_entry_index):
            print(
                f"Batched {own_entry_index+1} / {self._batchsize} at {time.ctime()}")

            self._wait_for_batch_and_submit()

            if self.error is not None:
                raise Exception("Batch processing failed") from self.error

            return [self.output[own_entry_index]]

        def _compute(self):
            assert self.thread is None, "Already computing!"
            # Increment BEFORE starting thread to close the
            # race window where another batch could see
            # active_computes == 0 after dispatch.
            with self.parent_batcher.active_computes_condition:
                self.parent_batcher.active_computes += 1
            self.thread = threading.Thread(
                target=self._compute_thread
            )
            self.thread.start()
            print(
                f"Batch started for config: {self.order}"
                f" at {time.ctime()}"
            )

        def _compute_thread(self):
            # Log batch submission with metadata and parameters
            logger = get_logger()
            logger.info(
                f"Batch submitted: batch_id={self.batch_id},"
                f" config_order={self.order},"
                f" real_count={self.real_param_count},"
                f" total_count={len(self.parameters)},"
                f" parameters={self.parameters}"
            )

            # Try this up to 3 times to avoid cluster issues
            last_exception = None
            try:
                for i in range(3):
                    try:
                        self.output = self.simulator(
                            self.parameters, self.config)
                        break
                    except Exception as e:
                        last_exception = e
                        print(
                            f"Failed to submit batch. Retrying {i+1} up to 3 times. Error message: {e}"
                        )
                        logger.exception(
                            f"Simulator call failed (attempt {i+1}/3)")
                        time.sleep(10)
            finally:
                # Decrement global active computes when done (success or fail)
                with self.parent_batcher.active_computes_condition:
                    self.parent_batcher.active_computes -= 1
                    self.parent_batcher.active_computes_condition.notify_all()

            if self.output is None:
                self.error = (
                    last_exception
                    if last_exception
                    else Exception("Batch processing failed with unknown error")
                )

            # Log output received with metadata and actual output
            if self.output is not None:
                output_len = len(self.output) if hasattr(
                    self.output, "__len__") else 1
                logger.info(
                    f"Output received: batch_id={self.batch_id},"
                    f" config_order={self.order},"
                    f" output_length={output_len},"
                    f" parameters={self.parameters},"
                    f" output={self.output}"
                )
            else:
                logger.error(
                    f"Output FAILED: batch_id={self.batch_id},"
                    f" config_order={self.order},"
                    f" parameters={self.parameters},"
                    f" error={str(self.error)}"
                )

            print(f"Output: {self.output}")

    def __init__(self, simulator, cli_args):
        super().__init__(cli_args.model)
        self.simulator = simulator
        self.cli_args = cli_args
        self.current_batches = {}
        self.lock = threading.Lock()
        # Additions for queue awareness
        self.active_computes = 0
        self.active_computes_condition = threading.Condition()

    def get_input_sizes(self, config):
        return [self.simulator.get_input_sizes(config)[0]]

    def get_output_sizes(self, config):
        return [self.simulator.get_output_sizes(config)[0]]

    def __call__(self, parameters, config):
        # Log incoming request with metadata and parameters
        logger = get_logger()
        config_order = config.get("order", "unknown")
        param_lengths = [len(p) if hasattr(p, "__len__")
                         else 1 for p in parameters]
        logger.info(
            f"Request received: config_order={config_order},"
            f" num_parameters={len(parameters)},"
            f" parameter_lengths={param_lengths},"
            f" parameters={parameters}"
        )

        assert (
            len(parameters) == 1
        ), "Batching requires models to have a single input vector!"

        config_unique_identifier = config[
            "order"
        ]  # Identify configurations to be batched separately
        print(f"Unique identifier: {config_unique_identifier}")

        current_batch = None
        own_entry_index = -1

        while True:
            with self.lock:
                current_batch = self.current_batches.get(
                    config_unique_identifier, None)

                if current_batch is None:
                    self.current_batches[config_unique_identifier] = self.Batch(
                        config, self.simulator, self.cli_args, self
                    )
                    current_batch = self.current_batches[config_unique_identifier]

            own_entry_index = current_batch.add_sample(parameters[0])

            if own_entry_index != -1:
                break

            # If full, reset and retry
            with self.lock:
                # Check if it wasn't already replaced by another thread
                if self.current_batches.get(config_unique_identifier) == current_batch:
                    del self.current_batches[config_unique_identifier]

        return current_batch.wait_for_result(own_entry_index)

    def supports_evaluate(self):
        return self.simulator.supports_evaluate()


if __name__ == "__main__":
    # Read CLI arguments
    parser = argparse.ArgumentParser(description="Minimal HTTP model demo.")
    parser.add_argument(
        "url",
        metavar="url",
        type=str,
        help="the URL at which the model is running, for example http://localhost:4242",
    )
    parser.add_argument(
        "model",
        metavar="model",
        type=str,
        help='the model name to connect to, for example "forward"',
    )
    parser.add_argument(
        "batchsize",
        metavar="batchsize",
        type=int,
        help="the batch size to use for coarser model, for example 8",
    )
    parser.add_argument(
        "batchsize2",
        metavar="batchsize2",
        type=int,
        help="the batch size to use for finer model, for example 2",
    )
    parser.add_argument(
        "port", metavar="port", type=int, help="the port to listen on, for example 4242"
    )
    parser.add_argument(
        "timeout",
        metavar="timeout",
        type=float,
        help="the timeout to wait for a batch to fill in seconds, for example 5",
    )
    args = parser.parse_args()
    print(f"Connecting to host URL {args.url}, model {args.model}")

    # Connect to a simulator that receives batches of parameters
    sim = umbridge.HTTPModel(args.url, args.model)

    umbridge.serve_models(
        [Batcher(sim, args)], args.port, max_workers=100, error_checks=False
    )
