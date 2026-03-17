import argparse
import umbridge
import threading
import time

# Define a model that batches parameters per config before sending them to the simulator
class Batcher(umbridge.Model):

    class Batch():
        def __init__(self, config, simulator, cli_args):
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
            self.batchLock = threading.Condition()
            print(f"batch instance created with config: {self.order} at {time.ctime()}")
            self._batchsize = self.cli_args.batchsize2 if self.order=="4" else self.cli_args.batchsize
            print(f"Batch Size for this batch is: {self._batchsize}")

        def is_full(self):
            return len(self.parameters) == self._batchsize 

        def _start_timeout_exceeded(self):
            return time.time() - self.last_input_time > self.cli_args.timeout

        def is_computing(self):
            return self.thread is not None

        def _wait_for_batch_and_submit(self):
            with self.batchLock:
                while not self.is_computing():
                    remaining_time = self.cli_args.timeout - (time.time() - self.last_input_time)
                    
                    if (self.is_full() or remaining_time <= 0):
                        # Pad parameters in case the batch is not full
                        print(f"The actual size of the parameters is {len(self.parameters)}")
                        # Use the last parameter for padding to maintain valid input shapes/values
                        if len(self.parameters) > 0:
                            padding_vector = self.parameters[-1]
                        else:
                             # This should not happen since we always add a sample before waiting
                            raise RuntimeError("Cannot pad an empty batch - no parameters available for shape inference")

                        while len(self.parameters) < self._batchsize:
                            self.parameters.append(padding_vector)
                        self._compute()
                        self.batchLock.notify_all()
                        break
                    
                    self.batchLock.wait(max(0, remaining_time))

            if self.thread.is_alive():
                self.thread.join()

            if self.output is None and self.error is None:
                raise RuntimeError("Batch processing finished but no output or error set.")

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
            print(f"Batched {own_entry_index+1} / {self._batchsize} at {time.ctime()}")
            
            self._wait_for_batch_and_submit()
            
            if self.error is not None:
                raise Exception("Batch processing failed") from self.error
                
            return [self.output[own_entry_index]]

        def _compute(self):
            assert self.thread is None, "Already computing!"
            self.thread = threading.Thread(target=self._compute_thread)
            self.thread.start()
            print(f"Batch started for config: {self.order} at {time.ctime()}")

        def _compute_thread(self):
            # Try this up to 3 times to avoid cluster issues
            last_exception = None
            for i in range(3):
                try:
                    self.output = self.simulator(self.parameters, self.config)
                    break
                except Exception as e:
                    last_exception = e
                    print(f"Failed to submit batch. Retrying {i+1} up to 3 times. Error message: {e}")
                    time.sleep(10)

            if self.output is None:
                self.error = last_exception if last_exception else Exception("Batch processing failed with unknown error")

            print(f"Output: {self.output}")

    def __init__(self, simulator, cli_args):
        super().__init__(cli_args.model)
        self.simulator = simulator
        self.cli_args = cli_args
        self.current_batches = {}
        self.lock = threading.Lock()

    def get_input_sizes(self, config):
        return [self.simulator.get_input_sizes(config)[0]] #Isn't this just the batch size? -> No

    def get_output_sizes(self, config):
        return [self.simulator.get_output_sizes(config)[0]] #Isn't this just the batch size? -> No

    def __call__(self, parameters, config):
        assert len(parameters) == 1, "Batching requires models to have a single input vector!"

        config_unique_identifier = config["order"] # Identify configurations to be batched separately
        print(f"Unique identifier: {config_unique_identifier}")

        current_batch = None
        own_entry_index = -1

        while True:
            with self.lock:
                current_batch = self.current_batches.get(config_unique_identifier, None)
                
                if (current_batch is None):
                    self.current_batches[config_unique_identifier] = self.Batch(config, self.simulator, self.cli_args)
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
    parser = argparse.ArgumentParser(description='Minimal HTTP model demo.')
    parser.add_argument('url', metavar='url', type=str,
                        help='the URL at which the model is running, for example http://localhost:4242')
    parser.add_argument('model', metavar='model', type=str,
                        help='the model name to connect to, for example "forward"')
    parser.add_argument('batchsize', metavar='batchsize', type=int,
                        help='the batch size to use for coarser model, for example 8')
    parser.add_argument('batchsize2', metavar='batchsize2', type=int,
                        help='the batch size to use for finer model, for example 2')
    parser.add_argument('port', metavar='port', type=int,
                        help='the port to listen on, for example 4242')
    parser.add_argument('timeout', metavar='timeout', type=float,
                        help='the timeout to wait for a batch to fill in seconds, for example 5')
    args = parser.parse_args()
    print(f"Connecting to host URL {args.url}, model {args.model}")

    # Connect to a simulator that receives batches of parameters
    sim = umbridge.HTTPModel(args.url, args.model)

    umbridge.serve_models([Batcher(sim, args)], args.port, max_workers=100, error_checks=False)
