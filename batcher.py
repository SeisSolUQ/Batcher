import argparse
import umbridge
import threading
import time

# Read CLI arguments
parser = argparse.ArgumentParser(description='Minimal HTTP model demo.')
parser.add_argument('url', metavar='url', type=str,
                    help='the URL at which the model is running, for example http://localhost:4242')
parser.add_argument('model', metavar='model', type=str,
                    help='the model name to connect to, for example "forward"')
parser.add_argument('batchsize', metavar='batchsize', type=int,
                    help='the batch size to use, for example 4')
parser.add_argument('port', metavar='port', type=int,
                    help='the port to listen on, for example 4242')
parser.add_argument('timeout', metavar='timeout', type=int,
                    help='the timeout to wait for a batch to fill, for example 5')
args = parser.parse_args()
print(f"Connecting to host URL {args.url}, model {args.model}")

# Define a model that batches parameters per config before sending them to the simulator
class Batcher(umbridge.Model):

    class Batch():
        def __init__(self, config):
            self.parameters = []
            self.output = None
            self.thread = None
            self.start_time = time.time()
            self.config = config
            mesh_value = self.config["mesh"]
            print(f"batch instant created with config: {mesh_value} at {time.ctime()}")


        def is_full(self):
            return len(self.parameters) == args.batchsize

        def _start_timeout_exceeded(self):
            return time.time() - self.start_time > args.timeout

        def is_computing(self):
            return self.thread is not None

        def _wait_for_batch_and_submit(self):
            while (not self.is_computing()):
                if (self.is_full() or self._start_timeout_exceeded()):
                    self._compute()
                else:
                    time.sleep(.1)

            if self.thread.is_alive():
                self.thread.join()
            while (self.output is None): # Ugly but just to be safe...
                time.sleep(.1)

        def evaluate_batched(self, parameters):
            assert len(parameters) == 1, "Batching requires models to have a single input vector!"

            own_entry_index = len(self.parameters)
            self.parameters.append(parameters[0])

            print(f"Batched {own_entry_index+1} / {args.batchsize}")
            print(f"Parameters: {parameters}")

            self._wait_for_batch_and_submit()
            return [self.output[own_entry_index]]

        def _compute(self):
            assert self.thread is None, "Already computing!"
            self.thread = threading.Thread(target=self._compute_thread)
            self.thread.start()
            mesh_value = self.config["mesh"]
            print(f"Batch started for config: {mesh_value} at {time.ctime()}")

        def _compute_thread(self):
            # Pad parameters in case the batch is not full
            print(f"The actual size of the parameters is {len(self.parameters)}")

            while len(self.parameters) < args.batchsize:
                self.parameters.append([0] * simulator.get_input_sizes()[0])

            # Try this up to 3 times to avoid cluster issues
            for i in range(3):
                try:
                    self.output = simulator(self.parameters, self.config)
                    break
                except Exception as e: 
                    print(f"Failed to submit batch. Retrying {i+1} up to 3 times. Error message: {e}")
                    time.sleep(10)

            print(f"Output: {self.output}")

    def __init__(self, simulator):
        super().__init__(args.model)
        self.simulator = simulator
        self.current_batches = {}

    def get_input_sizes(self, config):
        return [self.simulator.get_input_sizes(config)[0]]

    def get_output_sizes(self, config):
        return [self.simulator.get_output_sizes(config)[0]]

    def __call__(self, parameters, config):
        assert len(parameters) == 1, "Batching requires models to have a single input vector!"

        config_unique_identifier = config["mesh"] # Identify configurations to be batched separately
        print(f"Unique identifier: {config_unique_identifier}")

        current_batch = self.current_batches.get(config_unique_identifier, None)

        if (current_batch is None or current_batch.is_full()):
            self.current_batches[config_unique_identifier] = self.Batch(config)
            current_batch = self.current_batches[config_unique_identifier]

        return current_batch.evaluate_batched(parameters)

    def supports_evaluate(self):
        return self.simulator.supports_evaluate()

# Connect to a simulator that receives batches of parameters
simulator = umbridge.HTTPModel(args.url, args.model)

umbridge.serve_models([Batcher(simulator)], args.port, max_workers=100, error_checks=False)
