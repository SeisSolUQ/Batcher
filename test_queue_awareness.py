import unittest
from unittest.mock import MagicMock
import argparse
import time
import threading
import batcher

class TestQueueAwareness(unittest.TestCase):
    def test_timeout_delayed_by_active_compute(self):
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 2 # Order 3 needs 2 samples to fill
        args.batchsize2 = 1 # Order 4 needs 1 samples to fill
        args.port = 4242
        args.timeout = 0.5 # Wait for 0.5s before submitting

        received_order3_batches = []
        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True

        def mock_compute(params, config):
            if config["order"] == "4":#
                # simulate a long SLURM wait for the order 4 simulation
                time.sleep(1.5)
            elif config["order"] == "3":
                received_order3_batches.append(params)
            return [0.5] * len(params)

        mock_sim.side_effect = mock_compute

        #Initialize batcher
        b = batcher.Batcher(mock_sim, args)

        # Submit an Order 4 query. This fills its batch immediately and
        # blocks the simulator for 5s
        t_order4 = threading.Thread(target=lambda: b([[0.1]], {"order": "4"}))
        t_order4.start()
        time.sleep(0.1)
        t_order3_first = threading.Thread(target=lambda: b([[0.1]], {"order": "3"}))
        t_order3_first.start()

        # We now wait 5.0 on the main thread
        # * OLD BEHAVIOR: The 0.5s timeout expires. The batch pads [0.1] to [[0.1], [0.1]] and submits
        # * NEW_BEHAVIOR: The batch sees Order 4 is still computing, pauses the timeout, and waits.
        time.sleep(1.0)

        # Submit the second Order 3 query.
        # If the batcher waited properly, this perfectly fills the batch to [[0.1], [0.2]]
        t_order3_second = threading.Thread(target=lambda: b([[0.2]], {"order": "3"}))
        t_order3_second.start()

        # Wait for all threads to finish
        t_order4.join()
        t_order3_first.join()
        t_order3_second.join()

        # --- ASSERTIONS ---
        self.assertEqual(len(received_order3_batches), 1, "Order 3 batch only be submitted once")
        submitted_batch = received_order3_batches[0]
        self.assertEqual(submitted_batch, [[0.1], [0.2]], "Batcher padded prematurely due to"
                                                          "timeout! It did not wait for the active Order 4 compute.")

if __name__ == '__main__':
    unittest.main()
