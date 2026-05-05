import unittest
from unittest.mock import MagicMock
import argparse
import time
import threading
import batcher


class TestQueueAwareness(unittest.TestCase):
    def test_full_chain_starvation_scenario(self):
        """
        Simulates 8 parallel chains where 1 chain is delayed by an Order 4 simulation.
        Ensures the Order 3 batcher waits for the 8th chain to finish its Order 4
        compute instead of prematurely padding the 7 waiting queries.
        """
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 8  # Order 3 needs all 8 chains
        args.batchsize2 = 1  # Order 4 processes individually
        args.port = 4242
        args.timeout = 0.5  # Short timeout to force the padding issue

        received_order3_batches = []
        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True

        def mock_compute(params, config):
            if config["order"] == "4":
                # Simulate SLURM queue wait for the delayed chain
                time.sleep(1.5)
            elif config["order"] == "3":
                # Store a sorted version of the batch so we can easily assert its contents
                # since threads might append in a slightly unpredictable order
                sorted_params = sorted(params, key=lambda x: x[0])
                received_order3_batches.append(sorted_params)
            return [0.5] * len(params)

        mock_sim.side_effect = mock_compute
        b = batcher.Batcher(mock_sim, args)

        # --- The Delayed Chain (Chain 8) ---
        def chain8_worker():
            # First, it gets stuck doing an Order 4 simulation
            b([[0.8]], {"order": "4"})
            # Once it finally finishes, it generates its Order 3 query
            b([[0.8]], {"order": "3"})

        t_delayed_chain = threading.Thread(target=chain8_worker)
        t_delayed_chain.start()

        # Give Chain 8 a moment to register its active compute
        time.sleep(0.1)

        # --- The Fast Chains (Chains 1 to 7) ---
        # These chains submit their Order 3 queries and hit the timeout waiting for Chain 8
        fast_chain_threads = []
        for i in range(1, 8):
            val = i / 10.0  # Values: 0.1, 0.2, ..., 0.7
            t = threading.Thread(target=lambda v=val: b([[v]], {"order": "3"}))
            fast_chain_threads.append(t)
            t.start()

        # Wait for all 8 chains to complete their lifecycles
        t_delayed_chain.join()
        for t in fast_chain_threads:
            t.join()

        # --- ASSERTIONS ---
        self.assertEqual(
            len(received_order3_batches),
            1,
            "The batcher panicked and submitted multiple fragmented batches!",
        )

        submitted_batch = received_order3_batches[0]

        # Verify it waited for all 8 unique queries.
        # If it failed, it would have padded the last fast chain's value (e.g., duplicate [0.7]s)
        # and missed the [0.8] from the delayed chain entirely.
        expected_batch = [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]]

        self.assertEqual(
            len(submitted_batch), 8, "Batch does not contain exactly 8 items."
        )
        self.assertEqual(
            submitted_batch,
            expected_batch,
            "Batcher padded the batch with duplicates instead of waiting for Chain 8's [0.8] query.",
        )

    def test_timeout_delayed_by_active_compute(self):
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 2  # Order 3 needs 2 samples to fill
        args.batchsize2 = 1  # Order 4 needs 1 samples to fill
        args.port = 4242
        args.timeout = 0.5  # Wait for 0.5s before submitting

        received_order3_batches = []
        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True

        def mock_compute(params, config):
            if config["order"] == "4":
                # simulate a long SLURM wait for the order 4 simulation
                time.sleep(1.5)
            elif config["order"] == "3":
                received_order3_batches.append(params)
            return [0.5] * len(params)

        mock_sim.side_effect = mock_compute

        # Initialize batcher
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
        self.assertEqual(
            len(received_order3_batches), 1, "Order 3 batch only be submitted once"
        )
        submitted_batch = received_order3_batches[0]
        self.assertEqual(
            submitted_batch,
            [[0.1], [0.2]],
            "Batcher padded prematurely due to"
            "timeout! It did not wait for the active Order 4 compute.",
        )


if __name__ == "__main__":
    unittest.main()
