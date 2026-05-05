import unittest
from unittest.mock import MagicMock
import argparse
import time
import threading
import batcher


class TestQueueAwareness(unittest.TestCase):
    def test_race_condition_active_computes_increment(self):
        """
        Bug 1: active_computes was previously incremented
        inside _compute_thread(), AFTER thread.start()
        returned. This test verifies the fix: we monkey-patch
        _compute() to read active_computes right after
        thread.start(), proving the counter is already
        incremented before the worker body runs.
        """
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 1
        args.batchsize2 = 1
        args.port = 4242
        args.timeout = 0.5

        sim_blocked = threading.Event()
        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True

        def mock_compute(params, config):
            sim_blocked.wait(timeout=5)
            return [0.5] * len(params)

        mock_sim.side_effect = mock_compute
        b = batcher.Batcher(mock_sim, args)

        # Track active_computes right after _compute() returns
        # by hooking _compute to record the value immediately
        # after thread.start().
        observed_count = [None]
        original_compute = (
            batcher.Batcher.Batch._compute
        )

        def patched_compute(batch_self):
            original_compute(batch_self)
            # Read active_computes RIGHT AFTER _compute()
            # returns (thread.start() just happened).
            # If increment is in the thread, this is 0.
            observed_count[0] = (
                batch_self.parent_batcher.active_computes
            )

        batcher.Batcher.Batch._compute = patched_compute
        try:
            t = threading.Thread(
                target=lambda: b(
                    [[1.0]], {"order": "3"}
                )
            )
            t.start()
            time.sleep(0.5)

            self.assertGreaterEqual(
                observed_count[0],
                1,
                "active_computes was"
                f" {observed_count[0]} right after"
                " _compute() returned. The increment"
                " is racing inside the worker thread.",
            )
        finally:
            batcher.Batcher.Batch._compute = (
                original_compute
            )
            sim_blocked.set()
            t.join(timeout=5)

    def test_batch_wakes_immediately_after_compute_finishes(
        self,
    ):
        """
        Bug 2: The batch polls on batchLock.wait(timeout) but
        notify_all() fires on active_computes_condition. Nobody
        listens on it, so the batch sits for an extra full
        timeout cycle after computes finish.

        Uses a long timeout (2.0s) so the extra polling delay
        is clearly measurable. Order 4 finishes at ~1.5s.
        With notify: batch pads at ~1.5s + 2.0s = ~3.5s
        Without notify (bug): batch pads at ~1.5s + ~2*2.0s
        = ~5.5s (extra cycle from polling miss).
        """
        ORDER4_SLEEP = 1.5
        TIMEOUT = 2.0  # Long timeout to make delay obvious

        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 2
        args.batchsize2 = 1
        args.port = 4242
        args.timeout = TIMEOUT

        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True

        def mock_compute(params, config):
            if config["order"] == "4":
                time.sleep(ORDER4_SLEEP)
            return [0.5] * len(params)

        mock_sim.side_effect = mock_compute
        b = batcher.Batcher(mock_sim, args)

        # Start Order 4 compute (blocks for ORDER4_SLEEP)
        t_order4 = threading.Thread(
            target=lambda: b([[0.1]], {"order": "4"})
        )
        t_order4.start()
        time.sleep(0.1)

        # Submit 1 Order 3 query (needs 2 to fill).
        order3_start = time.time()
        t_order3 = threading.Thread(
            target=lambda: b([[0.1]], {"order": "3"})
        )
        t_order3.start()

        t_order4.join(timeout=15)
        t_order3.join(timeout=15)
        order3_elapsed = time.time() - order3_start

        # With proper notify: batch wakes at ~1.5s, gets
        # fresh timeout of 2.0s, submits at ~3.5s.
        # Total from order3_start: ~3.5s
        # Without notify (bug): batch polls at 2.0s
        # intervals. First poll at ~2.0s sees active > 0,
        # resets. Second poll at ~4.0s sees active == 0,
        # resets. Third poll at ~6.0s, submits.
        # Total: ~6.0s
        max_acceptable = ORDER4_SLEEP + 2.0 * TIMEOUT
        self.assertLess(
            order3_elapsed,
            max_acceptable,
            f"Order 3 batch took {order3_elapsed:.2f}s"
            f" but should have completed within"
            f" {max_acceptable:.2f}s. The batch is not"
            " waking up when active computes finish.",
        )

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
        # blocks the simulator for 1.5s
        t_order4 = threading.Thread(target=lambda: b([[0.1]], {"order": "4"}))
        t_order4.start()
        time.sleep(0.1)
        t_order3_first = threading.Thread(target=lambda: b([[0.1]], {"order": "3"}))
        t_order3_first.start()

        # We now wait 1.0s on the main thread
        # * OLD BEHAVIOR: The 0.5s timeout expires. The batch pads [0.1] to [[0.1], [0.1]] and submits
        # * NEW BEHAVIOR: The batch sees Order 4 is still computing, pauses the timeout, and waits.
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

    def test_thread_start_failure_rollback(self):
        """
        Copilot review #7: If Thread() or start() raises
        (e.g. thread exhaustion), active_computes must be
        rolled back. Otherwise the counter stays positive
        forever and every later partial batch hangs.
        """
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 1
        args.batchsize2 = 1
        args.port = 4242
        args.timeout = 0.5

        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True
        mock_sim.side_effect = lambda params, config: (
            [0.5] * len(params)
        )
        b = batcher.Batcher(mock_sim, args)

        # Monkey-patch threading.Thread to raise on the first
        # call, simulating thread exhaustion.
        original_thread = threading.Thread
        call_count = [0]

        def failing_thread(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Thread exhaustion!")
            return original_thread(*a, **kw)

        import unittest.mock as um
        with um.patch("threading.Thread", side_effect=failing_thread):
            # _compute should raise but rollback active_computes
            batch = batcher.Batcher.Batch(
                {"order": "3"}, mock_sim, args, b
            )
            batch.parameters = [[1.0]]
            batch.real_param_count = 1
            with self.assertRaises(RuntimeError):
                batch._compute()

        # active_computes must be back to 0
        self.assertEqual(
            b.active_computes, 0,
            "active_computes was not rolled back after"
            " thread creation failure!",
        )


if __name__ == "__main__":
    unittest.main()
