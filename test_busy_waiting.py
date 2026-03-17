import unittest
from unittest.mock import MagicMock, patch
import time
import argparse
import batcher

class TestBusyWaiting(unittest.TestCase):
    def test_busy_wait_loop(self):
        # Setup
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 2 # Don't fill immediately, wait for timeout
        args.batchsize2 = 2
        args.port = 4242
        args.timeout = 0.5 # Wait for 0.5s before submitting
        
        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True
        
        def mock_compute(params, config):
            time.sleep(0.01) # Simulate computation time
            return [0.5] * len(params)
        mock_sim.side_effect = mock_compute
        
        b = batcher.Batcher(mock_sim, args)
        
        # We need to spy on time.sleep while letting it actually run
        # Use wraps=time.sleep so it calls the real function but tracks calls
        with patch('time.sleep', wraps=time.sleep) as mock_sleep:
            # Submit one request. Since batchsize=2, it will wait for 0.5s timeout.
            # During this wait, the current implementation sleeps every 0.1s.
            # So expected calls ~ 0.5 / 0.1 = 5 calls.
            print("Submitting request...")
            b([[0.1]], {"order": "3"})
            
            call_count = mock_sleep.call_count
            print(f"time.sleep called {call_count} times")
            
            # Assert that we are NOT busy waiting (sleep calls should be minimal)
            # With Condition.wait, time.sleep should only be called by our mock_compute (once)
            self.assertLessEqual(call_count, 2, "Expected minimal sleep calls (no busy waiting)")
            
if __name__ == '__main__':
    unittest.main()
