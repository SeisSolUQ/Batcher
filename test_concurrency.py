import threading
import unittest
from unittest.mock import MagicMock
import argparse
import batcher

class TestRaceCondition(unittest.TestCase):
    def run_concurrency_test(self, batchsize1, batchsize2, num_threads=20):
        """Helper to run concurrent requests with specific batch sizes"""
        
        # Create args object
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = batchsize1
        args.batchsize2 = batchsize2
        args.port = 4242
        # Use a very small timeout to encourage race conditions between timeout and full batch
        args.timeout = 0.05 
        
        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True
        
        def check_batch_size(params, config):
            # Verify that we never receive more items than the configured batch size
            current_batch_size = batchsize2 if config["order"] == "4" else batchsize1
            if len(params) > current_batch_size:
                raise Exception(f"Batch size exceeded: {len(params)} > {current_batch_size}")
            return [0.5] * len(params)

        mock_sim.side_effect = check_batch_size
        
        # Initialize Batcher with args
        b = batcher.Batcher(mock_sim, args)
        
        results = []
        errors = []
        
        barrier = threading.Barrier(num_threads)

        def submit_request():
            try:
                barrier.wait()
                # order="3" uses batchsize1
                res = b([[0.1]], {"order": "3"})
                results.append(res)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=submit_request)
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        if errors:
            print(f"Errors with config ({batchsize1}, {batchsize2}): {errors[0]}")

        self.assertEqual(len(results), num_threads, f"Failed with config ({batchsize1}, {batchsize2})")
        self.assertEqual(len(errors), 0, f"Errors occurred with config ({batchsize1}, {batchsize2})")

    def test_scenarios(self):
        scenarios = [
            (1, 1),
            (2, 1),
            (8, 1),
            (8, 2),
            (8, 8)
        ]
        
        for b1, b2 in scenarios:
            with self.subTest(b1=b1, b2=b2):
                print(f"Testing scenario: batchsize={b1}, batchsize2={b2}")
                self.run_concurrency_test(b1, b2)

if __name__ == '__main__':
    unittest.main()
