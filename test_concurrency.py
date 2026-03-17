import threading
import time
import unittest
from unittest.mock import MagicMock
import argparse
import batcher

class TestRaceCondition(unittest.TestCase):
    def test_concurrent_requests(self):
        # Create args object
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 1
        args.batchsize2 = 1
        args.port = 4242
        args.timeout = 0.5
        
        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True
        
        def check_batch_size(params, config):
            if len(params) > 1:
                raise Exception(f"Batch size exceeded: {len(params)} > 1")
            return [0.5] * len(params)

        mock_sim.side_effect = check_batch_size
        
        # Initialize Batcher with args
        b = batcher.Batcher(mock_sim, args)
        
        results = []
        errors = []
        
        num_threads = 20
        barrier = threading.Barrier(num_threads)

        def submit_request():
            try:
                barrier.wait()
                # order="3" uses batchsize (1)
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
            
        print(f"Results: {len(results)}")
        print(f"Errors: {len(errors)}")
        if errors:
            print(f"First error: {errors[0]}")

        self.assertEqual(len(results), num_threads)
        self.assertEqual(len(errors), 0)

if __name__ == '__main__':
    unittest.main()