import unittest
from unittest.mock import MagicMock
import argparse
import batcher
import threading

class TestMultiDim(unittest.TestCase):
    def test_multidim_vectors(self):
        # Scenario: Input vector has size 2 (e.g., [0.1, 0.2]).
        # The simulator expects input size [2].
        # We submit multiple vectors of size 2.
        # The batcher should batch them as [[0.1, 0.2], [0.3, 0.4]] (or similar).
        
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 2
        args.batchsize2 = 2
        args.port = 4242
        args.timeout = 5.0
        
        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True
        # Mock simulator to return input size 2
        mock_sim.get_input_sizes.return_value = [2]
        mock_sim.get_output_sizes.return_value = [1]
        
        submitted_batches = []
        def process_batch(params, config):
            submitted_batches.append(params)
            return [0.5] * len(params)
            
        mock_sim.side_effect = process_batch
        
        b = batcher.Batcher(mock_sim, args)
        
        # Verify get_input_sizes returns [2]
        self.assertEqual(b.get_input_sizes({"order": "3"}), [2])
        
        results = []
        errors = []
        
        def submit(vec):
            try:
                res = b([vec], {"order": "3"})
                results.append(res)
            except Exception as e:
                errors.append(e)
            
        t1 = threading.Thread(target=submit, args=([0.1, 0.2],))
        t2 = threading.Thread(target=submit, args=([0.3, 0.4],))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        if errors:
            self.fail(f"Errors during submission: {errors}")
        
        # Verify simulation was called once
        mock_sim.assert_called_once()
        self.assertEqual(len(submitted_batches), 1)
        batch = submitted_batches[0]
        
        # Expect [[0.1, 0.2], [0.3, 0.4]] (order might vary due to threading, but content must match)
        batch_tuples = set(tuple(x) for x in batch)
        expected_tuples = {(0.1, 0.2), (0.3, 0.4)}
        
        self.assertEqual(batch_tuples, expected_tuples)
        self.assertEqual(len(batch), 2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], [0.5])
        self.assertEqual(results[1], [0.5])

if __name__ == '__main__':
    unittest.main()
