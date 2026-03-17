import unittest
from unittest.mock import MagicMock
import argparse
import batcher


class TestPadding(unittest.TestCase):
    def test_padding_shape_mismatch(self):
        # Scenario: Input vector has size 2 and batch size is 2.
        # We submit 1 item and let the timeout occur so the batch is completed via padding.
        # The padding behavior is to reuse the last submitted parameter, producing [[0.1, 0.2], [0.1, 0.2]].
        # The simulator (mock) should therefore receive a batch where all vectors have the same length.
        
        args = argparse.Namespace()
        args.url = "http://localhost:4242"
        args.model = "test_model"
        args.batchsize = 2
        args.batchsize2 = 2
        args.port = 4242
        args.timeout = 0.1
        
        mock_sim = MagicMock()
        mock_sim.supports_evaluate.return_value = True
        
        def check_input_shape(params, config):
            # params should be a list of vectors
            # All vectors should have the same length
            first_len = len(params[0])
            for i, p in enumerate(params):
                if len(p) != first_len:
                    raise ValueError(f"Shape mismatch at index {i}: expected {first_len}, got {len(p)}")
            return [[0.5]] * len(params)
            
        mock_sim.side_effect = check_input_shape
        
        b = batcher.Batcher(mock_sim, args)
        
        # Submit a vector of size 2
        try:
            # This should now succeed because padding will match the input vector [0.1, 0.2]
            b([[0.1, 0.2]], {"order": "3"})
        except Exception as e:
            self.fail(f"Test failed with unexpected error: {e}")

if __name__ == '__main__':
    unittest.main()
