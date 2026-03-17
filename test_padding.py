import unittest
from unittest.mock import MagicMock
import argparse
import batcher

class TestPadding(unittest.TestCase):
    def test_padding_shape_mismatch(self):
        # Scenario: Input vector has size 2. Batch size is 2.
        # We submit 1 item. Timeout occurs.
        # Verify that the batcher pads with the last submitted item ([0.1, 0.2])
        # to ensure the simulator receives a valid, consistent batch.
        
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
            return [0.5] * len(params)
            
        mock_sim.side_effect = check_input_shape
        
        b = batcher.Batcher(mock_sim, args)
        
        # Submit a vector of size 2
        # This should now succeed because padding will match the input vector [0.1, 0.2]
        b([[0.1, 0.2]], {"order": "3"})
        
        # Verify that the simulator was called with a batch of size 2
        # And that both elements are [0.1, 0.2] (the padded one matches the original)
        mock_sim.assert_called_once()
        call_args = mock_sim.call_args
        submitted_params = call_args[0][0] # First arg is parameters
        
        self.assertEqual(len(submitted_params), 2, "Batch size should be padded to 2")
        self.assertEqual(submitted_params[0], [0.1, 0.2], "First item mismatch")
        self.assertEqual(submitted_params[1], [0.1, 0.2], "Padded item mismatch (should match last input)")

if __name__ == '__main__':
    unittest.main()
