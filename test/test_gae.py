import sys
import os
import unittest
import torch

# Add parent directory to path to import unstable modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unstable.learners.ppo_learner import compute_gae


class TestComputeGAE(unittest.TestCase):
    """Test compute_gae function with consistent dtypes"""
    
    def test_compute_gae_dtype_consistency(self):
        """Test that compute_gae handles different dtypes correctly"""
        # Create rewards as float64 and values as float32
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        values = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
        
        # This should not raise any errors due to dtype mismatch
        advantages, returns = compute_gae(rewards, values, last_value=0.0, done=True)
        
        # Check that outputs are float32
        self.assertEqual(advantages.dtype, torch.float32)
        self.assertEqual(returns.dtype, torch.float32)
        
        # Test with last_value as scalar
        advantages, returns = compute_gae(rewards, values, last_value=1.5, done=False)
        self.assertEqual(advantages.dtype, torch.float32)
        self.assertEqual(returns.dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
