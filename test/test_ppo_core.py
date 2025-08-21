import sys
import os
import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

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
        
        # Test with tensor values
        last_value_tensor = torch.tensor(1.5, dtype=torch.float32)
        advantages, returns = compute_gae(rewards, values, last_value=last_value_tensor, done=False)
        self.assertEqual(advantages.dtype, torch.float32)
        self.assertEqual(returns.dtype, torch.float32)


class TestTokenLevelLogProbs(unittest.TestCase):
    """Test token-level log probability handling"""
    
    def test_padding_alignment(self):
        """Test padding and alignment of token-level log probabilities"""
        # Create sample data
        old_logps = [
            [-1.0, -2.0, -3.0],  # 3 tokens
            [-0.5, -1.5]         # 2 tokens
        ]
        
        label_masks = [
            [1, 1, 1],  # All valid
            [1, 1]      # All valid
        ]
        
        # Compute max length and pad
        max_len = max(len(lp) for lp in old_logps)
        self.assertEqual(max_len, 3)
        
        # Pad logps
        padded_logps = []
        for logps in old_logps:
            padded = logps + [0.0] * (max_len - len(logps))
            padded_logps.append(padded)
            
        # Pad masks
        padded_masks = []
        for mask in label_masks:
            padded = mask + [0] * (max_len - len(mask))
            padded_masks.append(padded)
        
        # Convert to tensors
        old_logps_tensor = torch.tensor(padded_logps, dtype=torch.float32)
        label_masks_tensor = torch.tensor(padded_masks, dtype=torch.bool)
        
        # Check shapes
        self.assertEqual(old_logps_tensor.shape, (2, 3))
        self.assertEqual(label_masks_tensor.shape, (2, 3))
        
        # Check values
        self.assertEqual(old_logps_tensor[0, 0].item(), -1.0)
        self.assertEqual(old_logps_tensor[0, 1].item(), -2.0)
        self.assertEqual(old_logps_tensor[0, 2].item(), -3.0)
        self.assertEqual(old_logps_tensor[1, 0].item(), -0.5)
        self.assertEqual(old_logps_tensor[1, 1].item(), -1.5)
        self.assertEqual(old_logps_tensor[1, 2].item(), 0.0)  # Padded
        
        # Check masks
        self.assertTrue(label_masks_tensor[0, 0].item())
        self.assertTrue(label_masks_tensor[0, 1].item())
        self.assertTrue(label_masks_tensor[0, 2].item())
        self.assertTrue(label_masks_tensor[1, 0].item())
        self.assertTrue(label_masks_tensor[1, 1].item())
        self.assertFalse(label_masks_tensor[1, 2].item())  # Padded
        
        # Test masking
        tok_logp = torch.tensor([
            [-0.8, -1.8, -2.8],
            [-0.3, -1.3, -2.3]
        ], dtype=torch.float32)
        
        # Apply mask
        tok_logp_masked = tok_logp.masked_fill(~label_masks_tensor, 0.0)
        
        # Check masked values
        self.assertEqual(tok_logp_masked[0, 0].item(), -0.8)
        self.assertEqual(tok_logp_masked[0, 1].item(), -1.8)
        self.assertEqual(tok_logp_masked[0, 2].item(), -2.8)
        self.assertEqual(tok_logp_masked[1, 0].item(), -0.3)
        self.assertEqual(tok_logp_masked[1, 1].item(), -1.3)
        self.assertEqual(tok_logp_masked[1, 2].item(), 0.0)  # Masked
        
        # Calculate sequence-level log probs
        valid_tokens = label_masks_tensor.sum(dim=1).clamp_min(1)
        seq_logp = tok_logp_masked.sum(dim=1) / valid_tokens
        
        # Check sequence log probs
        self.assertAlmostEqual(seq_logp[0].item(), (-0.8 - 1.8 - 2.8) / 3, places=5)
        self.assertAlmostEqual(seq_logp[1].item(), (-0.3 - 1.3) / 2, places=5)


class TestPPORatio(unittest.TestCase):
    """Test PPO ratio calculation with token-level log probs"""
    
    def test_token_level_ratio(self):
        """Test token-level ratio calculation"""
        # Create sample data
        tok_logp = torch.tensor([
            [-0.8, -1.8, -2.8],
            [-0.3, -1.3, -2.3]
        ], dtype=torch.float32)
        
        old_logps = torch.tensor([
            [-1.0, -2.0, -3.0],
            [-0.5, -1.5, 0.0]  # Last token is padding
        ], dtype=torch.float32)
        
        label_mask = torch.tensor([
            [True, True, True],
            [True, True, False]  # Last token is padding
        ], dtype=torch.bool)
        
        # Calculate token-level ratio logits
        token_ratio_logits = (tok_logp - old_logps).masked_fill(~label_mask, 0.0)
        
        # Check token ratio logits
        self.assertAlmostEqual(token_ratio_logits[0, 0].item(), 0.2, places=5)
        self.assertAlmostEqual(token_ratio_logits[0, 1].item(), 0.2, places=5)
        self.assertAlmostEqual(token_ratio_logits[0, 2].item(), 0.2, places=5)
        self.assertAlmostEqual(token_ratio_logits[1, 0].item(), 0.2, places=5)
        self.assertAlmostEqual(token_ratio_logits[1, 1].item(), 0.2, places=5)
        self.assertAlmostEqual(token_ratio_logits[1, 2].item(), 0.0, places=5)  # Masked
        
        # Calculate sequence-level ratio logits
        valid_tokens = label_mask.sum(dim=1).clamp_min(1)
        seq_ratio_logits = token_ratio_logits.sum(dim=1) / valid_tokens
        
        # Check sequence ratio logits
        self.assertAlmostEqual(seq_ratio_logits[0].item(), 0.2, places=5)  # (0.2+0.2+0.2)/3
        self.assertAlmostEqual(seq_ratio_logits[1].item(), 0.2, places=5)  # (0.2+0.2)/2
        
        # Calculate ratio
        ratio = torch.exp(seq_ratio_logits)
        
        # Check ratio
        self.assertAlmostEqual(ratio[0].item(), np.exp(0.2), places=5)
        self.assertAlmostEqual(ratio[1].item(), np.exp(0.2), places=5)


class TestKLDivergence(unittest.TestCase):
    """Test KL divergence calculation with token-level log probs"""
    
    def test_token_level_kl(self):
        """Test token-level KL divergence calculation"""
        # Create sample data
        tok_logp = torch.tensor([
            [-0.8, -1.8, -2.8],
            [-0.3, -1.3, -2.3]
        ], dtype=torch.float32)
        
        old_logps = torch.tensor([
            [-1.0, -2.0, -3.0],
            [-0.5, -1.5, 0.0]  # Last token is padding
        ], dtype=torch.float32)
        
        label_mask = torch.tensor([
            [True, True, True],
            [True, True, False]  # Last token is padding
        ], dtype=torch.bool)
        
        # Calculate token-level KL
        token_kl = (old_logps - tok_logp).masked_fill(~label_mask, 0.0)
        
        # Check token KL values
        self.assertAlmostEqual(token_kl[0, 0].item(), -0.2, places=5)
        self.assertAlmostEqual(token_kl[0, 1].item(), -0.2, places=5)
        self.assertAlmostEqual(token_kl[0, 2].item(), -0.2, places=5)
        self.assertAlmostEqual(token_kl[1, 0].item(), -0.2, places=5)
        self.assertAlmostEqual(token_kl[1, 1].item(), -0.2, places=5)
        self.assertAlmostEqual(token_kl[1, 2].item(), 0.0, places=5)  # Masked
        
        # Calculate sequence-level KL
        valid_tokens = label_mask.sum(dim=1).clamp_min(1)
        kl_div = (token_kl.sum(dim=1) / valid_tokens).mean()
        
        # Check KL divergence
        self.assertAlmostEqual(kl_div.item(), -0.2, places=5)  # ((-0.2*3)/3 + (-0.2*2)/2)/2


if __name__ == '__main__':
    unittest.main()
