import sys
import os
import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Add parent directory to path to import unstable modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import ray before other modules to avoid conflicts
import ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

from unstable.learners.ppo_learner import PPOLearner, compute_gae, SharedBackbonePPOModel


@dataclass
class MockStep:
    """Mock step for testing PPO learner"""
    obs: str
    act: str
    reward: float
    step_info: Dict[str, Any] = field(default_factory=dict)


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


class TestPPOLearner(unittest.TestCase):
    """Test PPOLearner class"""
    
    @patch('unstable.learners.base.BaseLearner.__init__')
    @patch('unstable.learners.ppo_learner.build_peft_model')
    def setUp(self, mock_build_peft, mock_base_init):
        """Set up test environment with mocked components"""
        mock_base_init.return_value = None
        # Mock tokenizer to return an object with a .to() method
        self.mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long)
        }
        self.mock_tokenizer.return_value = mock_encoding
        
        # Mock base model
        self.mock_base_model = MagicMock(spec=torch.nn.Module)
        self.mock_base_model.config = MagicMock()
        self.mock_base_model.config.hidden_size = 768
        self.mock_base_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        
        # Mock outputs
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(1, 10, 32000)  # [B, T, V]
        mock_outputs.hidden_states = [torch.randn(1, 10, 768) for _ in range(3)]
        self.mock_base_model.return_value = mock_outputs
        
        # Set up build_peft_model to return our mocks
        mock_build_peft.return_value = (self.mock_base_model, self.mock_tokenizer)
        
        # Since we patched BaseLearner.__init__, we instantiate first and then set attributes
        ActualPPOLearner = PPOLearner.__ray_actor_class__
        self.learner = ActualPPOLearner()
        
        # Manually set attributes that would have been set by BaseLearner.__init__
        self.learner.device = torch.device("cpu")
        self.learner.model_name = "mock_model"
        self.learner.lora_cfg = {}
        self.learner.use_trainer_cache = False
        self.learner.gradient_checkpointing = False
        self.learner.activation_checkpointing = False
        self.learner.batch_size = 4
        self.learner.mini_batch_size = 2
        self.learner.grad_clip = 1.0
        self.learner.logger = MagicMock()
        self.learner.buffer = MagicMock()
        self.learner.tracker = MagicMock()
        self.learner.model_registry = MagicMock()
        
        # Initialize the algorithm
        self.learner.initialize_algorithm(
            infer_mini_batch_size=2,
            learning_rate=1e-5,
            critic_learning_rate=1e-5,
            normalize_adv=True,
            max_train_len=20,
            clip_ratio=0.2,
            ppo_epochs=2,
            use_fallback_advantages=False
        )
        
        # Replace the shared model with a mock
        self.learner.shared_model = MagicMock()
        self.learner.shared_model.get_policy_logits.return_value = torch.randn(2, 10, 32000)
        self.learner.shared_model.get_values.return_value = torch.randn(2)
        
        # Mock tokenizer behavior for _prepare_batch
        self.learner.tokenizer = self.mock_tokenizer
    
    def test_prepare_batch_token_level_logp(self):
        """Test that _prepare_batch correctly handles token-level log probabilities"""
        # Create mock steps with token-level old_logp
        steps = [
            MockStep(
                obs="prompt1", 
                act="response1", 
                reward=1.0, 
                step_info={
                    "old_logp": [-1.0, -2.0, -3.0],
                    "label_mask": [1, 1, 1]
                }
            ),
            MockStep(
                obs="prompt2", 
                act="response2", 
                reward=2.0, 
                step_info={
                    "old_logp": [-0.5, -1.5],
                    "label_mask": [1, 1]
                }
            )
        ]
        
        # Mock tokenizer behavior
        def mock_tokenizer_side_effect(texts, **kwargs):
            output = {
                "input_ids": torch.ones(len(texts), 5, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 5, dtype=torch.long)
            }
            mock_encoding = MagicMock()
            mock_encoding.to.return_value = output
            return mock_encoding
        self.mock_tokenizer.side_effect = mock_tokenizer_side_effect
        
        # Call _prepare_batch
        enc, state_enc, advs, rets, old_logps, label_masks, obs, avg_len, pct_truncated = self.learner._prepare_batch(steps)
        
        # Check that old_logps is a tensor with the right shape
        self.assertIsInstance(old_logps, torch.Tensor)
        self.assertEqual(old_logps.shape[0], 2)  # Batch size
        self.assertEqual(old_logps.shape[1], 3)  # Max sequence length
        
        # Check that padding was applied correctly
        self.assertAlmostEqual(old_logps[0, 0].item(), -1.0)
        self.assertAlmostEqual(old_logps[0, 1].item(), -2.0)
        self.assertAlmostEqual(old_logps[0, 2].item(), -3.0)
        self.assertAlmostEqual(old_logps[1, 0].item(), -0.5)
        self.assertAlmostEqual(old_logps[1, 1].item(), -1.5)
        self.assertAlmostEqual(old_logps[1, 2].item(), 0.0)  # Padded value
        
        # Check label masks
        self.assertIsInstance(label_masks, torch.Tensor)
        self.assertEqual(label_masks.shape, old_logps.shape)
        self.assertEqual(label_masks.dtype, torch.bool)
        
        # First sample: all tokens are valid
        self.assertTrue(label_masks[0, 0].item())
        self.assertTrue(label_masks[0, 1].item())
        self.assertTrue(label_masks[0, 2].item())
        
        # Second sample: first two tokens valid, third is padding
        self.assertTrue(label_masks[1, 0].item())
        self.assertTrue(label_masks[1, 1].item())
        self.assertFalse(label_masks[1, 2].item())  # Padded mask
    
    def test_prepare_batch_scalar_logp_fallback(self):
        """Test that _prepare_batch correctly handles scalar log probabilities (backward compatibility)"""
        # Create mock steps with scalar old_logp
        steps = [
            MockStep(
                obs="prompt1", 
                act="response1", 
                reward=1.0, 
                step_info={
                    "old_logp": -5.0,  # Scalar
                }
            ),
            MockStep(
                obs="prompt2", 
                act="response2", 
                reward=2.0, 
                step_info={
                    "old_logp": -3.0,  # Scalar
                }
            )
        ]
        
        # Mock tokenizer behavior
        def mock_tokenizer_side_effect(texts, **kwargs):
            output = {
                "input_ids": torch.ones(len(texts), 5, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 5, dtype=torch.long)
            }
            mock_encoding = MagicMock()
            mock_encoding.to.return_value = output
            return mock_encoding
        self.mock_tokenizer.side_effect = mock_tokenizer_side_effect
        
        # Call _prepare_batch
        enc, state_enc, advs, rets, old_logps, label_masks, obs, avg_len, pct_truncated = self.learner._prepare_batch(steps)
        
        # Check that old_logps is a tensor with the right shape
        self.assertIsInstance(old_logps, torch.Tensor)
        self.assertEqual(old_logps.shape[0], 2)  # Batch size
        self.assertEqual(old_logps.shape[1], 1)  # Scalar expanded to length 1
        
        # Check values
        self.assertAlmostEqual(old_logps[0, 0].item(), -5.0)
        self.assertAlmostEqual(old_logps[1, 0].item(), -3.0)
        
        # Check default label masks (all True)
        self.assertTrue(label_masks[0, 0].item())
        self.assertTrue(label_masks[1, 0].item())
    
    def test_mini_batch_update_step(self):
        """Test that _mini_batch_update_step correctly uses token-level log probabilities"""
        # Create mock steps
        steps = [
            MockStep(
                obs="prompt1", 
                act="response1", 
                reward=1.0, 
                step_info={
                    "old_logp": [-1.0, -2.0, -3.0],
                    "label_mask": [1, 1, 1],
                    "return": 1.5
                }
            ),
            MockStep(
                obs="prompt2", 
                act="response2", 
                reward=2.0, 
                step_info={
                    "old_logp": [-0.5, -1.5],
                    "label_mask": [1, 1],
                    "return": 2.5
                }
            )
        ]
        
        # Mock _prepare_batch to return controlled values
        def mock_prepare_batch(steps):
            B = len(steps)
            T = 5  # Sequence length
            L = 3  # Max old_logps length
            
            enc = MagicMock()
            enc.input_ids = torch.ones(B, T, dtype=torch.long)
            enc.attention_mask = torch.ones(B, T, dtype=torch.long)
            
            state_enc = MagicMock()
            state_enc.input_ids = torch.ones(B, T, dtype=torch.long)
            state_enc.attention_mask = torch.ones(B, T, dtype=torch.long)
            
            advs = torch.tensor([1.0, 2.0], dtype=torch.float32)
            rets = torch.tensor([1.5, 2.5], dtype=torch.float32)
            
            # Token-level old_logps with padding
            old_logps = torch.zeros(B, L, dtype=torch.float32)
            old_logps[0, :3] = torch.tensor([-1.0, -2.0, -3.0])
            old_logps[1, :2] = torch.tensor([-0.5, -1.5])
            
            # Label masks
            label_masks = torch.zeros(B, L, dtype=torch.bool)
            label_masks[0, :3] = True
            label_masks[1, :2] = True
            
            return enc, state_enc, advs, rets, old_logps, label_masks, ["prompt1", "prompt2"], 10.0, 0.0
        
        # Replace _prepare_batch with our mock
        self.learner._prepare_batch = mock_prepare_batch
        
        # Mock shared_model methods to return controlled values with gradients
        def mock_get_policy_logits(input_ids, attention_mask):
            B, T = input_ids.shape
            return torch.ones(B, T, 10, requires_grad=True) * -1.0
        
        def mock_get_values(input_ids, attention_mask):
            B = input_ids.shape[0]
            return torch.ones(B, requires_grad=True) * 2.0
        
        self.learner.shared_model.get_policy_logits.side_effect = mock_get_policy_logits
        self.learner.shared_model.get_values.side_effect = mock_get_values
        
        # Call _mini_batch_update_step
        with patch('torch.nn.functional.log_softmax', return_value=torch.ones(2, 5, 10) * -2.0):
            with patch('torch.nn.functional.softmax', return_value=torch.ones(2, 5, 10) * 0.1):
                metrics = self.learner._mini_batch_update_step(steps, scaling=1.0)
        
        # Check that the function ran without errors
        self.assertIsInstance(metrics, dict)
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('kl_div', metrics)
    
    def test_normalization_flag_reset(self):
        """Test that _already_normalized flag is reset after each update"""
        # Set the flag to True
        self.learner._already_normalized = True
        
        # Mock _run_ppo_epochs to avoid actual computation
        self.learner._run_ppo_epochs = MagicMock(return_value={})
        
        # Create a simple batch
        batch = [[
            MockStep(
                obs="prompt1", 
                act="response1", 
                reward=1.0, 
                step_info={"return": 1.5, "old_logp": [-1.0]}
            )
        ]]
        
        # Mock tree.flatten to return a list of steps
        with patch('tree.flatten', return_value=[batch[0][0]]):
            # Mock tokenizer to return a value with a .to() method
            mock_encoding = MagicMock()
            mock_encoding.to.return_value = {
                "input_ids": torch.ones(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long)
            }
            self.learner.tokenizer.return_value = mock_encoding

            # Mock compute_gae
            with patch('unstable.learners.ppo_learner.compute_gae', 
                      return_value=(torch.tensor([1.0]), torch.tensor([1.5]))):
                # Call _update
                self.learner._update(batch)
        
        # Check that the flag was reset
        self.assertFalse(self.learner._already_normalized)
    
    def test_fallback_advantages_control(self):
        """Test that fallback advantages are only used when enabled"""
        # Create a batch that would trigger the fallback
        batch = [[
            MockStep(
                obs="prompt1", 
                act="response1", 
                reward=1.0, 
                step_info={"return": 1.5, "old_logp": [-1.0]}
            )
        ]]
        
        # Mock methods to simulate a scenario where GAE computation fails
        def mock_compute_gae(*args, **kwargs):
            raise ValueError("Simulated GAE computation failure")
        
        # Test with fallback disabled
        self.learner.use_fallback_advantages = False
        with patch('unstable.learners.ppo_learner.compute_gae', side_effect=mock_compute_gae):
            with patch('tree.flatten', return_value=[batch[0][0]]):
                # This should not use fallback
                with self.assertRaises(ValueError): # Expecting the GAE error to propagate
                    self.learner._update(batch)
                self.learner.logger.info.assert_not_called()
        
        # Reset mock
        self.learner.logger.reset_mock()
        
        # Test with fallback enabled
        self.learner.use_fallback_advantages = True
        with patch('unstable.learners.ppo_learner.compute_gae', side_effect=mock_compute_gae):
            with patch('tree.flatten', return_value=[batch[0][0]]):
                # This should attempt to use fallback
                self.learner._update(batch)
                # Check that the logger was called with the fallback message
                fallback_called = any(
                    "fallback" in str(call).lower()
                    for call in self.learner.logger.info.call_args_list
                )
                self.assertTrue(fallback_called)


if __name__ == '__main__':
    unittest.main()
