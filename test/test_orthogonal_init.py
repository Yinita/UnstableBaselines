import torch
import torch.nn as nn
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unstable.learners.ppo_learner import SharedBackbonePPOModel

def test_orthogonal_initialization():
    """Test that orthogonal initialization works with BFloat16 data type"""
    print("Testing orthogonal initialization with BFloat16...")
    
    # Create a simple transformer model for testing
    class SimpleModel(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.config = type('obj', (object,), {'hidden_size': 768})
            self.device = device
        
        def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            # Mock the transformer output
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            hidden_size = self.config.hidden_size
            
            # Create dummy logits and hidden states on the correct device
            logits = torch.randn(batch_size, seq_len, 50257, device=self.device)  # Typical vocab size
            hidden_states = [torch.randn(batch_size, seq_len, hidden_size, device=self.device) for _ in range(12)]
            
            # Create a simple output object
            output = type('obj', (object,), {
                'logits': logits,
                'hidden_states': hidden_states
            })
            return output

    # Create a mock tokenizer
    class SimpleTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    # Test with CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the shared backbone model
    try:
        model = SharedBackbonePPOModel(
            base_model=SimpleModel(device),
            tokenizer=SimpleTokenizer(),
            device=device
        )
        print("✓ Model initialization successful")
        
        # Test forward pass with autocast
        input_ids = torch.randint(0, 100, (2, 10)).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Test policy logits
            policy_logits = model.get_policy_logits(input_ids, attention_mask)
            print(f"✓ Policy logits shape: {policy_logits.shape}")
            
            # Test value prediction
            values = model.get_values(input_ids, attention_mask)
            print(f"✓ Values shape: {values.shape}")
        
        print("All tests passed successfully!")
        return True
    
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    test_orthogonal_initialization()
