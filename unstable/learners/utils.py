import torch, pathlib
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
try:                from torch.utils.checkpoint import CheckpointImpl
except ImportError: from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, apply_activation_checkpointing


def _load_base(name: str, dtype, device, **kwargs): 
    with torch.device(device): 
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, trust_remote_code=True, **kwargs)

def _freeze(model, ignore_substr: Optional[str] = None):
    for n, p in model.named_parameters():
        if ignore_substr and ignore_substr in n: continue
        p.requires_grad_(False)

def _build_lora(model, lora_cfg: Dict[str, Any], task_type: str):
    return get_peft_model(model, LoraConfig(
        r=lora_cfg.get("lora_rank", 32), lora_alpha=lora_cfg.get("lora_alpha", 32), lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none", task_type=task_type, target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ))

def build_peft_model(base_name: str, device: torch.device, lora_cfg: Dict[str, Any]|None, initial_lora_path: Optional[str]=None, freeze_base: bool=True, critic_model: bool=False, value_head_prefix: str="value_head") -> Tuple[torch.nn.Module, "transformers.PreTrainedTokenizer"]:
    dtype = torch.bfloat16
    task_type = "TOKEN_CLS" if critic_model else "CAUSAL_LM"
    base = get_critic_model(base_name, device, torch_dtype=dtype, value_head_prefix=value_head_prefix) if critic_model else _load_base(base_name, dtype, device)
    if freeze_base: _freeze(base, None if not critic_model else value_head_prefix)
    model = _build_lora(base, lora_cfg or {}, task_type).to(device)
    if initial_lora_path: _load_lora_state(model, initial_lora_path)
    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    return model, tok

def _json_safe(obj):
    if isinstance(obj, set): return list(obj) # turn sets into lists
    raise TypeError # let json handle the rest
    
def enable_full_activation_ckpt(model):
    def checkpoint_everything(mod):
        if isinstance(mod, LoraLayer): return False
        for _, child in mod.named_modules():
            if isinstance(child, LoraLayer):
                return False
        return True
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT), check_fn=checkpoint_everything)  # “always recompute”