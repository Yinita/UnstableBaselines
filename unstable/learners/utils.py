import torch, pathlib
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
try:                from torch.utils.checkpoint import CheckpointImpl
except ImportError: from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, apply_activation_checkpointing

def build_critic_cls(base_cls, base_pretrain_cls, value_head_prefix):
    class CriticModel(base_pretrain_cls):
        supports_gradient_checkpointing = True
        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_cls(config))
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, torch.nn.Linear(config.hidden_size, 1, bias=False))

        def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, return_output=False, **_) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1; position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(input_ids, attention_mask=attention_mask, position_ids=position_ids)
            last_hidden_states = outputs["last_hidden_state"]; values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)
            if return_output:   return (values, outputs)
            else:               return values
    return CriticModel

def get_critic_model(pretrain_or_model: str, device: torch.device, torch_dtype, use_flash_attention_2: bool=False, value_head_prefix: str="value_head"):
    config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    base_class = AutoModel._model_mapping[type(config)]
    critic_cls = build_critic_cls(base_class, base_class.__base__, value_head_prefix)
    # 重要：不要把 torch.device 直接传入 device_map，否则在受限 CUDA_VISIBLE_DEVICES 场景会触发 invalid device ordinal
    # 先用 auto 进行安全加载，后续在调用方统一 model.to(safe_device)
    model = critic_cls.from_pretrained(
        pretrain_or_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None  # load on CPU, caller will move to a single safe_device explicitly
    )
    value_head = getattr(model, value_head_prefix)
    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
    return model

def _load_base(name: str, dtype, device, **kwargs): 
    # 安全地处理CUDA设备
    try:
        return AutoModelForCausalLM.from_pretrained(
            name, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            device_map=None,  # 不进行自动分片，由调用方统一迁移到单一设备
            **kwargs
        )
    except Exception as e:
        print(f"加载模型时出错: {e}")
        # 尝试在CPU上加载模型作为后备方案
        print("尝试在CPU上加载模型...")
        return AutoModelForCausalLM.from_pretrained(
            name, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            device_map='cpu',  # 强制使用CPU
            **kwargs
        )

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
    # 打印设备信息以便调试
    print(f"Using device: {device}")
    
    # 确保设备有效
    safe_device = ensure_valid_device(device)
    print(f"Safe device for model building: {safe_device}")
    
    task_type = "TOKEN_CLS" if critic_model else "CAUSAL_LM"
    
    # 使用安全设备加载基础模型
    base = get_critic_model(base_name, safe_device, torch_dtype=torch.bfloat16, value_head_prefix=value_head_prefix) if critic_model else _load_base(base_name, torch.bfloat16, safe_device)
    
    if freeze_base: 
        _freeze(base, None if not critic_model else value_head_prefix)
    
    # 构建LoRA模型，但不立即移动到设备
    try:
        model = _build_lora(base, lora_cfg or {}, task_type)
        print(f"LoRA模型构建成功，当前设备: {next(model.parameters()).device}")
        
        # 安全地将模型移动到目标设备
        try:
            # 尝试使用device_map参数而不是to(device)
            model = model.to(safe_device)
            print(f"模型成功移动到设备: {next(model.parameters()).device}")
        except Exception as e:
            print(f"移动模型到设备{safe_device}失败: {e}")
            print("尝试使用CPU作为后备...")
            model = model.to('cpu')
            print(f"模型已移动到CPU: {next(model.parameters()).device}")
    except Exception as e:
        print(f"构建LoRA模型失败: {e}")
        raise
    
    # 加载初始LoRA状态（如果提供）
    if initial_lora_path: 
        _load_lora_state(model, initial_lora_path)
    
    # 加载分词器
    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    
    return model, tok

# 辅助函数：确保设备有效
def ensure_valid_device(device):
    """确保设备有效，如果无效则回退到安全设备"""
    try:
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            cuda_device_count = torch.cuda.device_count()
            print(f"系统中可用的CUDA设备数量: {cuda_device_count}")
            
            # 如果是CUDA设备，验证设备ID是否有效
            if isinstance(device, torch.device) and device.type == 'cuda':
                device_id = device.index if device.index is not None else 0
                if device_id >= cuda_device_count:
                    print(f"警告: 请求的CUDA设备ID {device_id} 超出了可用范围 (0-{cuda_device_count-1})")
                    # 回退到第一个可用的CUDA设备
                    return torch.device('cuda:0')
                return device
            return device
        else:
            print("CUDA不可用，使用CPU")
            return torch.device('cpu')
    except Exception as e:
        print(f"设备验证出错: {e}，回退到CPU")
        return torch.device('cpu')

def _json_safe(obj):
    if isinstance(obj, set): return list(obj) # turn sets into lists
    raise TypeError # let json handle the rest
    
def enable_full_activation_ckpt(model):
    def checkpoint_everything(mod):
        if isinstance(mod, LoraLayer): return False
        for _, child in mod.named_modules():
            if isinstance(child, LoraLayer): return False
        return True
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT), check_fn=checkpoint_everything)  # "always recompute"