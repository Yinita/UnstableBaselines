import unstable

from patch_collector_for_openai import patch_collector_for_openai
import os


# OpenAI configuration
OPENAI_MODEL_NAME = "gpt-4o"  # You can change this to any OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
openai_config = {
    "model_name": OPENAI_MODEL_NAME,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
    "verbose": True,
}

# Memory optimization: Reduced LoRA rank
lora_config = {
    "lora_rank": 16,       # Reduced from 32
    "lora_alpha": 16,      # Reduced from 32
    "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj"] # ,"o_proj","gate_proj", "up_proj","down_proj"
}

# Memory optimization: Reduced parallel sequences
vllm_config = {
    "model_name": "Qwen/Qwen3-1.7B-Base", 
    "temperature": 0.6, 
    "max_tokens": 4096,
    "max_parallel_seq": 32,  # Reduced from 128
    "max_loras": 4,          # Reduced from 8
    "lora_config": lora_config,
    "max_model_len": 16000
}

patch_collector_for_openai(openai_config)
run = unstable.build(
    model_name = "Qwen/Qwen3-1.7B-Base",
    train_envs = [unstable.TrainEnvSpec(env_id="SimpleTak-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs")],
    eval_envs = [
        unstable.EvalEnvSpec(env_id="SimpleTak-v0-train", num_players=2, prompt_template="qwen3-zs"),
        unstable.EvalEnvSpec(env_id="KuhnPoker-v0-train", num_players=2, prompt_template="qwen3-zs")
    ]
)
run.start(learning_steps=200, num_collection_workers=256, num_eval_workers=16)
