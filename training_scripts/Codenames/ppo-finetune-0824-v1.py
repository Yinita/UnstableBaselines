from mixed_play_builder import build_mixed_play, MixedPlayEvalEnvSpec
import os
import unstable

# Resource/env configuration
os.environ["COLLECTOR_GPUS"] = "2"
os.environ["RECORD_MODELS"] = "openai-gpt-5,openai-gpt-4o"

# OpenAI global config (used by the OpenAI shim in patch_collector_for_openai)
openai_global_config = {
    "verbose": False,
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "quiet_console": True,
}

# Model and LoRA config
# model_name = "Qwen/Qwen3-8B"
model_name = "yinita/qwen3-8b-v1-lora-0812-3epochs"

fixed_opponents = []  # Example: ["openai-gpt-4o", "openai-gpt-5-chat"]

lora_config = {
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj"],  # , "o_proj","gate_proj", "up_proj","down_proj"
}

# Eval envs (enable as needed)
eval_envs = [
    # MixedPlayEvalEnvSpec(
    #     env_id="Codenames-v0",
    #     num_players=4,
    #     prompt_template="qwen3-no-reasoning",
    #     opponent_mapping={1: "openai-gpt-4o", 2: "openai-gpt-5-chat", 3: "openai-gpt-5"},
    # ),
    # MixedPlayEvalEnvSpec(
    #     env_id="SecretMafia-v0",
    #     num_players=6,
    #     prompt_template="qwen3-no-reasoning",
    #     opponent_mapping={1: "openai-gpt-4o", 2: "openai-gpt-4o-mini", 3: "openai-gpt-5", 4: "openai-gpt-5-chat", 5: "openai-gpt-4o"},
    # ),
    # MixedPlayEvalEnvSpec(
    #     env_id="ThreePlayerIPD-v0",
    #     num_players=3,
    #     prompt_template="qwen3-no-reasoning",
    #     opponent_mapping={1: "openai-gpt-4o", 2: "openai-gpt-5"},
    # ),
    MixedPlayEvalEnvSpec(
        env_id="ColonelBlotto-v0",
        num_players=2,
        prompt_template="qwen3-no-reasoning",
        opponent_mapping={1: "openai-gpt-5"},
    ),
    # MixedPlayEvalEnvSpec(
    #     env_id="TwentyQuestions-v0",
    #     num_players=1,
    #     prompt_template="qwen3-no-reasoning",
    #     opponent_mapping={},
    # ),
]

if __name__ == "__main__":
    # Build and start training
    run = build_mixed_play(
        model_name=model_name,
        train_envs=[
            # unstable.TrainEnvSpec(env_id="Codenames-v0-train", num_players=4, num_actors=4, prompt_template="qwen3-no-reasoning"),
            # unstable.TrainEnvSpec(env_id="SecretMafia-v0-train", num_players=6, num_actors=6, prompt_template="qwen3-no-reasoning"),
            # unstable.TrainEnvSpec(env_id="ThreePlayerIPD-v0-train", num_players=3, num_actors=3, prompt_template="qwen3-no-reasoning"),
            unstable.TrainEnvSpec(env_id="ColonelBlotto-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-no-reasoning"),
            # unstable.TrainEnvSpec(env_id="TwentyQuestions-v0-train", num_players=1, num_actors=1, prompt_template="qwen3-no-reasoning"),
        ],
        eval_envs=eval_envs,
        openai_config=openai_global_config,
        fixed_opponents=fixed_opponents,
        algorithm="ppo",
        lora_config=lora_config,
        vllm_config={
            "model_name": model_name,
            "temperature": 0.8,
            "max_tokens": 2048,
            "max_parallel_seq": 32,  # Reduced from 128
            "max_loras": 2,          # Reduced from 8
            "lora_config": lora_config,
            "max_model_len": 6000,
            "gpu_memory_utilization": 0.9,
        },
        wandb_project="UB-ColonelBlotto",    # or None to disable wandb
        wandb_run_name="ppo-finetune-0824-v1"  # optional; auto-generated if omitted
    )

    # Start training
    run.start(learning_steps=1000, num_collection_workers=32, num_eval_workers=4)
