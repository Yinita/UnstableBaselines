from mixed_play_builder import build_mixed_play, MixedPlayEvalEnvSpec
import os
import unstable

# OpenAI全局配置
openai_global_config = {
    "verbose": True,
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
}

# 定义固定对手
fixed_opponents = ["openai-gpt-4o", "openai-gpt-4o-mini", "openai-gpt-5", "openai-gpt-5-chat"]

# 创建评估环境规范
eval_envs = [
    MixedPlayEvalEnvSpec(
        env_id="Codenames-v0", 
        num_players=4, 
        prompt_template="qwen3-zs",
        opponent_mapping={1: "openai-gpt-4o", 2: "openai-gpt-5-chat", 3: "openai-gpt-5"}
    ),
]

# 构建并启动训练
run = build_mixed_play(
    model_name="Qwen/Qwen3-1.7B-Base",
    train_envs=[
        unstable.TrainEnvSpec(env_id="Codenames-v0-train", num_players=4, num_actors=2, prompt_template="qwen3-zs")
    ],
    eval_envs=eval_envs,
    openai_config=openai_global_config,
    fixed_opponents=fixed_opponents,
    algorithm="a2c"
)

# 开始训练
run.start(learning_steps=200, num_collection_workers=32, num_eval_workers=4)