#!/usr/bin/env python3
"""
PPO 训练示例
基于 A2C 示例修改，展示如何使用 PPO 算法进行训练
"""

import unstable

# 配置训练环境
train_envs = [
    unstable.TrainEnvSpec(
        env_id="TicTacToe",
        num_players=2,
        prompt_template="tictactoe_basic"
    ),
    unstable.TrainEnvSpec(
        env_id="ConnectFour", 
        num_players=2,
        prompt_template="connect_four_basic"
    )
]

# 配置评估环境
eval_envs = [
    unstable.EvalEnvSpec(
        env_id="TicTacToe",
        num_players=2, 
        prompt_template="tictactoe_basic"
    )
]

# 构建 PPO 训练运行
run = unstable.build(
    model_name="microsoft/DialoGPT-medium",
    train_envs=train_envs,
    eval_envs=eval_envs,
    algorithm="ppo",  # 使用 PPO 算法
    batch_size=64,
    mini_batch_size=8,
    learning_rate=1e-5,
    max_generation_len=512,
    max_train_len=1024,
    gradient_clipping=1.0,
    wandb_project="UnstableBaselines-PPO"
)

# 开始训练
if __name__ == "__main__":
    print("开始 PPO 训练...")
    run.start(
        learning_steps=100,
        num_collection_workers=32,
        num_eval_workers=8
    )
    print("PPO 训练完成!")
