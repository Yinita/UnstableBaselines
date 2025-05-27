python3 unstable.py \
    --model_name "Qwen/Qwen3-4B-base" \
    --train_env_id "SimpleTak-v0-train:2" \
    --eval_env_id "SimpleTak-v0-train:2,Nim-v0-train:2,UltimateTicTacToe-v0-train:2"\
    --wandb \
    --num_actors 7 \
    --num_learners 1 \
    --lr 1e-4 \
    --batch_size 384 \
    --gradient_accumulation_steps 384 \
    --max_tokens 4096 \
    --gradient_checkpointing \
    --bf16_training \
    --num_collection_workers 384 \
    --num_evaluation_workers 32 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --self_play_opponent_lag_lower 3 \
    --self_play_opponent_lag_upper 9 \
    --format_reward_think 1.5 \
    --format_reward_valid_move 1.0 \
    --format_penalty_invalid_move -1.0\
    --observation_format_template "qwen3"


