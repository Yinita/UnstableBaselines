python3 unstable.py \
    --model_name "Qwen/Qwen3-4B-base" \
    --train_env_id "SimpleTak-v0" \
    --eval_env_id "SimpleTak-v0,TicTacToe-v0"\
    --wandb \
    --num_actors 2 \
    --num_learners 1 \
    --lr 2e-5 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_tokens 1500 \
    --gradient_checkpointing \
    --bf16_training \
    --num_collection_workers 16 \
    --num_evaluation_workers 0 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --self_play_opponent_lag 5 \
    --format_reward_think 0.5 \
    --format_reward_valid_move 0.5 \
    --format_penalty_invalid_move -0.5\
    --observation_format_template "qwen3"


    # --batch_size 512 \
    # --gradient_accumulation_steps 512 \
    # --num_collection_workers 512 \
    # --num_evaluation_workers 32 \

    # --initial_lora_path "checkpoint-3" \