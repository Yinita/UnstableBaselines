import unstable

run = unstable.build(
    model_name = "Qwen/Qwen3-1.7B-Base",
    train_envs = [unstable.TrainEnvSpec(env_id="SimpleTak-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs")],
    eval_envs = [
        unstable.EvalEnvSpec(env_id="SimpleTak-v0-train", num_players=2, prompt_template="qwen3-zs"),
        unstable.EvalEnvSpec(env_id="KuhnPoker-v0-train", num_players=2, prompt_template="qwen3-zs")
    ]
)
run.start(learning_steps=200, num_collection_workers=256, num_eval_workers=16)
