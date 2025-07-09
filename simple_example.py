import unstable

run = unstable.build(
    model_name = "Qwen/Qwen3-1.7B-Base",
    train_envs = [("SimpleTak-v0-train", 2, "qwen3-zs")],
    eval_envs  = [("SimpleTak-v0-train", 2, "qwen3-zs")],
    opponent_fixed = ["google/gemini-2.0-flash-lite-001"],
    opponent_strategy = "mirror", # or "match-quality", "lagged-5", …
    buffer_size = 768,
    iterations  = 200,
)
run.start()                # spins everything up
run.wait()                 # blocks until learner is done
