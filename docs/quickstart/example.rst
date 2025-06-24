Example
=======

Prerequisites
------------

.. code-block:: bash

    # Prerequisites
    conda create -n unstable python=3.12 && conda activate unstable
    pip install --upgrade ray torch vllm textarena wandb rich trueskill peft transformers pynvml

    # Clone repository
    git clone https://github.com/<you>/unstable-baselines && cd unstable-baselines

Usage Example
------------

.. code-block:: python

    import ray, unstable
    import unstable.reward_transformations as retra

    ray.init(namespace="unstable")
    tracker = unstable.Tracker.options(name="Tracker").remote(run_name="demo", wandb_project="UB")
    step_buffer = unstable.StepBuffer.options(name="StepBuffer").remote(
        max_buffer_size=768,
        tracker=tracker,
        final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.WinDrawLossFormatter()]),
        step_reward_transformation=None,
        sampling_reward_transformation=None,
    )
    model_pool = unstable.ModelPool.options(name="ModelPool").remote(sample_mode="mirror", max_active_lora=3, tracker=tracker)
    ray.get(model_pool.add_checkpoint.remote(path=None, iteration=-1))
    collector = unstable.Collector.options(name="Collector").remote(
        num_actors=2,
        step_buffer=step_buffer,
        model_pool=model_pool,
        tracker=tracker,
        vllm_config={
            "model_name": "Qwen/Qwen3-1.7B-base",
            "max_parallel_seq": 64,
            "max_tokens": 2048,
            "max_loras": 4,
            "lora_config": {"lora_rank": 32},
            "max_model_len": 8192
        },
        training_envs=[("Nim-v0-train", 2, "qwen3-zs")],
        evaluation_envs=[],
    )
    learner = unstable.StandardLearner.options(num_gpus=1, name="Learner").remote(
        model_name="Qwen/Qwen3-1.7B-base",
        step_buffer=step_buffer,
        model_pool=model_pool,
        tracker=tracker,
        algorithm=unstable.algorithms.Reinforce(),
        batch_size=384,
        mini_batch_size=1,
        learning_rate=1e-5,
        grad_clip=0.2,
        lora_cfg={"lora_rank": 32},
    )
    collector.collect.remote(num_workers=256, num_eval_workers=0)
    ray.get(learner.train.remote(100))
