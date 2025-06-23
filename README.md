
<div align="center">

<div style="line-height: 1.0;">
  <h1 style="font-size: 3em; font-weight: bold; margin: 0; border: none; padding: 0;">UnstableBaselines</h1>
</div>



An Async, Online, Multi-Trun, Multi-Agent RL library for training reasoning models on TextArena games.

<h3>

[Documentation](https://github.com/LeonGuertler/UnstableBaselines/blob/main/unstable/documentation.md)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/LeonGuertler/UnstableBaselines)](https://github.com/LeonGuertler/UnstableBaselines/stargazers)
[![Discord](https://img.shields.io/discord/1257951838322561075?color=7289DA&label=Discord)](https://discord.gg/KPacHzK23e)
[![TextArena](https://img.shields.io/badge/TextArena-v0.6.9-181717)](https://github.com/LeonGuertler/TextArena)

</div>

---
> **Work in progress — interfaces will change.**

## Updates
* 23/06/2025: Early release of the pip package (`pip install unstable-rl`)
* 22/06/2025: Early release of the code base


## Introduction
UnstableBaselines is an Async-, Online-, Multi-Agent RL library focused on simplicity and hackability. Since multiple recent papers showed the sufficiency of LoRA for reasoning tuning, and the fact that opponent sampling for self-play strategies beyond mirror self-play work best when using LoRA weights (since vLLM allows for hot-swapping), we built UnstableBaselines as a LoRA first RL library. We tried to keep the code as straight forward as possible. It is currently around **1.2K** lines long and semi-readable. The main focus of unstable-baselines is to enable fast prototyping/research. For something a bit more production ready we recommend to use [oat](https://github.com/sail-sg/oat) or [verifiers](https://github.com/willccbb/verifiers).



## Key Features
* **Asynchronous collection & learning** – actors generate data while learners train.
* **Multi‑agent, multi‑turn** focus with self‑play or fixed opponents.
* **LoRA‑first** fine‑tuning workflow for fast, lightweight updates.
* **Composable reward transforms** at step, game, and sampling stages.


## Structure
<div align="center">

<pre style="text-align: left;">
                                                ┌───────────────┐
                                                │               │
                                                │   Algorithm   │
                                                │               │
                                                └───────────────┘
                                                        ▲        
                                                        │ Get Loss &
                                                        │ update weights
                                                        ▼
    ┌───────────────┐                           ┌───────────────┐
    │               │    Register new lora      │               │
    │   Model Pool  │◀──────────────────────────│    Learner    │
    │               │       checkpoint          │               │
    └───────────────┘                           └───────────────┘
           ▲ │                                         ▲ │ 
           │ │ Sample                        If enough │ │ Check if enough
    Update │ │ Opponent                     data, pull │ │ data for training
 Trueskill │ │                          the next batch │ │ is available
           │ ▼                                         │ ▼
    ┌───────────────┐                           ┌───────────────┐
    │               │     Process and store     │               │
    │   Collector   │──────────────────────────▶│   StepBuffer  │
    │               │  collected Trajectories   │               │
    └───────────────┘                           └───────────────┘
           ▲ │
           │ │ Maintain
    return │ │ Pool of 
Trajectory │ │ n parallel
           │ │ workers
           │ ▼
     ┌─────────────┐
     │  run_game() │
     │  train\eval │
     └─────────────┘

</pre>

</div>

## Installation

```bash
# build TextArena v0.6.9 (until it’s on PyPI)
git clone https://github.com/LeonGuertler/TextArena.git
cd TextArena
git checkout v0.6.9
python3 setup.py sdist bdist_wheel
pip install -e .
cd ..

# install UnstableBaselines
pip install unstable-rl
```

## Example
To get you started, in this short example we will run you through the process of training `Qwen3-1.7B-Base` via **mirror self-play** on _SimpleTak_ and evaluating it against `google/gemini-2.0-flash-lite-001` on _SimpleTak_ and _KuhnPoker_. We will be running the experiments on 3xRTX6000 ada. If you are limited to 24gb of vRam, you can reduce the `MAX_TRAIN_SEQ_LEN` to around _2500_; this means that the model will only be trained on the first 2500 prompt+answer tokens, but can still generate answer that are longer than that. Since (in our experience) models tend to shorten their reasoning throughout training, this works very well.


### Training script

```python
import ray, unstable
import unstable.reward_transformations as retra

ray.init(namespace="unstable")

tracker = unstable.Tracker.options(name="Tracker").remote(run_name="demo", wandb_project="UB")

step_buffer = unstable.StepBuffer.options(name="StepBuffer").remote(
    max_buffer_size=768, 
    tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

model_pool = unstable.ModelPool.options(name="ModelPool").remote(sample_mode="mirror", max_active_lora=3, tracker=tracker)
ray.get(model_pool.add_checkpoint.remote(path=None, iteration=-1)) # set initial checkpoint as no LoRA

lora_cfg = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
collector = unstable.Collector.options(name="Collector").remote(
    num_actors=2, 
    step_buffer=step_buffer, 
    model_pool=model_pool, 
    tracker=tracker,
    vllm_config={
        "model_name": "Qwen/Qwen3-1.7B-base", 
        "max_parallel_seq": 128,
        "max_tokens": 4096, 
        "max_loras": 5, 
        "lora_config": lora_cfg, 
        "max_model_len": 8192
    },
    training_envs=[("SimpleTak-v0-train", 2, "qwen3-zs")], # (env-id, num players, prompt template)
    evaluation_envs=[("SimpleTak-v0-train", 2, "qwen3-zs"), ("KuhnPoker-v0-train", 2, "qwen3-zs")],
    evaluation_opponent="google/gemini-2.0-flash-lite-001",
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
    lora_cfg=lora_cfg,
    activation_checkpointing=False,
    gradient_checkpointing=False,
    max_train_len=None, # always train on the full sequence
    max_generation_len=4096, # important for Dr. GRPO
)

# start the collection and training loops
collector.collect.remote(num_workers=384, num_eval_workers=16)  
ray.get(learner.train.remote(200)) # total update steps
```
In a Nutshell, the **Collector** will maintain `384` and `16` in parallel running collection and evaluation games (respectively). Whenever a game finishes, the trajectory is passed to the **StepBuffer** and a new game is started. The **StepBuffer** splits each trajectory into steps and applies the specified reward transformations (on the game and step level first; and batch level once the Learner pulls the next batch).

The **Learner** will periodically (once every 0.2 seconds) check if the **StepBuffer** has accumulated enough data for training. If so, it'll request a full training batch from the **StepBuffer**, train on the data, and push the new set of LoRA weights to the **ModelPool**.

The **Collector** will keep collecting episodes until the Learner tells it to stop (in this case, after `200` update steps).


### Monitoring Progress
If you want to monitor key metrics (in addition to logging them via W&B) during training you can run the following command in a seperate terminal:
```bash
python3 -m unstable.terminal_interface
```
The rendered interface will currently look something like this: (please not that it might change in the future as UnstableBaselines is very much still under development)
![](https://github.com/LeonGuertler/UnstableBaselines/blob/main/_docs/terminal_interface.gif)
The .gif doesn't do it justice, looks nice when you run it yourself haha.

### Results
Since we set `num_eval_workers=16`, throughout training there are always 16 eval games running in parallel (using the most recent lora checkpoint). Running 200 learner steps took a total of ~12h on the 3xRTX6000 ada setup we used.
![Results (light)](https://raw.githubusercontent.com/LeonGuertler/UnstableBaselines/main/_docs/results_plot_light.png#gh-light-mode-only)
![Results (dark)](https://raw.githubusercontent.com/LeonGuertler/UnstableBaselines/main/_docs/results_plot_dark.png#gh-dark-mode-only)


As can be seen in the plots the Win-Rate against a fixed opponent (in this case `google/gemini-2.0-flash-lite-001`) improves significantly for both the training and evaluation environment, showing that at least some of learned reasoning patterns generalize to other tasks and problems.



## Collaboration
Developed in partnership with [PlasticLabs](https://plasticlabs.ai/).


## Papers
We built this code-base as part of our research on self-play for reasoning models on text based games. We hope to finish and release both papers (one focused on the paradigm and one focused on the "scaling-laws" and analysis thereof) within the next couple of weeks!


## Citation

If you use **UnstableBaselines** in your research, please cite:

TODO
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567) -->

