# unstable-baselines (WIP)
(it's calles `unstable-baselines` becuase the maintained OpenAI baselines package is called `stable-baselines`)


# Getting Started
It makes sense to start from the game-tuned checkpoint (three epochs of sft on 750 observation-actions pairs generated via R1 self-play on TicTacToe-v0). 
The checkpoint is zipped `lora_ckpt.zip`. You have to unzip it. The `run.sh` script will use this as the initial set of lora weights by default. 
Afterward you can just run `bash run.sh`. Depending on how many GPUs you have, you can set the `--num_actors` and `--num_learners`. Keep in mind that collection is much much much more expensive than lora-training (so 7-1 is prob a good ratio).

(`bash run_24gb.sh` should work well for machine with 4x24gb (vRam) gpus)

Also, this is currently using the new version of textarena (wich is on the `v0.6.9` branch). To run this code, you will have to build textaren locally:
1. `git clone https://github.com/LeonGuertler/TextArena.git`
2. `cd TextArena`
3. `git checkout v0.6.9`
4. `pip uninstall textarena` (just to make sure)
5. `python3 setup.py sdist bdist_wheel`
6. `pip install -e .`
7. all done, you are now running textarena v0.6.9 



change run name to model + envs (no player count) + opponent type + time 

### TODOs for Leon now:
- Running vs Queued requests for actor logging not working. Everything is counted as running on wandb


### TODOs for Tim:
- track action/state diversity and log it to wandb (only really possible in fixed move games like TicTacToe, etc. but probably super interesting and valuable)
- store specific checkpoints



## KIV:
- track time per complete episode
- track time per turn
- track time per generated char
- maybe dynamically adjust the number of gpus used for training/collection

# Ideas:
- increase sample weight based on SD of reward 
- somehow account for stochacisity in environment rewards (maybe somehow include reward certainty)
- dynamic GRPO rollouts (either by window or by least return confidence)


### Legacy TODOs:
- TODO add a single gpu debugging mode frfr
- TODO optimize by grouping same lora paths to same gpus
- TODO (long-term) keep pool of learning lora weights for stability
