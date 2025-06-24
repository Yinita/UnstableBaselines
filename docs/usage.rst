# Usage & Customisation

## Sampling strategies

* `mirror` - self-play (default)
* `lagged` - random older checkpoints
* `match-quality` - TrueSkill match-quality weighted
* `ts-dist` - opponents with *different* TrueSkill μ

Select via `sample_mode` when constructing `ModelPool`.

## Reward shaping

Three levels of transforms live under `unstable.reward_transformations`:

\================ ================================
Stage            Typical use-case
\================ ================================
Final            Convert raw game reward → win/draw/loss, apply role advantage
Step             Per-step penalties: invalid move, formatting errors
Sampling         Batch-time normalisation (e.g. z-score) or curriculum scaling
\================ ================================

Compose multiple transforms with `Compose*RewardTransforms`.

## Extending algorithms

Implement `BaseAlgo`:

.. code-block:: python

class PPO(BaseAlgo):
def prepare\_batch(self, steps): ...
def update(self, batch, scaling): ...

Pass an instance to `StandardLearner`.

## Distributed tips

* Set `RAY_memory_...` env vars to avoid OOM actor restarts.
* Pass `num_gpus` when creating learners/actors so Ray places them correctly.
* Use `ray timeline --tree` to visualise actor scheduling.