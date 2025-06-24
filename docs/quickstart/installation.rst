# Installation

## Prerequisites

* Python ≥ 3.10
* CUDA-enabled GPUs (at least 2x 24GB VRAM)

## Steps

.. code-block:: bash

# 1. Install TextArena from source (until on PyPI)

git clone [https://github.com/LeonGuertler/TextArena.git](https://github.com/LeonGuertler/TextArena.git)
cd TextArena && git checkout v0.6.9
python3 -m pip install -e .
cd ..

# 2. Install UnstableBaselines + core deps

python3 -m pip install unstable-rl  # pulls vLLM, Ray, PEFT, …
