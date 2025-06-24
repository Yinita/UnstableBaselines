Welcome to the **Unstable Baselines** docs!
===================================

`Unstable Baselines <https://github.com/LeonGuertler/UnstableBaselines>`\_ is an Async, Online, Multi-Turn, Multi-Agent RL library for training reasoning models on `TextArena <https://github.com/LeonGuertler/TextArena>`\_ games.

What is Unstable Baselines?
-----------------------

**Unstable Baselines** is a lightweight reinforcement-learning research harness focused on *self-play* for text-based games. It couples:

* **Ray** - easy, elastic distributed execution.
* **vLLM** - high-throughput inference with LoRA hot-swapping.
* **TextArena** - a growing suite of competitive text games.

The goal is to iterate **quickly** on small language models (< 8B params) and benchmark new ideas in *reasoning and agentic behaviour*.
Since multiple recent papers showed the sufficiency of LoRA for reasoning tuning, and the fact that opponent sampling for self-play strategies beyond mirror self-play work best when using LoRA weights (since vLLM allows for hot-swapping), we built UnstableBaselines as a LoRA first RL library. 
We tried to keep the code as straight forward as possible. It is currently around **1.2K** lines long and semi-readable. 

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/quickstart
   guide/architecture
   guide/usage
   guide/troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog
   license


Citing Unstable Baselines
-------------------------

.. code-block:: bibtex

    @software{guertler_2025_unstablebaselines,
      author  = {Guertler, Leon and Grams, Tim and Liu, Zichen and Cheng, Bobby},
      title   = {Unstable Baselines},
      year    = {2025},
      version = {0.1.0},
      doi     = {10.5281/zenodo.15719271},
      url     = {https://github.com/LeonGuertler/UnstableBaselines}
    }


Contributing
------------

Bugs, feature requests, and PRs are **very** welcome — the code is research-grade, not production-grade.  
See `CONTRIBUTING.md <https://github.com/LeonGuertler/UnstableBaselines/blob/main/CONTRIBUTING.md>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
