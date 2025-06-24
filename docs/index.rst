Welcome to the Unstable Baselines docs!
===================================

Unstable Baselines is an Async, Online, Multi-Turn, Multi-Agent RL library for training reasoning models on TextArena games.


What is Unstable Baselines?
-----------------------

**Unstable Baselines** is a lightweight reinforcement‑learning research harness focused on *self‑play* for text‑based games. It couples:

* **Ray** – easy, elastic distributed execution.
* **vLLM** – high‑throughput inference with LoRA hot‑swapping.
* **TextArena** – a growing suite of competitive text games.

The goal is to iterate **quickly** on small language models (< 8B params) and benchmark new ideas in *reasoning and agentic behaviour*.
Since multiple recent papers showed the sufficiency of LoRA for reasoning tuning, and the fact that opponent sampling for self-play strategies beyond mirror self-play work best when using LoRA weights (since vLLM allows for hot-swapping), we built UnstableBaselines as a LoRA first RL library. 
We tried to keep the code as straight forward as possible. It is currently around **1.2K** lines long and semi-readable. 

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   quickstart/index



Citation
--------

|DOI Badge| |DOI Link|

If you use **UnstableBaselines** in your research, please cite:

.. code-block:: bibtex

   @software{guertler_leon_2025_15719271,
     author={Guertler, Leon and Grams, Tim and Zichen, Liu and Cheng, Bobby},
     title={{UnstableBaselines}},
     month=jun,
     year=2025,
     publisher={Zenodo},
     version={0.1.0},
     doi={10.5281/zenodo.15719271},
     url={https://doi.org/10.5281/zenodo.15719271}
   }

.. |DOI Badge| image:: https://zenodo.org/badge/975887163.svg
   :target: https://doi.org/10.5281/zenodo.15719270
   :alt: DOI

.. |DOI Link| raw:: html

   <a href="https://doi.org/10.5281/zenodo.15719270">https://doi.org/10.5281/zenodo.15719270</a>
