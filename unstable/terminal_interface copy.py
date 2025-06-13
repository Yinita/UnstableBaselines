"""Rich‑powered live dashboard for UnstableBaselines.

Run this in a **separate Ray actor** (TerminalInterface) so it does not block
learner or collector threads.  It queries other actors every few hundred
milliseconds and paints four panels:

1. Learner         – recent loss / tokens‑sec / grad‑norm
2. Inference Queue – per‑LoRA queued + running sequences
3. Model‑Pool      – TrueSkill μ / σ and top‑K match‑ups
4. System          – buffer size, GPU VRAM, CPU %, etc.
"""
from __future__ import annotations

import asyncio
import psutil
from datetime import datetime
from typing import Dict, Any, List

import ray
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# ---------------------------------------------------------------------------
# helpers to format panels
# ---------------------------------------------------------------------------

def _learner_panel(stats: Dict[str, Any]) -> Panel:
    body = (
        f"[bold]step[/]: {stats.get('step', '-') }\n"
        f"loss      : {stats.get('loss', 0):.4f}\n"
        f"ppl       : {stats.get('ppl', 0):.1f}\n"
        f"tokens/sec: {stats.get('tok_s', 0):.0f}\n"
        f"grad‑norm : {stats.get('grad_norm', 0):.1f}"
    )
    return Panel(body, title="Learner", box=box.ROUNDED)


def _inference_panel(q: Dict[str, Dict[str, int]]) -> Panel:
    tbl = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    tbl.add_column("LoRA")
    tbl.add_column("queued", justify="right")
    tbl.add_column("running", justify="right")
    for name, meta in q.items():
        tbl.add_row(name, str(meta["queue"]), str(meta["running"]))
    return Panel(tbl, title="Inference", box=box.ROUNDED)


def _pool_panel(pool: Dict[str, Any]) -> Panel:
    tbl = Table(box=None)
    tbl.add_column("metric")
    tbl.add_column("value", justify="right")
    for k, v in pool.items():
        tbl.add_row(k, f"{v}")
    return Panel(tbl, title="Model‑Pool", box=box.ROUNDED)


def _system_panel(sys_stats: Dict[str, Any]) -> Panel:
    text = Text()
    text.append(f"CPU %    : {sys_stats['cpu']}\n")
    text.append(f"RAM used : {sys_stats['ram']:.1f} GB\n")
    text.append(f"GPU mem  : {sys_stats['vram']:.1f} GB\n")
    text.append(f"buffer sz: {sys_stats['buffer']}\n")
    text.append(f"updated  : {datetime.now().strftime('%H:%M:%S')}")
    return Panel(text, title="System", box=box.ROUNDED)

# ---------------------------------------------------------------------------
# Remote actor
# ---------------------------------------------------------------------------

# @ray.remote
class TerminalInterface:
    """Continuously renders live stats in the terminal using Rich."""

    REFRESH_SEC: float = 0.5

    def __init__(
        self,
        tracker,            # ray ActorHandle (unstable.Tracker)
        model_pool,         # ray ActorHandle
        actors: List,       # list[ActorHandle] of VLLMActor
        step_buffer=None,   # optional ActorHandle for buffer stats
    ) -> None:
        self.tracker = tracker
        self.model_pool = model_pool
        self.actors = actors
        self.step_buffer = step_buffer

    # ---------------------------------------------------------------------
    # polling helpers – each returns a small dict serialisable by Ray
    # ---------------------------------------------------------------------
    async def _learner_stats(self) -> Dict[str, Any]:
        try:                return ray.get(self.tracker.get_latest_learner_metrics.remote())
        except Exception:   return {}

    async def _inference_stats(self) -> Dict[str, Any]:
        out: Dict[str, Dict[str, int]] = {}
        for a in self.actors:
            try:
                st = ray.get(a.get_queue_stats.remote())
                out.update(st)  # returns {lora_name: {queue: int, running: int}}
            except Exception:
                continue
        return out

    async def _pool_stats(self) -> Dict[str, Any]:
        try:
            snap = ray.get(self.model_pool.get_snapshot.remote())
            return {"ckpts": snap.get("num_ckpts"), "μ±σ": f"{snap['mu']:.1f}±{snap['sigma']:.1f}",}
        except Exception:
            return {}

    async def _system_stats(self) -> Dict[str, Any]:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().used / 1e9
        vram = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            vram = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9
        except Exception:
            pass
        buf = 0
        if self.step_buffer:
            try:
                buf = ray.get(self.step_buffer.size.remote())
            except Exception:
                pass
        return {"cpu": cpu, "ram": ram, "vram": vram, "buffer": buf}

    # ---------------------------------------------------------------------
    # main loop
    # ---------------------------------------------------------------------
    async def run(self):
        layout = Layout()
        layout.split_row(Layout(name="learner"), Layout(name="inference"), Layout(name="pool"), Layout(name="system"))
        print("JUST BEFORE LOOP")
        with Live(layout, refresh_per_second=1 / self.REFRESH_SEC):
            print("INSIDE IF ")
            while True:
                # gather all stats concurrently
                learner, inf, pool, sys = await asyncio.gather(self._learner_stats(), self._inference_stats(), self._pool_stats(), self._system_stats())
                print("got data")
                layout["learner"].update(_learner_panel(learner))
                layout["inference"].update(_inference_panel(inf))
                layout["pool"].update(_pool_panel(pool))
                layout["system"].update(_system_panel(sys))
                await asyncio.sleep(self.REFRESH_SEC)
