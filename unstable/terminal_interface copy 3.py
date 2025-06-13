"""Rich‑powered live dashboard for UnstableBaselines.

Run this in a **separate script** or actor so it does not block
learner or collector threads. It queries remote actors every few hundred
milliseconds and paints four panels in a 2×2 grid:

1. Learner         – recent loss / tokens‑sec / grad‑norm
2. Inference Queue – per‑LoRA queued + running sequences
3. Model‑Pool      – TrueSkill μ / σ and top‑K match‑ups
4. System          – CPU/RAM/Buffer + GPU mem/power + dynamic sparkline
"""
from __future__ import annotations

import asyncio
import psutil
from datetime import datetime
from typing import Dict, Any, List
from collections import deque

import ray
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
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
        tbl.add_row(name, str(meta.get("queue",0)), str(meta.get("running",0)))
    return Panel(tbl, title="Inference", box=box.ROUNDED)


def _pool_panel(pool: Dict[str, Any]) -> Panel:
    tbl = Table(box=None)
    tbl.add_column("metric")
    tbl.add_column("value", justify="right")
    for k, v in pool.items():
        tbl.add_row(k, f"{v}")
    return Panel(tbl, title="Model‑Pool", box=box.ROUNDED)


# ---------------------------------------------------------------------------
# TerminalInterface class
# ---------------------------------------------------------------------------

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
        # histories: map gpu_id -> deque of recent power-usage percentages
        self.gpu_histories: Dict[int, deque[float]] = {}

    # ---------------------------------------------------------------------
    # polling helpers
    # ---------------------------------------------------------------------
    async def _learner_stats(self) -> Dict[str, Any]:
        try:
            return await self.tracker.get_latest_learner_metrics.remote()
        except Exception:
            return {}

    async def _inference_stats(self) -> Dict[str, Any]:
        out: Dict[str, Dict[str, int]] = {}
        for a in self.actors:
            try:
                st = await a.get_queue_stats.remote()
                out.update(st)
            except Exception:
                continue
        return out

    async def _pool_stats(self) -> Dict[str, Any]:
        try:
            snap = await self.model_pool.get_snapshot.remote()
            return {"ckpts": snap.get("num_ckpts"),
                    "μ±σ": f"{snap['mu']:.1f}±{snap['sigma']:.1f}"}
        except Exception:
            return {}

    async def _system_stats(self) -> Dict[str, Any]:
        # CPU / RAM / buffer
        cpu = psutil.cpu_percent()
        vm = psutil.virtual_memory()
        ram_used = vm.used / 1e9
        ram_pct  = vm.percent
        buf = 0
        if self.step_buffer:
            try:
                buf = await self.step_buffer.size.remote()
            except Exception:
                pass

        # per-GPU power & memory
        gpus: List[Dict[str, Any]] = []
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(min(count, 8)):
                h     = pynvml.nvmlDeviceGetHandleByIndex(i)
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                pct   = (power / limit * 100) if limit > 0 else 0.0

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
                used  = mem_info.used  / 1e9
                total = mem_info.total / 1e9

                gpus.append({
                    "id":    i,
                    "used":  used,
                    "total": total,
                    "power": power,
                    "limit": limit,
                    "pct":   pct,
                })
        except Exception:
            pass

        return {
            "cpu":      cpu,
            "ram_used": ram_used,
            "ram_pct":  ram_pct,
            "buffer":   buf,
            "gpus":     gpus,
        }

    # ---------------------------------------------------------------------
    # sparkline generation
    # ---------------------------------------------------------------------
    def _char_for_pct(self, pct: float) -> str:
        # map percent to block height or ascii
        if pct < 10:
            return ' '
        elif pct < 20:
            return '.'
        elif pct < 40:
            return '-'
        elif pct < 60:
            return '='
        elif pct < 80:
            return '+'
        else:
            return '#'

    def _make_spark(self, hist: deque[float], width: int) -> str:
        # ensure history deque maxlen = width
        if hist.maxlen != width:
            old = list(hist)
            hist = deque(old[-width:], maxlen=width)
        # pad front if short
        padded = [0.0] * (width - len(hist)) + list(hist)
        # translate to chars
        chars = [self._char_for_pct(p) for p in padded]
        # split into two rows
        half = width // 2
        return ''.join(chars[:half]) + '\n' + ''.join(chars[half:])

    # ---------------------------------------------------------------------
    # render helper – equal-width columns + sparkline
    # ---------------------------------------------------------------------
    def _system_panel(self, sys_stats: Dict[str, Any]) -> Panel:
        console = Console()
        term_w = console.size.width
        # total panels = 1 CPU + len(gpus)
        total = 1 + len(sys_stats.get("gpus", []))
        # approximate single panel width (subtract borders)
        panel_w = max((term_w // total) - 4, 10)

        # CPU/RAM/Buffer sub-panel
        cpu_text = Text()
        cpu_text.append(f"CPU %    : {sys_stats['cpu']:.1f}\n")
        cpu_text.append(f"RAM used : {sys_stats['ram_used']:.1f}/{sys_stats['ram_pct']:.0f}% GB\n")
        cpu_text.append(f"buffer   : {sys_stats['buffer']}\n")
        cpu_panel = Panel(cpu_text, title="CPU / RAM / Buffer", box=box.ROUNDED)

        # GPU panels
        gpu_panels: List[Panel] = []
        for gpu in sys_stats.get("gpus", []):
            idx   = gpu["id"]
            usage = gpu["pct"]
            # update history
            hist = self.gpu_histories.setdefault(idx, deque(maxlen=panel_w))
            hist.append(usage)
            # border color by usage
            style = "red" if usage < 50 else "yellow" if usage < 75 else "green"
            text = Text()
            text.append(f"Mem   : {gpu['used']:.1f}/{gpu['total']:.1f} GB\n")
            text.append(f"Power : {gpu['power']:.0f}/{gpu['limit']:.0f} W ({usage:.0f}%)\n")
            # sparkline two rows
            spark = self._make_spark(hist, panel_w)
            text.append(spark)
            gpu_panels.append(Panel(text, title=f"GPU {idx}", box=box.ROUNDED, border_style=style))

        # assemble columns
        all_panels = [cpu_panel] + gpu_panels
        cols = Columns(all_panels, equal=True, expand=True)
        return Panel(cols, title="System", box=box.DOUBLE)

    # ---------------------------------------------------------------------
    # main loop – 2×2 grid
    # ---------------------------------------------------------------------
    async def run(self):
        layout = Layout()
        layout.split_column(
            Layout(name="upper"),
            Layout(name="lower"),
        )
        layout["upper"].split_row(Layout(name="learner"), Layout(name="inference"))
        layout["lower"].split_row(Layout(name="pool"), Layout(name="system"))

        with Live(layout, refresh_per_second=1 / self.REFRESH_SEC):
            while True:
                learner, inf, pool, sys = await asyncio.gather(
                    self._learner_stats(),
                    self._inference_stats(),
                    self._pool_stats(),
                    self._system_stats(),
                )
                layout["learner"].update(_learner_panel(learner))
                layout["inference"].update(_inference_panel(inf))
                layout["pool"].update(_pool_panel(pool))
                layout["system"].update(self._system_panel(sys))
                await asyncio.sleep(self.REFRESH_SEC)
