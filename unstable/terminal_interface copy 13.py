"""
Rich-powered live dashboard for UnstableBaselines – *v2*
"""

from __future__ import annotations

import asyncio, psutil, time
from typing import Dict, Any, List

import ray
from rich.console import Console, Group
from rich.live   import Live
from rich.layout import Layout
from rich.panel  import Panel
from rich.text   import Text
from rich.columns import Columns
from rich import box
from nvitop.tui.library import BufferedHistoryGraph


# ────────────────────────────────────────────────────────────────────────────────
class TerminalInterface:
    REFRESH_SEC: float = 0.5        # UI refresh period (s)

    # ...........................................................................
    def __init__(self, tracker, model_pool, step_buffer=None) -> None:
        self.tracker     = tracker
        self.model_pool  = model_pool
        self.step_buffer = step_buffer
        self.console     = Console()

        # ── (optional) NVML initialisation ──
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml    = pynvml
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            self.pynvml, self.gpu_count = None, 0

        # ── history graphs ──
        self.power_graphs: Dict[int, BufferedHistoryGraph] = {}
        self.tok_graphs  : Dict[int, BufferedHistoryGraph] = {}
        self.mem_graphs  : Dict[int, BufferedHistoryGraph] = {}

        width = self.console.size.width - 10          # graph width

        for gid in range(self.gpu_count):
            self.power_graphs[gid] = BufferedHistoryGraph(
                100.0, width=width, height=3,
                format=lambda v: f"{v:.0f}%", dynamic_bound=False
            )
            self.tok_graphs[gid]   = BufferedHistoryGraph(
                1e4,   width=width, height=3,
                format=lambda v: f"{v:.0f}", dynamic_bound=True, upsidedown=True
            )
            self.mem_graphs[gid]   = BufferedHistoryGraph(
                100.0, width=width, height=3,
                format=lambda v: f"{v:.0f}%", dynamic_bound=False
            )

    # ────────────────────────────────────────────────────────────────────────────
    # RPC helpers
    async def _system_stats(self) -> Dict[str, Any]:
        import psutil
        cpu = psutil.cpu_percent()
        vm  = psutil.virtual_memory()

        buf = 0
        if self.step_buffer:
            try:                buf = await self.step_buffer.size.remote()
            except Exception:   pass

        gpus = []
        if self.pynvml:
            for gid in range(self.gpu_count):
                h     = self.pynvml.nvmlDeviceGetHandleByIndex(gid)
                power = self.pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                limit = self.pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                pct   = power / limit * 100 if limit else 0.0
                m     = self.pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append({
                    "id": gid,
                    "used": m.used  / 1e9,
                    "total": m.total / 1e9,
                    "power": power,
                    "limit": limit,
                    "pct": pct,
                })

        try:    gpu_tok = await self.tracker.get_gpu_tok_rates.remote()
        except Exception:
            gpu_tok = {}

        return {
            "cpu": cpu,
            "ram_used": vm.used / 1e9,
            "ram_pct": vm.percent,
            "buffer": buf,
            "gpus": gpus,
            "gpu_tok": gpu_tok,
        }

    # ────────────────────────────────────────────────────────────────────────────
    # visual helpers
    def _colour_for_util(self, pct: float) -> str:
        """Unconventional: idle = red, heavy = green."""
        return "green" if pct >= 80 else "yellow" if pct >= 40 else "red"

    # graph block for one GPU -----------------------------------------------------
    def _gpu_block(self, g: Dict[str, float], tok_rate: float | None) -> Panel:
        gid       = g["id"]
        mem_pct   = g["used"] / g["total"] * 100
        util_pct  = g["pct"]

        # update graphs
        self.power_graphs[gid].add(util_pct)
        if tok_rate:
            self.tok_graphs[gid].add(tok_rate)
        else:
            self.mem_graphs[gid].add(mem_pct)

        # pick colour by *power util*
        style = self._colour_for_util(util_pct)

        # string graphs
        power_str = "\n".join(self.power_graphs[gid].graph)
        lower_graph = (
            "\n".join(self.tok_graphs[gid].graph)
            if tok_rate else
            "\n".join(self.mem_graphs[gid].graph)
        )

        # legends
        top_legend    = f"{util_pct:>3.0f}% util"
        bottom_legend = f"{tok_rate:>4.0f} tok/s" if tok_rate else f"{mem_pct:>3.0f}% mem"

        # build inner box
        inner = Group(
            Text(top_legend, style=style),
            Text(power_str,   style=style),
            Text("-" * (len(self.power_graphs[gid].graph[0]) or 20), style="dim"),
            Text(bottom_legend, style=style),
            Text(lower_graph,   style=style),
        )

        return Panel(inner, title=f"GPU{gid}", box=box.SQUARE, style=style)

    # whole bottom panel ---------------------------------------------------------
    def _gpu_panel(self, sys: Dict[str, Any]) -> Panel:
        blocks = [
            self._gpu_block(g, sys["gpu_tok"].get(g["id"], 0.0))
            for g in sys["gpus"]
        ]
        # stack vertically
        body = Group(*blocks) if blocks else Text("no GPUs")
        return Panel(body, title="GPU Performance", box=box.DOUBLE)

    # ────────────────────────────────────────────────────────────────────────────
    async def run(self):
        # layout – top row split; bottom row GPU panel
        layout = Layout()
        layout.split_column(
            Layout(name="top_half"),
            Layout(name="gpu_half", ratio=1),
        )
        layout["top_half"].split_row(
            Layout(name="blank_left"),
            Layout(name="blank_right"),
        )

        with Live(layout, refresh_per_second=1 / self.REFRESH_SEC):
            while True:
                sys = await self._system_stats()
                layout["gpu_half"].update(self._gpu_panel(sys))
                await asyncio.sleep(self.REFRESH_SEC)
