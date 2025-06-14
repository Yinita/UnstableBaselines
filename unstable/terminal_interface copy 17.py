"""
Rich-powered live dashboard for UnstableBaselines – v3
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
    # REFRESH_SEC: float = 10.0         # UI refresh period (s)

    # ...........................................................................
    def __init__(self, tracker, model_pool, step_buffer=None) -> None:
        self.tracker, self.model_pool, self.step_buffer = (tracker, model_pool, step_buffer)
        self.console = Console()
        self._latest_stats: Dict[str, Any] = {}


        # ── NVML init (optional) ──
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

        full_w   = self.console.size.width
        graph_w  = int(full_w * 0.75)           # right-hand graph width

        for gid in range(self.gpu_count):
            self.power_graphs[gid] = BufferedHistoryGraph(
                100.0, width=graph_w, height=3,
                format=lambda v: f"{v:.0f}%", dynamic_bound=False
            )
            self.tok_graphs[gid]   = BufferedHistoryGraph(
                1e4,  width=graph_w, height=3,
                format=lambda v: f"{v:.0f}", dynamic_bound=False, upsidedown=True
                # format=lambda v: f"{v:.0f}", dynamic_bound=True, upsidedown=True
            )
            self.mem_graphs[gid]   = BufferedHistoryGraph(
                100.0, width=graph_w, height=3,
                format=lambda v: f"{v:.0f}%", dynamic_bound=False
            )

    # ────────────────────────── helper RPCs ─────────────────────────────────────
    async def _system_stats(self) -> Dict[str, Any]:
        # self.console.log(f"NVML ok? {bool(self.pynvml)}, GPUs seen: {self.gpu_count}")

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
                    "used":  m.used  / 1e9,
                    "total": m.total / 1e9,
                    "power": power,
                    "limit": limit,
                    "pct":   pct,
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

    # ────────────────────────── visual helpers ────────────────────────────────
    def _colour_for_util(self, pct: float) -> str:
        """Idle = red; heavy = green."""
        return "green" if pct >= 80 else "yellow" if pct >= 40 else "red"

    # single GPU block ----------------------------------------------------------
    def _gpu_block(self, g: Dict[str, float], tok_rate: float | None) -> Panel:
        gid      = g["id"]
        mem_pct  = g["used"] / g["total"] * 100
        util_pct = g["pct"]
        role     = "Actor" if tok_rate else "Learner"
        colour   = self._colour_for_util(util_pct)

        # update graphs
        self.power_graphs[gid].add(util_pct)
        if tok_rate: self.tok_graphs[gid].add(tok_rate)
        else:        self.mem_graphs[gid].add(mem_pct)

        # build left-hand info (≈ 15 %)
        info_lines = [
            f"GPU{gid}",
            role,
            f"Memory  {mem_pct:>3.0f}%",
            f"Power    {util_pct:>3.0f}%",
        ]
        left = Panel(
            Text("\n".join(info_lines)),
            box=box.MINIMAL, padding=(0,1) #, style=colour
        )

        # right-hand graphs
        width_cut = int(self.console.size.width * 0.70)

        power_str  = "\n".join(line[-width_cut:] for line in self.power_graphs[gid].graph)
        lower_str  = (
            "\n".join(line[-width_cut:] for line in self.tok_graphs[gid].graph)
            if tok_rate else
            "\n".join(line[-width_cut:] for line in self.mem_graphs[gid].graph)
        )

        # legend_top    = f"{util_pct:.0f}% Power util."
        # legend_bottom = (f"{tok_rate:.0f} tok/s" if tok_rate else f"{mem_pct:.0f}% Memory")
        legend_top    = f"{util_pct:3.0f}% Power util."
        legend_bottom = (
            f"{tok_rate:5.0f} tok/s" if tok_rate else f"{mem_pct:3.0f}% Memory"
        )

        graphs = Group(Text(legend_top, style=colour), Text(power_str, style=colour), Text("-" * width_cut, style="dim"), Text(legend_bottom, style=colour), Text(lower_str, style=colour))
        body = Columns([left, graphs], expand=True, equal=False, align="left")
        return Panel(body, title=f"GPU{gid}", box=box.SQUARE, style=colour)

    # stack of GPU blocks -------------------------------------------------------
    def _gpu_panel(self, sys: Dict[str, Any]) -> Panel:
        # print(sys)
        blocks = [
            self._gpu_block(g, sys["gpu_tok"].get(g["id"], 0.0))
            for g in sys["gpus"]
        ]
        return Panel(Group(*blocks) if blocks else Text("no GPUs"), title="GPU Performance", box=box.DOUBLE)

    async def _fetch_loop(self, interval: float = 1.0):
        while True:
            try:
                stats = await self._system_stats()
                self._latest_stats = stats
            except Exception as e:
                self.console.log(f"[red]Error during stat fetch: {e}")
            await asyncio.sleep(interval)

    # ─────────────────────────────── main loop ────────────────────────────────
    # async def run(self):
    #     layout = Layout()
    #     layout.split_column(
    #         Layout(name="top_half"),
    #         Layout(name="gpu_half", ratio=1),
    #     )
    #     layout["top_half"].split_row(
    #         Layout(name="blank_left"),
    #         Layout(name="blank_right"),
    #     )

    #     with Live(layout, refresh_per_second=1): # / self.REFRESH_SEC):
    #         while True:
    #             sys = await self._system_stats()
    #             layout["gpu_half"].update(self._gpu_panel(sys))
    #             # await asyncio.sleep(self.REFRESH_SEC)

    async def run(self):
        layout = Layout()
        layout.split_column(
            Layout(name="top_half"),
            Layout(name="gpu_half", ratio=1),
        )
        layout["top_half"].split_row(
            Layout(name="blank_left"),
            Layout(name="blank_right"),
        )

        # Launch background fetcher
        asyncio.create_task(self._fetch_loop())

        with Live(layout, refresh_per_second=1.0):  # display update speed
            while True:
                if self._latest_stats:
                    layout["gpu_half"].update(self._gpu_panel(self._latest_stats))
                await asyncio.sleep(0.2)  # light refresh wait
