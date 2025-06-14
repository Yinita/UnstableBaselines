from __future__ import annotations

import asyncio, psutil, time
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict

import ray
from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box
from nvitop.tui.library import BufferedHistoryGraph


# ────────────────────────── helpers (still useful later) ──────────────────────────
def _fmt_size(gib: float) -> str: # format GB with one decimal
    return f"{gib:.1f}"


# ───────────────────────────────────────────────────────────────────────────────────
class TerminalInterface:
    REFRESH_SEC: float = 0.5 # UI update period

    # ..............................................................................
    def __init__(self, tracker, model_pool, step_buffer=None) -> None:
        self.tracker      = tracker
        self.model_pool   = model_pool
        self.step_buffer  = step_buffer
        self.console      = Console()

        # --- init NVML (optional) -------------------------------------------------
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except Exception:                # gracefully handle “no GPU” nodes
            self.pynvml = None
            self.gpu_count = 0

        # --- history graphs per GPU ---------------------------------------------
        self.power_graphs: Dict[int, BufferedHistoryGraph] = {}
        self.tok_graphs: Dict[int, BufferedHistoryGraph] = {}
        self.mem_graphs: Dict[int, BufferedHistoryGraph] = {}

        for gid in range(self.gpu_count):
            w = self.console.size.width // 4 - 6
            self.power_graphs[gid] = BufferedHistoryGraph(100.0, width=w, height=3, format=lambda v: f"{v:.0f}%", dynamic_bound=False)
            self.tok_graphs[gid] = BufferedHistoryGraph(1e4,  width=w, height=3, format=lambda v: f"{v:.0f}", dynamic_bound=True, upsidedown=True)
            self.mem_graphs[gid] = BufferedHistoryGraph(100.0, width=w, height=3, format=lambda v: f"{v:.0f}%", dynamic_bound=False)

    # ───────────────────────── asynchronous metric RPC calls ──────────────────────
    async def _system_stats(self) -> Dict[str, Any]:
        cpu = psutil.cpu_percent()
        vm  = psutil.virtual_memory()

        buf = 0
        if self.step_buffer:
            try:                buf = await self.step_buffer.size.remote()
            except Exception:   pass

        # GPU info
        gpus = []
        if self.pynvml:
            for gid in range(self.gpu_count):
                h = self.pynvml.nvmlDeviceGetHandleByIndex(gid)
                power = self.pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                limit = self.pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                pct = (power / limit * 100) if limit else 0.0
                m = self.pynvml.nvmlDeviceGetMemoryInfo(h)
                used = m.used  / 1e9
                total = m.total / 1e9
                gpus.append({'id': gid, 'power': power, 'limit': limit, 'pct': pct, 'used': used, 'total': total})

        # per-GPU token rates (from Tracker)
        try:                gpu_tok = await self.tracker.get_gpu_tok_rates.remote()
        except Exception:   gpu_tok = {}

        return {'cpu': cpu, 'ram_used': vm.used/1e9, 'ram_pct': vm.percent, 'buffer': buf, 'gpus': gpus, 'gpu_tok': gpu_tok}

    # ───────────────────────────────────────────────────────────────────────────────
    # GPU PERFORMANCE PANEL
    # ───────────────────────────────────────────────────────────────────────────────
    def _gpu_panel(self, sys_stats: Dict[str, Any]) -> Panel:
        gpu_tok = sys_stats['gpu_tok'] # {gpu_id: tok/s}
        blocks = []

        for g in sys_stats['gpus']:
            gid = g['id']
            util_pct = g['pct']
            mem_used = g['used']; mem_tot = g['total']
            mem_pct = mem_used / mem_tot * 100
            tok_rate = gpu_tok.get(gid, 0.0)

            # update graphs
            self.power_graphs[gid].add(util_pct)
            self.mem_graphs[gid].add(mem_pct)
            if tok_rate: self.tok_graphs[gid].add(tok_rate)

            # format header
            hdr = Text(f"GPU{gid}  MEM {_fmt_size(mem_used)}/{_fmt_size(mem_tot)} GB ({mem_pct:.0f}%)  UTIL {util_pct:.0f}%")

            # graphs
            g_power = '\n'.join(self.power_graphs[gid].graph)
            if tok_rate:
                graphs = Group(Text(g_power), Text('\n'.join(self.tok_graphs[gid].graph)))
                title = f"Actor  •  {tok_rate:.0f} tok/s"
            else:
                graphs = Group(Text(g_power), Text('\n'.join(self.mem_graphs[gid].graph)))
                title = "Learner"
            blocks.append(Panel(Group(hdr, graphs), title=title, box=box.SQUARE, width=self.console.size.width//4+2))
        return Panel(Columns(blocks, expand=True), title="GPU Performance", box=box.DOUBLE)

    # ───────────────────────────────────────────────────────────────────────────────
    async def run(self):
        layout = Layout()
        layout.split_column(Layout(name='top_half'), Layout(name='gpu_half', ratio=1))
        layout['top_half'].split_row(Layout(name='blank_left'), Layout(name='blank_right'))
        with Live(layout, refresh_per_second=1/self.REFRESH_SEC):
            while True:
                sys = await self._system_stats()
                layout['gpu_half'].update(self._gpu_panel(sys))
                await asyncio.sleep(self.REFRESH_SEC)
