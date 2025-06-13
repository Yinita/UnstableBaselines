"""Rich-powered live dashboard for UnstableBaselines.

Run this in a **separate script** or actor so it does not block
learner or collector threads. It queries remote actors every few hundred
milliseconds and paints four panels in a 2×2 grid:

1. Learner         – recent loss / tokens-sec / grad-norm
2. Inference Queue – per-LoRA queued + running sequences
3. Model-Pool      – TrueSkill μ / σ and top-K match-ups
4. System          – CPU/RAM/Buffer + GPU mem + power-history graphs
"""
from __future__ import annotations

import asyncio
import psutil
from datetime import datetime
from typing import Dict, Any, List

import ray
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from nvitop.tui.library import BufferedHistoryGraph
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
        f"grad-norm : {stats.get('grad_norm', 0):.1f}"
    )
    return Panel(body, title="Learner", box=box.ROUNDED)


def _inference_panel(q: Dict[str, Dict[str, int]]) -> Panel:
    tbl = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    tbl.add_column("LoRA")
    tbl.add_column("queued", justify="right")
    tbl.add_column("running", justify="right")
    for name, meta in q.items():
        tbl.add_row(name, str(meta.get("queue", 0)), str(meta.get("running", 0)))
    return Panel(tbl, title="Inference", box=box.ROUNDED)


def _pool_panel(pool: Dict[str, Any]) -> Panel:
    tbl = Table(box=None)
    tbl.add_column("metric")
    tbl.add_column("value", justify="right")
    for k, v in pool.items():
        tbl.add_row(k, f"{v}")
    return Panel(tbl, title="Model-Pool", box=box.ROUNDED)

# ---------------------------------------------------------------------------
# TerminalInterface class
# ---------------------------------------------------------------------------
class TerminalInterface:
    """Continuously renders live stats in the terminal using Rich."""

    REFRESH_SEC: float = 0.5

    def __init__(
        self,
        tracker,
        model_pool,
        actors: List,
        step_buffer=None,
    ) -> None:
        self.tracker = tracker
        self.model_pool = model_pool
        self.actors = actors
        self.step_buffer = step_buffer
        # GPU power history graphs
        self.gpu_graphs: Dict[int, BufferedHistoryGraph] = {}
        self.console = Console()

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
        cpu = psutil.cpu_percent()
        vm = psutil.virtual_memory()
        ram_used = vm.used / 1e9
        ram_pct = vm.percent
        buf = 0
        if self.step_buffer:
            try:
                buf = await self.step_buffer.size.remote()
            except Exception:
                pass

        gpus: List[Dict[str, Any]] = []
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(min(count, 8)):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                pct = (power / limit * 100) if limit > 0 else 0.0
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                used = mem.used / 1e9
                total = mem.total / 1e9
                gpus.append({
                    'id': i,
                    'used': used,
                    'total': total,
                    'power': power,
                    'limit': limit,
                    'pct': pct,
                })
        except Exception:
            pass

        return {'cpu': cpu, 'ram_used': ram_used, 'ram_pct': ram_pct, 'buffer': buf, 'gpus': gpus}

    # ---------------------------------------------------------------------
    # system section: top summary + per-gpu history rows
    # ---------------------------------------------------------------------
    async def _system_stats(self) -> Dict[str, Any]:
        cpu = psutil.cpu_percent()
        vm = psutil.virtual_memory()
        ram_used = vm.used / 1e9
        ram_pct = vm.percent
        buf = 0
        if self.step_buffer:
            try:
                buf = await self.step_buffer.size.remote()
            except Exception:
                pass
        # gather GPUs
        gpus = []
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(min(count,8)):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                power = pynvml.nvmlDeviceGetPowerUsage(h)/1000.0
                limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h)/1000.0
                pct = (power/limit*100) if limit>0 else 0.0
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                used = mem.used/1e9
                total = mem.total/1e9
                gpus.append({'id':i,'used':used,'total':total,'power':power,'limit':limit,'pct':pct})
        except Exception:
            pass
        return {'cpu':cpu,'ram_used':ram_used,'ram_pct':ram_pct,'buffer':buf,'gpus':gpus}

    def _system_panel(self, sys_stats: Dict[str, Any]) -> Panel:
        # Top box: cpu/ram/buffer + each gpu mem usage
        txt = Text()
        txt.append(f"CPU %    : {sys_stats['cpu']:.1f}%\n")
        txt.append(f"RAM used : {sys_stats['ram_used']:.1f}/{sys_stats['ram_pct']:.0f}% GB\n")
        txt.append(f"Buffer   : {sys_stats['buffer']}\n")
        # GPU mem lines, colored by power
        for gpu in sys_stats['gpus']:
            style = 'red' if gpu['pct']<50 else 'yellow' if gpu['pct']<75 else 'green'
            txt.append(f"GPU{gpu['id']} Mem: {gpu['used']:.1f}/{gpu['total']:.1f} GB ({gpu['pct']:.0f}% power)\n", style=style)
        top = Panel(txt, title='System Overview', box=box.ROUNDED)

        # Bottom: one history graph per GPU, stacked
        panels = []
        for gpu in sys_stats['gpus']:
            idx = gpu['id']
            if idx not in self.gpu_graphs:
                # self.gpu_graphs[idx]=BufferedHistoryGraph(100.0, width=self.console.size.width-6, height=3, format=lambda v:f"{v:.0f}%", init_bound=100.0,dynamic_bound=False)
                self.gpu_graphs[idx]=BufferedHistoryGraph(100.0, width=self.console.size.width-6, height=3, format=lambda v:f"{v:.0f}%") #,dynamic_bound=False)
            
            graph = self.gpu_graphs[idx]
            graph.add(gpu['pct'])
            input(graph.graph)
            body = '\n'.join(graph.graph)
            panels.append(Panel(body, title=f"GPU {idx} Power History", box=box.SQUARE))
        bottom = Columns(panels, expand=True)

        # full System panel layout: two rows
        layout = Columns([top], expand=True)
        # stack bottom rows below
        return Panel(Columns([top, Columns(panels, expand=True)], expand=True), title='System', box=box.DOUBLE)

    async def run(self):
        layout = Layout()
        layout.split_column(Layout(name='upper'), Layout(name='lower'))
        layout['upper'].split_row(Layout(name='learner'), Layout(name='inference'))
        layout['lower'].split_row(Layout(name='pool'), Layout(name='system'))
        with Live(layout, refresh_per_second=1/self.REFRESH_SEC):
            while True:
                learner,inf,pool,sys = await asyncio.gather(
                    self._learner_stats(),self._inference_stats(),self._pool_stats(),self._system_stats())
                layout['learner'].update(_learner_panel(learner))
                layout['inference'].update(_inference_panel(inf))
                layout['pool'].update(_pool_panel(pool))
                layout['system'].update(self._system_panel(sys))
                await asyncio.sleep(self.REFRESH_SEC)
