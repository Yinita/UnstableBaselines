"""Rich-powered live dashboard for UnstableBaselines.

Run this in a **separate script** or actor so it does not block
learner or collector threads. It queries remote actors every few hundred
milliseconds and paints four panels in a 2×2 grid:

1. Learner         – recent loss / tokens-sec / grad-norm
2. Inference Queue – per-LoRA queued + running sequences + token rate history
3. Model-Pool      – TrueSkill μ / σ and top-K match-ups
4. System          – CPU/RAM/Buffer + GPU mem + power-history graphs
"""
from __future__ import annotations

import asyncio
import psutil
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict, deque

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
from rich.console import Group


def _learner_panel(stats: Dict[str, Any]) -> Panel:
    body = (
        f"[bold]step[/]: {stats.get('step', '-') }\n"
        f"loss: {stats.get('loss', 0):.4f}\n"
        f"ppl: {stats.get('ppl', 0):.1f}\n"
        f"tokens/sec: {stats.get('tok_s', 0):.0f}\n"
        f"grad-norm : {stats.get('grad_norm', 0):.1f}"
    )
    return Panel(body, title="Learner", box=box.ROUNDED)


def _inference_panel(stats: Dict[str, Dict[str, float]], tok_graphs: Dict[str, Dict[str, BufferedHistoryGraph]], console_width: int) -> Panel:
    print(stats)
    print(tok_graphs)
    # Main table showing current stats
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    tbl.add_column("Actor",          style="bold")
    tbl.add_column("LoRA",           style="bold")
    tbl.add_column("Queued",  justify="right")
    tbl.add_column("Running", justify="right")
    tbl.add_column("Tok/s",  justify="right")

    # Group by actor first, then by LoRA
    for actor_name, lora_stats in sorted(stats.items()):
        for lora_name, meta in sorted(lora_stats.items()):
            tok_rate = meta.get('tok_s', 0)
            
            # Initialize graph for this actor+LoRA combination
            if actor_name not in tok_graphs:
                tok_graphs[actor_name] = {}
            if lora_name not in tok_graphs[actor_name]:
                tok_graphs[actor_name][lora_name] = BufferedHistoryGraph(10_000.0, width=console_width//4-8, height=4, format=lambda v: f"{v:.0f}", dynamic_bound=True)
            tok_graphs[actor_name][lora_name].add(tok_rate)
            
            # Style based on activity
            queue_style = "red" if meta.get("queue", 0) > 10 else "yellow" if meta.get("queue", 0) > 5 else "green"
            running_style = "green" if meta.get("running", 0) > 0 else "dim"
            tok_style = "green" if tok_rate > 10 else "yellow" if tok_rate > 1 else "dim"
            
            tbl.add_row(actor_name, lora_name, f"[{queue_style}]{meta.get('queue', 0)}[/]", f"[{running_style}]{meta.get('running', 0)}[/]", f"[{tok_style}]{tok_rate:.1f}[/]")
    
    # Create history graphs for active actor+LoRA combinations
    graph_panels = []
    for actor_name, lora_graphs in sorted(tok_graphs.items()):
        if actor_name in stats:  # Only show graphs for currently active actors
            for lora_name, graph in sorted(lora_graphs.items()):
                if lora_name in stats[actor_name]:  # Only show graphs for currently active LoRAs
                    current_rate = stats[actor_name][lora_name].get('tok_s', 0)
                    
                    # Style based on current rate
                    if current_rate < 1:        style = 'dim'
                    elif current_rate < 10:     style = 'yellow'
                    elif current_rate < 50:     style = 'green'
                    else:                       style = 'bold green'
                    
                    # Truncate graph lines to fit panel width
                    graph_str = '\n'.join([line[-(console_width // 4 - 10):] for line in graph.graph])
                    
                    title = Text(f"{actor_name}:{lora_name} ({current_rate:.1f} tok/s)", style=style)
                    graph_panels.append(Panel(graph_str, title=title, box=box.SQUARE, width=console_width // 4 - 3, style=style))
    
    # Combine table and graphs
    if graph_panels:    content = Group(tbl, Text(""), Columns(graph_panels, expand=True))
    else:               content = tbl
    return Panel(content, title="Inference (queues, rates & history)", box=box.ROUNDED)


def _pool_panel(pool: Dict[str, Any]) -> Panel:
    tbl = Table(box=None)
    tbl.add_column("metric")
    tbl.add_column("value", justify="right")
    for k, v in pool.items(): tbl.add_row(k, f"{v}")
    return Panel(tbl, title="Model-Pool", box=box.ROUNDED)

class TerminalInterface:
    REFRESH_SEC: float = 0.5

    def __init__(self, tracker, model_pool, actors: List, step_buffer=None) -> None:
        self.tracker = tracker
        self.model_pool = model_pool
        self.actors = actors
        self.step_buffer = step_buffer
        self.console = Console()

        # Try initializing NVML and GPU graphs once
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            self.pynvml = None
            self.gpu_count = 0

        self.gpu_graphs: Dict[int, BufferedHistoryGraph] = {}
        for i in range(self.gpu_count):
            self.gpu_graphs[i] = BufferedHistoryGraph(100.0, width=self.console.size.width//4-5, height=3, format=lambda v: f"{v:.0f}%", dynamic_bound=False)
        
        # Token rate graphs for each actor+LoRA combination
        self.tok_graphs: Dict[str, Dict[str, BufferedHistoryGraph]] = {}

    async def _learner_stats(self) -> Dict[str, Any]:
        try:                return await self.tracker.get_latest_learner_metrics.remote()
        except Exception:   return {}

    async def _inference_stats(self):
        try:                return await self.tracker.get_latest_inference_metrics.remote() #self.tracker.get_latest_inference_metrics.remote()
        except Exception:   return {}

    async def _pool_stats(self) -> Dict[str, Any]:
        try:
            snap = await self.model_pool.get_snapshot.remote()
            return {"ckpts": snap.get("num_ckpts"), "μ±σ": f"{snap['mu']:.1f}±{snap['sigma']:.1f}"}
        except Exception:
            return {}

    async def _system_stats(self) -> Dict[str, Any]:
        cpu = psutil.cpu_percent()
        vm = psutil.virtual_memory()
        ram_used = vm.used / 1e9
        ram_pct = vm.percent
        buf = 0
        if self.step_buffer:
            try:                buf = await self.step_buffer.size.remote()
            except Exception:   pass

        gpus = []
        if self.pynvml:
            try:
                for i in range(min(self.gpu_count, 8)):
                    h = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    power = self.pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    limit = self.pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                    pct = (power / limit * 100) if limit > 0 else 0.0
                    mem = self.pynvml.nvmlDeviceGetMemoryInfo(h)
                    used = mem.used / 1e9
                    total = mem.total / 1e9
                    gpus.append({'id': i, 'used': used, 'total': total, 'power': power, 'limit': limit, 'pct': pct})
            except Exception:
                pass

        return {'cpu': cpu, 'ram_used': ram_used, 'ram_pct': ram_pct, 'buffer': buf, 'gpus': gpus}

    def _system_panel(self, sys_stats: Dict[str, Any]) -> Panel:
        txt = Text()
        txt.append(f"CPU %    : {sys_stats['cpu']:.1f}%\n")
        txt.append(f"RAM used : {sys_stats['ram_used']:.1f}/{sys_stats['ram_pct']:.0f}% GB\n")
        txt.append(f"Buffer   : {sys_stats['buffer']}\n")

        panels = []
        for gpu in sys_stats['gpus']:
            style = 'green' if gpu['pct'] < 50 else 'yellow' if gpu['pct'] < 75 else 'red'
            txt.append(f"GPU{gpu['id']} Mem: {gpu['used']:.1f}/{gpu['total']:.1f} GB ({gpu['pct']:.0f}% power)\n", style=style)
            
            if gpu['pct'] < 30:     style = 'dim'
            elif gpu['pct'] < 60:   style = 'yellow'
            elif gpu['pct'] < 85:   style = 'orange1'
            else:                   style = 'red bold'
                
            # Update pre-initialized graph
            self.gpu_graphs[gpu['id']].add(gpu['pct'])
            graph_str = '\n'.join([line[-(self.console.size.width // 4 - 9):] for line in self.gpu_graphs[gpu['id']].graph])
            title = Text(f"GPU {gpu['id']} Power History ({gpu['pct']:.2f}%)", style=style)
            panels.append(Panel(graph_str, title=title, box=box.SQUARE, width=self.console.size.width//4-5, style=style))

        top = Panel(txt, title='System Overview', box=box.ROUNDED)
        return Panel(Columns([top, Columns(panels, expand=True)], expand=True),  title='System', box=box.DOUBLE)

    async def run(self):
        layout = Layout()
        layout.split_column(Layout(name='upper'), Layout(name='lower'))
        layout['upper'].split_row(Layout(name='learner'), Layout(name='inference'))
        layout['lower'].split_row(Layout(name='pool'), Layout(name='system'))
        
        with Live(layout, refresh_per_second=1/self.REFRESH_SEC):
            while True:
                try:
                    learner, inf, pool, sys = await asyncio.gather(self._learner_stats(), self._inference_stats(), self._pool_stats(), self._system_stats())
                    layout['learner'].update(_learner_panel(learner))
                    layout['inference'].update(_inference_panel(inf, self.tok_graphs, self.console.size.width))
                    layout['pool'].update(_pool_panel(pool))
                    layout['system'].update(self._system_panel(sys))
                except Exception as e:
                    # Handle any async errors gracefully
                    layout['learner'].update(Panel(f"Error: {e}", title="Error"))
                await asyncio.sleep(self.REFRESH_SEC)
