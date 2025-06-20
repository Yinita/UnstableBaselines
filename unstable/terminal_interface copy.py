from __future__ import annotations
import asyncio, psutil, time, collections, re
from typing import Dict, Any, List
from collections import deque

import ray
from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.table import Table
from rich.align import Align
from rich.style import Style
from rich.color import Color


from nvitop.tui.library import BufferedHistoryGraph

_CKPT_RE = re.compile(r"ckpt-(-?\d+)$")
_SPARK_BARS = " ▁▂▃▄▅▆▇█"          # 8-level spark chars; first one is “blank”

def _spark(vals: deque[float], width: int = 16) -> str:
    """Return a fixed-width sparkline from the last `width` values."""
    if not vals:
        return " " * width
    data = list(vals)[-width:]
    lo, hi = min(data), max(data)
    if hi == lo:
        return _SPARK_BARS[-1] * len(data)  # flat line
    rng = hi - lo
    idx = lambda v: int((v - lo) / rng * (len(_SPARK_BARS) - 1))
    return "".join(_SPARK_BARS[idx(v)] for v in data).rjust(width)

def _ckpt_idx(uid: str) -> int | None:
    """Return numeric index for 'ckpt-42' or 'ckpt--1'.  None otherwise."""
    m = _CKPT_RE.match(uid)
    return int(m.group(1)) if m else None

def _uid_sort_key(uid: str):
    """(0, idx) for checkpoints, (1, uid) for everything else."""
    idx = _ckpt_idx(uid)
    return (0, idx) if idx is not None else (1, uid)

def _trim_uid(uid: str, max_len: int = 30) -> str:
    """Keep the right-most `max_len` chars so ckpt numbers stay visible."""
    return uid[-max_len:] if len(uid) > max_len else uid


class TerminalInterface:
    _COLL_KEEP = ("invalid_move", "Win Rate", "Loss Rate", "Draw Rate", "Game Length", "Reward")
    _EVAL_KEEP = tuple(k for k in _COLL_KEEP if k != "invalid_move")
    def __init__(self, tracker, model_pool, step_buffer=None) -> None:
        self.tracker, self.model_pool, self.step_buffer = (tracker, model_pool, step_buffer)
        self.console = Console()
        self._latest_stats: Dict[str, Any] = {}
        self._hist: dict[str, deque[float]] = collections.defaultdict(lambda: deque(maxlen=128))
        self._coll_metrics = {
            "Win Rate": {"key": "collection-all/Win Rate", "fmt": lambda v: f"{v*100:5.1f}%"},
            "Loss Rate": {"key": "collection-all/Loss Rate", "fmt": lambda v: f"{v*100:5.1f}%"},
            "Draw Rate": {"key": "collection-all/Draw Rate", "fmt": lambda v: f"{v*100:5.1f}%"},
            "Game Length": {"key": "collection-all/Game Length", "fmt": lambda v: f"{v:5.2f}"},
            "Avg. Reward": {"key": "collection-all/Reward", "fmt": lambda v: f"{v:6.3f}"},
            "Has-Think %": {"key": "collection-all/Format Success Rate - has_think", "fmt": lambda v: f"{v*100:5.1f}%"},
        }
        _COLL_BOUNDS = {"Win Rate": 100.0, "Loss Rate": 100.0, "Draw Rate": 100.0, "Has-Think %": 100.0, "Game Length": 25.0, "Avg. Reward": 1.0}
        graph_w, graph_h = int(self.console.size.width * 0.28), 4
        self._coll_graphs = {name: BufferedHistoryGraph(_COLL_BOUNDS[name], width=graph_w, height=graph_h, format=lambda v: "", dynamic_bound=False, upsidedown=False) for name in self._coll_metrics}

        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml    = pynvml
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            self.pynvml, self.gpu_count = None, 0

        # ── history graphs ──
        self.power_graphs: Dict[int, BufferedHistoryGraph] = {}
        self.tok_graphs: Dict[int, BufferedHistoryGraph] = {}
        self.mem_graphs: Dict[int, BufferedHistoryGraph] = {}

        full_w = self.console.size.width
        graph_w = int(full_w) # * 0.75) # right-hand graph width

        for gid in range(self.gpu_count):
            self.power_graphs[gid] = BufferedHistoryGraph(100.0, width=graph_w, height=3, format=lambda v: f"{v:.0f}%", dynamic_bound=False)
            self.tok_graphs[gid] = BufferedHistoryGraph(1e4,  width=graph_w, height=3, format=lambda v: f"{v:.0f}", dynamic_bound=False, upsidedown=True)
            self.mem_graphs[gid] = BufferedHistoryGraph(100.0, width=graph_w, height=3, format=lambda v: f"{v:.0f}%", dynamic_bound=False)

        self._coll = None
    async def _system_stats(self) -> Dict[str, Any]:
        # self.console.log(f"NVML ok? {bool(self.pynvml)}, GPUs seen: {self.gpu_count}")
        cpu = psutil.cpu_percent()
        vm = psutil.virtual_memory()

        buf = 0
        if self.step_buffer:
            try:                buf = await self.step_buffer.size.remote()
            except Exception:   pass

        gpus = []
        if self.pynvml:
            for gid in range(self.gpu_count):
                h = self.pynvml.nvmlDeviceGetHandleByIndex(gid)
                power = self.pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                limit = self.pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                pct = power / limit * 100 if limit else 0.0
                m = self.pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append({"id": gid, "used": m.used/1e9, "total": m.total/1e9, "power": power, "limit": limit, "pct": pct})

        try:    gpu_tok = await self.tracker.get_gpu_tok_rates.remote()
        except Exception:
            gpu_tok = {}

        return {"cpu": cpu, "ram_used": vm.used/1e9, "ram_pct": vm.percent, "buffer": buf, "gpus": gpus, "gpu_tok": gpu_tok}

    def _colour_for_util(self, pct: float) -> str: return "green" if pct >= 80 else "yellow" if pct >= 40 else "red"

    def _gpu_block(self, g: Dict[str, float], tok_rate: float | None) -> Panel:
        gid = g["id"]
        mem_pct = g["used"] / g["total"] * 100
        util_pct = g["pct"]
        role = "Actor" if tok_rate else "Learner"
        colour = self._colour_for_util(util_pct)

        # update graphs
        self.power_graphs[gid].add(util_pct)
        if tok_rate: self.tok_graphs[gid].add(tok_rate)
        else:        self.mem_graphs[gid].add(mem_pct)

        # build left-hand info (≈ 15 %)
        info_lines = [f"GPU{gid}", role, f"Memory  {mem_pct:>3.0f}%", f"Power    {util_pct:>3.0f}%",]
        left = Panel(Text("\n".join(info_lines)), box=box.MINIMAL, padding=(0,1))

        # right-hand graphs
        width_cut = int(self.console.size.width * 0.70)
        lower_str  = "\n".join(line[-width_cut:] for line in self.tok_graphs[gid].graph) if tok_rate else "\n".join(line[-width_cut:] for line in self.mem_graphs[gid].graph)
        graphs = Group(
            Text(f"{util_pct:3.0f}% Power util.", style=colour), 
            Text("\n".join(line[-width_cut:] for line in self.power_graphs[gid].graph), style=colour), 
            Text("-" * width_cut, style="dim"), 
            Text(lower_str, style=colour),
            Text(f"{tok_rate:5.0f} tok/s" if tok_rate else f"{mem_pct:3.0f}% Memory", style=colour), 
        )
        
        body = Columns([left, graphs], expand=True, equal=False, align="left")
        return Panel(body, title=f"GPU{gid}", box=box.SQUARE, style=colour)

    def _gpu_panel(self, sys: Dict[str, Any]) -> Panel:
        gpus = sys["gpus"]
        gpu_tok = sys["gpu_tok"]
        if not gpus: return Panel(Text("no GPUs"), title="GPU Performance", box=box.DOUBLE)
        term_w, term_h = self.console.size
        full_height = 7 * len(gpus) + 2
        use_full = (term_w >= 240 and full_height <= term_h)
        if use_full:
            blocks = [self._gpu_block(g, gpu_tok.get(g["id"], 0.0)) for g in gpus]
            return Panel(Group(*blocks), title="GPU Performance", box=box.DOUBLE)

        compact_blocks = [self._compact_gpu_panel(g, gpu_tok.get(g["id"], None)) for g in gpus]
        # build a 2-column grid (rows = ceil(N/2))
        tbl = Table.grid(expand=True, padding=0)
        tbl.add_column(ratio=1)
        tbl.add_column(ratio=1)
        from itertools import zip_longest
        blank = Panel("")        # filler for odd counts
        for a, b in zip_longest(compact_blocks[0::2], compact_blocks[1::2], fillvalue=blank):
            tbl.add_row(a, b)
        return Panel(tbl, title="GPU Performance", box=box.DOUBLE)


    def _filter_panel(self, metrics: dict[str, float], keep: tuple[str], title: str) -> Panel:
        by_env: dict[str, list[str]] = collections.defaultdict(list)
        for full_key, val in metrics.items():
            # full_key = "collection-SimpleTak-v0-train/Win Rate"
            _, env_id, metric_name = full_key.split("-", 2) if full_key.count("-") >= 2 else ("", "all", full_key)
            match_any = any(k in metric_name for k in keep)
            if not match_any:
                continue
            pretty = f"{metric_name}: {val:.3f}"
            by_env[env_id].append(pretty)

        if not by_env:
            return Panel(Text("waiting …"), title=title, box=box.DOUBLE)

        # format lines env-by-env
        blocks = []
        for env, lines in sorted(by_env.items()):
            hdr = Text(f"[bold]{env}[/]")
            body = Text("\n".join(lines))
            blocks.append(hdr + Text("\n") + body)
        return Panel(Text("\n\n").join(blocks), title=title, box=box.DOUBLE)

    def _collection_panel(self) -> Panel:
        if not self._coll:
            return Panel(Text("waiting …"), title=title, box=box.DOUBLE)
        # build / refresh each mini-panel
        mini: list[Panel] = []
        for pretty, meta in self._coll_metrics.items():
            key = meta["key"]
            if key not in self._coll:
                mini.append(Panel(Text(" "), title=f"{pretty}\n–", box=box.SQUARE, padding=(0, 1)))
                continue
            title = f"{pretty}  {meta['fmt'](float(self._coll[key]))}"
            inner_w = self._coll_graphs[pretty].width        # graph_w you set in __init__
            buf_lines = self._coll_graphs[pretty].graph
            graph_str = "\n".join(line for line in buf_lines)
            mini.append(Panel(Align(Text(graph_str, no_wrap=True, overflow="crop"), align="right", vertical="bottom"), title=title, box=box.SQUARE, padding=(0, 1)))
        # insure exactly six panels (pad with blanks if metrics missing)
        while len(mini) < 6:
            mini.append(Panel(Text(" "), box=box.SQUARE, padding=(0, 1)))

        # explicit 3×2 grid via Layout
        grid = Layout()
        grid.split_row(Layout(name="col1"), Layout(name="col2")) # split into the two vertical columns
        # col1 gets rows 0,2,4   – col2 gets rows 1,3,5
        col1_panels = [mini[i] for i in range(0, 6, 2)]
        col2_panels = [mini[i] for i in range(1, 6, 2)]
        grid["col1"].split_column(*[Layout(p, ratio=1) for p in col1_panels])
        grid["col2"].split_column(*[Layout(p, ratio=1) for p in col2_panels])
        return Panel(grid, title="Collection", box=box.DOUBLE)

    async def _fetch_loop(self, interval: float = 1.0):
        while True:
            try:
                self._latest_stats = await self._system_stats() # system + GPUs
                # tracker-side snapshots
                (self._coll, self._eval, self._ts, self._counts) = await asyncio.gather(
                    self.tracker.get_collection_metrics.remote(), self.tracker.get_eval_metrics.remote(),
                    self.tracker.get_ts_snapshot.remote(), self.tracker.get_match_counts.remote(),
                )
            except Exception as e:
                self.console.log(f"[red]stat-fetch error: {e}")
            if hasattr(self, "_coll"):
                for k, v in self._coll.items():
                    if "-all/" in k:                  # keep only aggregated keys
                        name = k.split("/", 1)[1]     # → "Win Rate", "Reward", …
                        self._hist[name].append(float(v))
                        for pretty, meta in self._coll_metrics.items():
                            key = meta["key"]
                            if key not in self._coll:
                                continue
                            raw = float(self._coll[key])
                            # convert fractions to %
                            if pretty.endswith("%") or "Rate" in pretty:    val = raw * 100.0
                            else:                                           val = raw
                            self._coll_graphs[pretty].add(val)
            await asyncio.sleep(interval)

    def _ts_panel(self) -> Panel:
        if not self._ts: return Panel(Text("waiting …"), title="Model-Pool", box=box.DOUBLE) # nothing yet
        ckpts = [u for u in self._ts if _ckpt_idx(u) is not None]
        if not ckpts: return Panel(Text("no ckpt"), title="Model-Pool", box=box.DOUBLE)
        cur = max(ckpts, key=_ckpt_idx) # safe now
        if not cur: return Panel(Text("no ckpt"), title="Model-Pool")
        # mu/sigma ± two neighbours
        neigh = sorted(self._ts, key=_uid_sort_key)
        idx = neigh.index(cur)
        slice_ = neigh[max(0, idx - 2) : idx + 3]
        BAR_FIELD = 30 # max width of the bar block
        bars = []
        for u in slice_:
            mu, sig = self._ts[u]["mu"], self._ts[u]["sigma"]
            # build a left-aligned bar whose *start* column is identical everywhere
            bar = "█" * min(int(mu//4), BAR_FIELD)
            bar_blk = f"{bar:<{BAR_FIELD}}"      # pad so every row = BAR_FIELD chars
            # final line: [UID][two spaces][BAR][μ/σ]
            line = (f"{_trim_uid(u):>30}  {bar_blk} μ {mu:5.2f} (σ={sig:.2f})")
            bars.append(line)
        return Panel(Text("\n".join(bars)), title="TrueSkill (μ, σ)", box=box.SQUARE)

    def _heatmap_panel(self) -> Panel:
        if not self._counts:
            return Panel(Text("waiting …"), title="Match Frequencies", box=box.SQUARE)
        all_uid = {u for pair in self._counts for u in pair}
        uids = sorted(all_uid, key=_uid_sort_key)[:10]
        if not uids:
            return Panel(Text("no valid ckpts"), title="Match Frequencies", box=box.SQUARE)
        size = len(uids)
        max_cnt = max(self._counts.values())
        # convenience look-up that works no matter which order the tuple was stored
        def _cnt(a, b): return self._counts.get(tuple(sorted((a, b))), 0)
        tbl = Table.grid(padding=(0, 1))
        tbl.add_row("") # corner empty cell
        tbl.add_row(*([""] + [Text(uid[-3:], style="bold") for uid in uids]))

        for ua in uids:
            row = [Text(ua[-3:], style="bold")]        # row header
            for ub in uids:
                c = _cnt(ua, ub)
                if max_cnt == 0:  pct = 0.0
                else:             pct = c / max_cnt    # 0-1
                gray = int(255 * (1 - pct))            # darker ⇐ more games
                style = Style(bgcolor=Color.from_rgb(gray, gray, gray))
                row.append(Text(f"{c:3}", style=style))
            tbl.add_row(*row)
        return Panel(tbl, title="Match Frequencies", box=box.SQUARE)

    def _compact_gpu_panel(self, g: Dict[str, float], tok: float | None) -> Panel:
        mem_pct = g["used"] / g["total"] * 100
        power_pct = g["pct"]
        role = "Actor" if tok and tok > 0 else "Learner"
        bar_w = max(10, int(self.console.size.width * 0.48))  # never smaller
        def _bar(pct: float, width: int = bar_w) -> str: return "█" * int(pct / 100 * width)
        bar_pwr = Text(_bar(power_pct), style=self._colour_for_util(power_pct))
        bar_mem = Text(_bar(mem_pct),   style="green" if mem_pct < 80 else "red")
        if role == "Actor": # map tok/s to 0–100 % for a bar; 10 k tok/s saturates the bar
            tok_pct = min(tok / 5_000 * 100, 100) if tok else 0
            bar_tok = Text(_bar(tok_pct), style="yellow")
        if role == "Actor":
            line1 = Text.assemble(("PWR ", "dim"), bar_pwr, f" {power_pct:5.1f}%")
            line2 = Text.assemble(("TOK ", "dim"), bar_tok, f" {tok:5.0f} tok/s")
        else:  # Learner
            line1 = Text.assemble(("PWR ", "dim"), bar_pwr, f" {power_pct:5.1f}%")
            line2 = Text.assemble(("MEM ", "dim"), bar_mem, f" {mem_pct:5.1f}%")
        body = Group(line1, line2)
        return Panel(body, title=f"GPU{g['id']} - {role}", box=box.SQUARE, padding=(0, 1))


    async def run(self):
        layout = Layout()
        layout.split_column(Layout(name="grid", ratio=2), Layout(name="gpu", ratio=1))
        layout["grid"].split_column(Layout(name="row1"), Layout(name="row2"))
        layout["row1"].split_row(Layout(name="collection"), Layout(name="pool"))
        layout["row2"].split_row(Layout(name="evaluation"), Layout(name="heatmap"))

        # start background fetcher
        asyncio.create_task(self._fetch_loop())

        with Live(layout): #, refresh_per_second=2.0):
            while True:
                if self._latest_stats:
                    layout["gpu"].update(self._gpu_panel(self._latest_stats)) # GPU panel (bottom)
                    layout["collection"].update(self._collection_panel())
                    layout["evaluation"].update(self._filter_panel(self._eval, self._EVAL_KEEP, "Evaluation"))
                    layout["pool"].update(self._ts_panel()) # model-pool TrueSkill snapshot
                    layout["heatmap"].update(self._heatmap_panel()) # match-frequency heat-map

                await asyncio.sleep(0.5)
