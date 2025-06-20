import asyncio, psutil, time, collections, re, pynvml
from typing import Dict, Any, Tuple
from collections import deque
from itertools import zip_longest
from rich.console import Console, Group


"""
right (ts)
left top exploration
left bottom heatmap
right bottom (format rate, invalid move rate, game_len, buffer_size)


bottom gpu
"""
def _bar(pct: float, width: int) -> str: return "█" * int(pct / 100 * width)

class TerminalInterface:
    def __init__(self, tracker, step_buffer):
        self.tracker, self.step_buffer = tracker, step_buffer
        self.console = Console() 
        self._gpu_stats = None
        self._general_stats = None
        self._hist: dict[str, deque[float|int]] = collections.defaultdict(lambda: deque(maxlen=128))
        self._max_tok_s: int = 1_000

        pynvml.nvmlInit()
        self.pynvml=pynvml
        self.gpu_count=pynvml.nvmlDeviceGetCount()

    async def _system_stats(self) -> Dict[str, Any]:
        gpus = []
        if self.pynvml:
            for gid in range(self.gpu_count):
                h = self.pynvml.nvmlDeviceGetHandleByIndex(gid)
                power = self.pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                limit = self.pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                m = self.pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append({"id": gid, "used": m.used/1e9, "total": m.total/1e9, "mem_pct": (m.used/1e9)/(m.total/1e9), "power": power, "limit": limit, "power_pct": power/limit*100 if limit else 0.0})
        return gpus

    async def _fetch_loop(self, interval: float = 1.0):
        while True:
            try:
                self._gpu_stats = await self._system_stats() # system + GPUs
                self._buffer_size = await self.step_buffer.size.remote()
                self._tracker_stats = await self.tracker.get_interface_info() # remaining stats (includes gpus)
            except Exception as e:
                self.console.log(f"[red]stat-fetch error: {e}")

            # TODO track all histories
            await asyncio.sleep(interval)

    def _colour_for_util(self, pct: float) -> str: return "green" if pct >= 80 else "yellow" if pct >= 40 else "red"
    def _gpu_panel(self) -> Panel:
        gpu_panels = []
        bar_w = max(10, int(self.console.size.width * 0.45))
        for gpu_d in self._gpu_stats:
            tok_s = self._tracker_stats.get(gpu_d["id"], 0)
            self._max_tok_s = self._max_tok_s if tok_s<self._max_tok_s else tok_s
            tok_pct = tok_s/self._max_tok_s
            role = "Actor" if tok and tok > 0 else "Learner"
            line1 = Text.assemble(("PWR ", "dim"), Text(_bar(gpu_d['power_pct'],bar_w), style=self._colour_for_util(gpu_d['power_pct'])), f" {gpu_d['power_pct']:5.1f}%")
            line2 = Text.assemble(("MEM ", "dim"), Text(_bar(gpu_d['mem_pct'],bar_w), style=self._colour_for_util(1-gpu_d['mem_pct'])), f" {gpu_d['mem_pct']:5.1f}%")
            line2 = Text.assemble(("TOK ", "dim"), Text(_bar(tok_pct,bar_w), style=self._colour_for_util(tok_pct)), f" {tok:5.0f} tok/s")
        gpu_panel.append(Panel(Group(line1, line2, line3), title=f"GPU{g['id']} - {role}", box=box.SQUARE, padding=(0, 1)))
        
        # build a 2-column grid (rows = ceil(N/2))
        tbl = Table.grid(expand=True, padding=0)
        tbl.add_column(ratio=1)
        tbl.add_column(ratio=1)
        for a, b in zip_longest(compact_blocks[0::2], compact_blocks[1::2], fillvalue=Panel("")): tbl.add_row(a, b) # filler for odd counts
        return Panel(tbl, title="GPU Performance", box=box.DOUBLE)

    def _base_stats(self) -> Panel:
        format_success = self.tracker_stats["Format Success Rate - \\boxed"] # TODO fix
        inv_move_rate = self.tracker_stats["Format Success Rate - Invalid Move"] # TODO fix
        game_len = self.tracker_stats["Game Length"]
        buffer_size = self.buffer_size



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
                    layout["gpu"].update(self._gpu_panel()) # GPU panel (bottom)
                    layout["bs"].update(self._base_stats())
                    layout["ts"].update(self._ts_panel())


                    layout["collection"].update(self._collection_panel())
                    layout["evaluation"].update(self._filter_panel(self._eval, self._EVAL_KEEP, "Evaluation"))
                    layout["pool"].update(self._ts_panel()) # model-pool TrueSkill snapshot
                    layout["heatmap"].update(self._heatmap_panel()) # match-frequency heat-map

                await asyncio.sleep(0.5)
