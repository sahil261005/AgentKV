"""
Export data for the visualizer (docs/index.html) and draw the README charts.

    python -m benchmarks.run_eval            # first, so results/results.json is fresh
    python -m benchmarks.export_visualizer

Writes:
    docs/data.js          everything the web page shows (nothing in the page is made up:
                          every step comes from the real engine)
    docs/assets/*.svg     charts used in README.md

The web page replays a SMALL version of the real setup so every block is visible:
    system prompt = 1 block, each agent's fixed prompt = 4 blocks,
    GPU = 13 blocks (room for 3 agents, not 4).
"""

import json
import math
import os

import numpy as np

from benchmarks.workload_gen import branching_dag, agent_schedule
from core.agent_dag import make_cycle
from core.cache_engine import CacheEngine, make_policy
from core.config import BLOCK_SIZE, recompute_ms, transfer_ms
from core.eviction_policy import choose_victim
from core.tiered_engine import TieredCacheEngine

AGENTS = ["Planner", "Retriever", "Coder", "Reviewer"]
TOKEN_OF = {"System": 0, "Planner": 10, "Retriever": 20, "Coder": 30, "Reviewer": 40}
NAME_OF = {v: k for k, v in TOKEN_OF.items()}
GPU_BLOCKS = 13
CPU_BLOCKS = 20


def prompt(agent):
    return [0] * BLOCK_SIZE + [TOKEN_OF[agent]] * (4 * BLOCK_SIZE)


def label(node):
    return NAME_OF.get(node.tokens[0], "Tail")


def fmt(x):
    """JSON can't hold infinity, so use None for it."""
    if x is None or x == math.inf:
        return None
    return round(float(x), 2)


def describe_score(policy, node):
    if policy.name == "LRU":
        return f"last used at step {node.last_access}"
    if policy.name == "AgentKV":
        d = policy.distance_of(node)
        return "never runs again" if d == math.inf else f"runs again in {int(d)} step{'s' if d != 1 else ''}"
    if policy.name == "AgentKV-Markov":
        d = policy.distance_of(node)
        return "unknown yet (still learning)" if d == math.inf else f"runs again in ~{d:.2f} steps (expected)"
    if policy.name == "Belady":
        s = policy.steps_until_next_use(node)
        return "never needed again" if s == math.inf else f"needed again in {s} steps (future)"
    return ""


def memory_map(tree, pool, on_gpu):
    """Label of the owner of every block ID in a pool (None = free)."""
    cells = [None] * pool.num_blocks
    for node in tree.all_nodes():
        if node.on_gpu != on_gpu:
            continue
        for i, block in enumerate(node.blocks):
            cells[block] = NAME_OF.get(node.tokens[i * BLOCK_SIZE], "Tail")
    return cells


# ----------------------------------------------------------------------
# Engines that record what they decide
# ----------------------------------------------------------------------
class RecordingEngine(CacheEngine):
    """The normal engine, but it writes down every eviction and why."""

    def __init__(self, num_blocks, policy):
        super().__init__(num_blocks, policy)
        self.events = []

    def _make_room(self, need):
        freed = 0
        while self.pool.num_free() < need:
            leaves = self.tree.evictable_leaves()
            if not leaves:
                break
            victim = choose_victim(self.policy, leaves)
            candidates = sorted(leaves, key=self.policy.score, reverse=True)
            self.events.append({
                "type": "evict",
                "victim": label(victim),
                "blocks": len(victim.blocks),
                "candidates": [{"label": label(n), "why": describe_score(self.policy, n),
                                "chosen": n is victim} for n in candidates],
            })
            blocks = self.tree.remove_leaf(victim)
            self.pool.free(blocks)
            self.evictions += 1
            self.evicted_blocks += len(blocks)
            freed += len(blocks)
        return freed


class RecordingTieredEngine(TieredCacheEngine):
    """The tiered engine, but it writes down offloads, copies and prefetches."""

    def __init__(self, gpu_blocks, cpu_blocks, policy):
        super().__init__(gpu_blocks, cpu_blocks, policy)
        self.events = []
        self.phase = "request"

    def _offload(self, node):
        name = label(node)
        if not node.agents:
            self.events.append({"type": "drop", "label": name, "blocks": len(node.blocks)})
        else:
            self.events.append({"type": "offload", "label": name, "blocks": len(node.blocks),
                                "phase": self.phase})
        super()._offload(node)

    def _swap_in(self, node):
        kind = "prefetch" if self.phase == "prefetch" else "ondemand"
        self.events.append({"type": kind, "label": label(node), "blocks": len(node.blocks),
                            "ms": round(transfer_ms(len(node.blocks)), 3)})
        super()._swap_in(node)

    def prefetch(self, finished_agents):
        self.phase = "prefetch"
        super().prefetch(finished_agents)
        self.phase = "request"


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------
def make_sequence(workflow):
    if workflow == "cyclic":
        return AGENTS * 4
    rng = np.random.default_rng(3)
    return [step[0] for step in agent_schedule("branching", branching_dag(), 20, rng)]


def make_dag(workflow):
    return make_cycle(AGENTS) if workflow == "cyclic" else branching_dag()


def eviction_lane(workflow, policy_name):
    sequence = make_sequence(workflow)
    dag = make_dag(workflow)
    prompts = [prompt(a) for a in sequence]
    engine = RecordingEngine(GPU_BLOCKS, make_policy(policy_name, dag, prompts))

    steps = []
    for agent in sequence:
        engine.events = []
        result = engine.run_step([(agent, prompt(agent))])[0]
        distances = None
        if hasattr(engine.policy, "distances") and engine.policy.distances:
            distances = {a: fmt(engine.policy.distances.get(a, math.inf)) for a in AGENTS}
        steps.append({
            "agent": agent,
            "hit": result["hit_tokens"] // BLOCK_SIZE,
            "total": len(prompt(agent)) // BLOCK_SIZE,
            "recomputed": result["recomputed_tokens"] // BLOCK_SIZE,
            "events": engine.events,
            "gpu": memory_map(engine.tree, engine.pool, True),
            "distances": distances,
            "cum_hit": engine.hit_blocks,
            "cum_total": engine.total_blocks,
            "cum_recomputed": engine.recomputed_tokens // BLOCK_SIZE,
        })
    engine.check_invariants()
    return {"policy": policy_name, "steps": steps}


def tiered_lane(workflow, policy_name):
    sequence = make_sequence(workflow)
    dag = make_dag(workflow)
    prompts = [prompt(a) for a in sequence]
    engine = RecordingTieredEngine(GPU_BLOCKS, CPU_BLOCKS, make_policy(policy_name, dag, prompts))

    steps = []
    stall = 0.0
    for agent in sequence:
        engine.events = []
        before_cpu_hits = engine.cpu_hit_blocks
        result = engine.run_step([(agent, prompt(agent))])[0]
        ondemand = engine.cpu_hit_blocks - before_cpu_hits
        stall += transfer_ms(ondemand) + recompute_ms(result["recomputed_tokens"])
        steps.append({
            "agent": agent,
            "reused": result["hit_tokens"] // BLOCK_SIZE,
            "ondemand": ondemand,
            "recomputed": result["recomputed_tokens"] // BLOCK_SIZE,
            "events": engine.events,
            "gpu": memory_map(engine.tree, engine.pool, True),
            "cpu": memory_map(engine.tree, engine.cpu_pool, False),
            "stall_ms": round(stall, 3),
            "cum_prefetched": engine.prefetched_blocks,
            "cum_prefetch_used": engine.prefetch_used_blocks,
            "cum_ondemand": engine.cpu_hit_blocks,
        })
    engine.check_invariants()
    name = {"LRU": "LRU + CPU tier", "AgentKV": "AgentKV + prefetch",
            "AgentKV-Markov": "AgentKV-Markov + prefetch"}[policy_name]
    return {"policy": name, "steps": steps}


# ----------------------------------------------------------------------
# SVG charts for the README (hand-written SVG, no plotting library)
# ----------------------------------------------------------------------
COLORS = {"LRU": "#cf4a2e", "LRU-TailFirst": "#b07d2b", "AgentKV": "#2d6a4f",
          "AgentKV-Markov": "#1f8a8a", "Belady": "#1d1a14"}
BG = "#f5efe3"
GRID = "#ddd3bf"
TEXT = "#3d382e"
MUTED = "#7a7262"


def svg_line_chart(title, subtitle, xs, series, path, y_max=100):
    w, h = 760, 380
    left, right, top, bottom = 64, 170, 70, 52
    pw, ph = w - left - right, h - top - bottom

    def x_of(i):
        return left + pw * i / (len(xs) - 1)

    def y_of(v):
        return top + ph * (1 - v / y_max)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'font-family="Inter,Segoe UI,Helvetica,Arial,sans-serif">',
           f'<rect width="{w}" height="{h}" rx="16" fill="{BG}"/>',
           f'<text x="{left}" y="32" fill="#1d1a14" font-size="18" font-weight="700">{title}</text>',
           f'<text x="{left}" y="52" fill="{MUTED}" font-size="12">{subtitle}</text>']
    for v in range(0, y_max + 1, 20):
        y = y_of(v)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left - 10}" y="{y + 4:.1f}" fill="{MUTED}" font-size="11" text-anchor="end">{v}%</text>')
    for i, x in enumerate(xs):
        out.append(f'<text x="{x_of(i):.1f}" y="{top + ph + 22}" fill="{MUTED}" font-size="11" '
                   f'text-anchor="middle">{x}</text>')
    out.append(f'<text x="{left + pw / 2}" y="{h - 10}" fill="{MUTED}" font-size="11" text-anchor="middle">'
               f'GPU cache size (blocks of 2 MiB)</text>')
    for k, (name, values) in enumerate(series.items()):
        color = COLORS[name]
        dash = ' stroke-dasharray="6 5"' if name == "Belady" else ""
        width = 3.5 if name == "AgentKV" else 2.2
        points = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values))
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"'
                   f' stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        for i, v in enumerate(values):
            out.append(f'<circle cx="{x_of(i):.1f}" cy="{y_of(v):.1f}" r="3.4" fill="{color}"/>')
        ly = top + 10 + k * 24
        out.append(f'<rect x="{left + pw + 22}" y="{ly - 9}" width="14" height="4" rx="2" fill="{color}"/>')
        out.append(f'<text x="{left + pw + 44}" y="{ly - 3}" fill="{TEXT}" font-size="12">'
                   f'{"Bélády (oracle)" if name == "Belady" else name}</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def svg_bar_chart(title, subtitle, rows, path, unit="ms"):
    """rows = [(label, value, color)] drawn as horizontal bars."""
    w = 760
    bar_h, gap = 26, 14
    top, left = 72, 270
    h = top + len(rows) * (bar_h + gap) + 24
    vmax = max(v for _, v, _ in rows) or 1
    pw = w - left - 90
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'font-family="Inter,Segoe UI,Helvetica,Arial,sans-serif">',
           f'<rect width="{w}" height="{h}" rx="16" fill="{BG}"/>',
           f'<text x="28" y="32" fill="#1d1a14" font-size="18" font-weight="700">{title}</text>',
           f'<text x="28" y="52" fill="{MUTED}" font-size="12">{subtitle}</text>']
    for i, (name, value, color) in enumerate(rows):
        y = top + i * (bar_h + gap)
        bw = max(3, pw * value / vmax)
        out.append(f'<text x="{left - 14}" y="{y + bar_h / 2 + 4}" fill="{TEXT}" font-size="12.5" '
                   f'text-anchor="end">{name}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{pw}" height="{bar_h}" rx="6" fill="{GRID}"/>')
        out.append(f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="6" fill="{color}"/>')
        out.append(f'<text x="{left + bw + 8:.1f}" y="{y + bar_h / 2 + 4}" fill="#1d1a14" font-size="12.5" '
                   f'font-weight="600">{value:,.2f} {unit}</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def draw_charts(results, folder):
    os.makedirs(folder, exist_ok=True)
    policies = ["LRU", "LRU-TailFirst", "AgentKV", "AgentKV-Markov", "Belady"]
    for workflow, title in [("cyclic", "4-agent loop: block hit rate vs cache size"),
                            ("branching", "Branching loop (30% back to Coder): block hit rate")]:
        rows = [r for r in results["sweep"] if r["workflow"] == workflow]
        xs = [r["capacity_blocks"] for r in rows]
        series = {p: [r[p + "_hit_rate"] for r in rows] for p in policies}
        svg_line_chart(title, "Llama-3-8B sizing · 200 requests per trace · mean of 10 seeds",
                       xs, series, os.path.join(folder, f"sweep-{workflow}.svg"))

    t = results["headline"]["tiered"]["cyclic_300"]
    rows = [("LRU, GPU only", t["LRU, GPU only"]["stall_ms_per_request"], "#cf4a2e"),
            ("LRU + CPU tier", t["LRU + CPU tier"]["stall_ms_per_request"], "#b07d2b"),
            ("AgentKV + CPU tier", t["AgentKV + CPU tier, no prefetch"]["stall_ms_per_request"], "#c58a17"),
            ("AgentKV + CPU tier + prefetch", t["AgentKV + CPU tier + prefetch"]["stall_ms_per_request"], "#2d6a4f")]
    svg_bar_chart("Modeled stall per request (4-agent loop, 300 GPU blocks)",
                  "Estimate: A100 at 50% utilisation, PCIe 4.0 at 25 GB/s · lower is better",
                  rows, os.path.join(folder, "stall-cyclic.svg"))

    t = results["headline"]["tiered"]["branching_300"]
    rows = [("LRU + CPU tier", t["LRU + CPU tier"]["stall_ms_per_request"], "#b07d2b"),
            ("AgentKV (BFS) + prefetch", t["AgentKV + CPU tier + prefetch"]["stall_ms_per_request"], "#2d6a4f"),
            ("AgentKV-Markov + prefetch", t["AgentKV-Markov + CPU tier + prefetch"]["stall_ms_per_request"], "#1f8a8a")]
    svg_bar_chart("Branching loop: BFS vs Markov prefetching (300 GPU blocks)",
                  "Markov knows Planner comes next 70% of the time, so it prefetches the right agent",
                  rows, os.path.join(folder, "stall-branching.svg"))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    with open("results/results.json") as f:
        results = json.load(f)

    data = {
        "setup": {"gpu_blocks": GPU_BLOCKS, "cpu_blocks": CPU_BLOCKS, "block_size": BLOCK_SIZE,
                  "recompute_ms_per_block": round(recompute_ms(BLOCK_SIZE), 3),
                  "transfer_ms_per_block": round(transfer_ms(1), 3)},
        "eviction": {},
        "tiered": {},
        "results": results,
    }
    for workflow in ["cyclic", "branching"]:
        data["eviction"][workflow] = {
            "sequence": make_sequence(workflow),
            "lanes": [eviction_lane(workflow, p) for p in ["LRU", "AgentKV", "AgentKV-Markov", "Belady"]],
        }
        data["tiered"][workflow] = {
            "sequence": make_sequence(workflow),
            "lanes": [tiered_lane(workflow, p) for p in ["LRU", "AgentKV", "AgentKV-Markov"]],
        }

    os.makedirs("docs", exist_ok=True)
    with open("docs/data.js", "w") as f:
        f.write("// Generated by benchmarks/export_visualizer.py from the real simulator. Do not edit.\n")
        f.write("window.AGENTKV_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")

    draw_charts(results, "docs/assets")
    print("wrote docs/data.js and docs/assets/*.svg")


if __name__ == "__main__":
    main()
