"""
Benchmark runner.

    python -m benchmarks.run_eval            # full run (10 seeds)
    python -m benchmarks.run_eval --quick    # 3 seeds, for a fast check

Experiments
  1+2. Capacity sweep 120 -> 480 blocks, every policy (cyclic + branching workflows)
  3.   Branching workflow, 10 seeds, mean +- std, per-seed win count
  4.   All 5 workflows at 75% of the working set (where it wins, ties, loses)
  5.   Sensitivity: number of agents, agent prefix length, tail length
  6.   One scenario converted for four models (GQA vs no GQA)
  7.   GPU + CPU memory tiers, with and without workflow-aware prefetching

Writes results/RESULTS.md, results/results.json and results/sweep.csv.
"""

import argparse
import csv
import json
import os
import time

import numpy as np

from benchmarks.workload_gen import WorkloadConfig, make_trace, working_set_blocks, WORKFLOWS
from core.cache_engine import run_workload, POLICY_NAMES
from core.tiered_engine import run_workload_tiered, single_tier_stall_ms
from core.config import (
    blocks_to_mib, tokens_to_tflops, MODEL_PRESETS, model_bytes_per_block,
    model_bytes_per_token, MIB, GIB,
)

CAPACITIES = [120, 180, 240, 300, 360, 420, 480]
N_STEPS = 200


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------
def run_policies(workflow, capacity, seeds, config=None, n_steps=N_STEPS):
    """Run every policy on the same traces. Returns {policy: [summary per seed]}."""
    if config is None:
        config = WorkloadConfig()
    static = config.system_tokens + config.prefix_tokens

    results = {p: [] for p in POLICY_NAMES}
    for seed in seeds:
        dag, steps = make_trace(workflow, n_steps, seed, config)
        for policy in POLICY_NAMES:
            results[policy].append(run_workload(steps, dag, policy, capacity, static))
    return results


def mean(results, policy, key):
    return float(np.mean([r[key] for r in results[policy]]))


def std(results, policy, key):
    return float(np.std([r[key] for r in results[policy]]))


def percent_cut(old, new):
    if old == 0:
        return 0.0
    return 100.0 * (old - new) / old


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------
def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def ascii_chart(title, capacities, series):
    """Small text bar chart: one bar per policy per capacity."""
    lines = [title]
    for i, cap in enumerate(capacities):
        for name, values in series.items():
            bar = "#" * int(round(values[i] / 2.5))
            lines.append(f"{cap:>4} blk  {name:<8} {bar} {values[i]:.1f}%")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def experiment_sweep(workflow, seeds):
    rows = []
    raw = []
    lru_total = 0
    agentkv_total = 0
    for cap in CAPACITIES:
        res = run_policies(workflow, cap, seeds)
        lru_recomp = mean(res, "LRU", "recomputed_tokens")
        akv_recomp = mean(res, "AgentKV", "recomputed_tokens")
        lru_total += lru_recomp
        agentkv_total += akv_recomp

        rows.append([
            cap, f"{blocks_to_mib(cap):.0f}",
            f"{mean(res, 'LRU', 'hit_rate'):.1f}",
            f"{mean(res, 'LRU-TailFirst', 'hit_rate'):.1f}",
            f"**{mean(res, 'AgentKV', 'hit_rate'):.1f}**",
            f"{mean(res, 'AgentKV-Markov', 'hit_rate'):.1f}",
            f"{mean(res, 'Belady', 'hit_rate'):.1f}",
            f"{lru_recomp:,.0f} -> {akv_recomp:,.0f}",
            f"{percent_cut(lru_recomp, akv_recomp):.0f}%",
        ])
        record = {"workflow": workflow, "capacity_blocks": cap, "capacity_mib": blocks_to_mib(cap)}
        for p in POLICY_NAMES:
            record[p + "_hit_rate"] = mean(res, p, "hit_rate")
            record[p + "_hit_rate_std"] = std(res, p, "hit_rate")
            record[p + "_recomputed_tokens"] = mean(res, p, "recomputed_tokens")
        raw.append(record)

    headers = ["Blocks", "MiB", "LRU %", "LRU-TailFirst %", "AgentKV %", "AgentKV-Markov %", "Bélády %",
               "Recomputed tokens LRU -> AgentKV", "Cut"]
    return markdown_table(headers, rows), raw, lru_total, agentkv_total


def experiment_seeds(capacity, seeds):
    res = run_policies("branching", capacity, seeds)
    rows = []
    for i, seed in enumerate(seeds):
        rows.append([seed] + [f"{res[p][i]['hit_rate']:.1f}" for p in POLICY_NAMES])
    rows.append(["**mean ± std**"] + [
        f"**{mean(res, p, 'hit_rate'):.1f} ± {std(res, p, 'hit_rate'):.1f}**" for p in POLICY_NAMES
    ])
    wins = sum(1 for i in range(len(seeds)) if res["AgentKV"][i]["hit_rate"] > res["LRU"][i]["hit_rate"])
    gap = mean(res, "Belady", "hit_rate") - mean(res, "AgentKV", "hit_rate")
    headers = ["Seed"] + [p + " %" for p in POLICY_NAMES]
    summary = {
        "capacity_blocks": capacity,
        "agentkv_wins": wins,
        "seeds": len(seeds),
        "gap_to_belady": gap,
        "lru_mean": mean(res, "LRU", "hit_rate"),
        "agentkv_mean": mean(res, "AgentKV", "hit_rate"),
        "agentkv_std": std(res, "AgentKV", "hit_rate"),
        "belady_mean": mean(res, "Belady", "hit_rate"),
        "per_seed": {p: [r["hit_rate"] for r in res[p]] for p in POLICY_NAMES},
    }
    return markdown_table(headers, rows), summary


def verdict(lru, agentkv):
    if agentkv > lru + 0.5:
        return "AgentKV wins"
    if agentkv < lru - 0.5:
        return "AgentKV loses"
    return "tie"


def experiment_workflows(seeds):
    config = WorkloadConfig()
    rows = []
    raw = []
    for workflow in WORKFLOWS:
        dag, _ = make_trace(workflow, 10, 0, config)
        cap = int(0.75 * working_set_blocks(dag, config))
        res = run_policies(workflow, cap, seeds)
        lru = mean(res, "LRU", "hit_rate")
        akv = mean(res, "AgentKV", "hit_rate")
        rows.append([workflow, cap] + [f"{mean(res, p, 'hit_rate'):.1f}" for p in POLICY_NAMES] + [verdict(lru, akv)])
        raw.append({"workflow": workflow, "capacity_blocks": cap,
                    **{p + "_hit_rate": mean(res, p, "hit_rate") for p in POLICY_NAMES}})
    headers = ["Workflow", "Blocks (75% of working set)"] + [p + " %" for p in POLICY_NAMES] + ["Result"]
    return markdown_table(headers, rows), raw


def sensitivity_row(label, workflow, config, seeds):
    dag, _ = make_trace(workflow, 10, 0, config)
    cap = int(0.75 * working_set_blocks(dag, config))
    res = run_policies(workflow, cap, seeds, config)
    lru = mean(res, "LRU", "hit_rate")
    akv = mean(res, "AgentKV", "hit_rate")
    row = [label, workflow, cap] + [f"{mean(res, p, 'hit_rate'):.1f}" for p in POLICY_NAMES] + [f"{akv - lru:+.1f}"]
    raw = {"setting": label, "workflow": workflow, "capacity_blocks": cap,
           **{p + "_hit_rate": mean(res, p, "hit_rate") for p in POLICY_NAMES}}
    return row, raw


def experiment_sensitivity(seeds):
    settings = []
    for n in [4, 6, 8]:
        settings.append((f"{n} agents", "cyclic", WorkloadConfig(num_agents=n)))
    for prefix in [512, 1536, 3072]:
        settings.append((f"agent prefix {prefix} tok", "branching", WorkloadConfig(prefix_tokens=prefix)))
    for lo, hi in [(64, 256), (256, 1024), (1024, 2048)]:
        settings.append((f"tail {lo}-{hi} tok", "branching", WorkloadConfig(tail_min=lo, tail_max=hi)))

    rows = []
    raw = []
    for label, workflow, config in settings:
        row, record = sensitivity_row(label, workflow, config, seeds)
        rows.append(row)
        raw.append(record)
    headers = ["Setting", "Workflow", "Blocks"] + [p + " %" for p in POLICY_NAMES] + ["AgentKV - LRU (pts)"]
    return markdown_table(headers, rows), raw


def experiment_models(seeds, capacity=360, budget_mib=600):
    """
    The simulator works in blocks, so hit rates are the same for every model.
    What changes is how many bytes a block is and how many FLOPs a token costs.
    This table shows the same 360-block scenario converted for each model.
    """
    res = run_policies("branching", capacity, seeds)
    saved_tokens = mean(res, "LRU", "recomputed_tokens") - mean(res, "AgentKV", "recomputed_tokens")

    rows = []
    for model in MODEL_PRESETS:
        per_token = model_bytes_per_token(model)
        per_block = model_bytes_per_block(model)
        tflops = saved_tokens * 2 * MODEL_PRESETS[model]["params"] / 1e12
        rows.append([
            model,
            f"{per_token // 1024} KiB",
            f"{per_block / MIB:.2f} MiB",
            int(budget_mib * MIB // per_block),
            f"{capacity * per_block / GIB:.2f} GiB",
            f"{saved_tokens * per_token / GIB:.1f}",
            f"{tflops:,.0f}",
        ])
    headers = ["Model", "KV per token", "Block size", f"Blocks in {budget_mib} MiB",
               f"Memory for {capacity} blocks", "KV not rebuilt per trace (GiB)", "Prefill TFLOPs saved per trace"]
    note = (f"Same trace and same {capacity}-block cache for every model: LRU {mean(res, 'LRU', 'hit_rate'):.1f}% -> "
            f"AgentKV {mean(res, 'AgentKV', 'hit_rate'):.1f}%, {saved_tokens:,.0f} fewer recomputed tokens per trace. "
            f"Llama-2-7B has no GQA (32 KV heads), so 600 MiB holds only "
            f"{int(budget_mib * MIB // model_bytes_per_block('llama-2-7b'))} blocks, not even one agent prefix (96 blocks + 16 system).")
    return markdown_table(headers, rows) + "\n\n" + note


TIERED_SETUPS = [
    # (label, policy, use CPU tier, prefetch)
    ("LRU, GPU only", "LRU", False, False),
    ("LRU + CPU tier", "LRU", True, False),
    ("AgentKV + CPU tier, no prefetch", "AgentKV", True, False),
    ("AgentKV + CPU tier + prefetch", "AgentKV", True, True),
    ("AgentKV-Markov + CPU tier + prefetch", "AgentKV-Markov", True, True),
]


def experiment_tiered(workflow, gpu_blocks, seeds, cpu_blocks=1024):
    """Returns (table, {label: averaged numbers})."""
    config = WorkloadConfig()
    static = config.system_tokens + config.prefix_tokens
    collected = {label: [] for label, _, _, _ in TIERED_SETUPS}

    for seed in seeds:
        dag, steps = make_trace(workflow, N_STEPS, seed, config)
        for label, policy, use_cpu, prefetch in TIERED_SETUPS:
            if use_cpu:
                s = run_workload_tiered(steps, dag, policy, gpu_blocks, cpu_blocks, static, prefetch)
                collected[label].append([s["gpu_hit_rate"], s["hit_rate"], s["cpu_hit_blocks"] / s["requests"],
                                         s["prefetch_accuracy"], s["recomputed_tokens"], s["stall_ms_per_request"]])
            else:
                s = run_workload(steps, dag, policy, gpu_blocks, static)
                collected[label].append([s["hit_rate"], s["hit_rate"], 0.0, 0.0,
                                         s["recomputed_tokens"], single_tier_stall_ms(s)])

    rows = []
    averages = {}
    for label, _, _, prefetch in TIERED_SETUPS:
        m = np.mean(collected[label], axis=0)
        averages[label] = {"gpu_hit_rate": m[0], "reuse_rate": m[1], "on_demand_blocks_per_request": m[2],
                           "prefetch_accuracy": m[3], "recomputed_tokens": m[4], "stall_ms_per_request": m[5]}
        rows.append([label, f"{m[0]:.1f}", f"{m[1]:.1f}", f"{m[2]:.1f}",
                     f"{m[3]:.1f}" if prefetch else "-", f"{m[4]:,.0f}", f"{m[5]:.2f}"])
    headers = ["Setup", "GPU hit %", "Reused (GPU+CPU) %", "Copied back on demand (blocks/request)",
               "Prefetch accuracy %", "Recomputed tokens", "Modeled stall (ms/request)"]
    return markdown_table(headers, rows), averages


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AgentKV benchmark")
    parser.add_argument("--quick", action="store_true", help="3 seeds instead of 10")
    parser.add_argument("--out", default="results", help="output folder")
    args = parser.parse_args()

    seeds = list(range(3)) if args.quick else list(range(10))
    os.makedirs(args.out, exist_ok=True)
    start = time.time()

    sections = ["# AgentKV benchmark results", "",
                f"{N_STEPS} requests per trace, {len(seeds)} seeds, Llama-3-8B (2 MiB per 16-token block). "
                "Hit rate = cached blocks / full prompt blocks. Every number is an exact count from the simulator."]

    print("Experiment 1+2: capacity sweep ...")
    cyc_table, cyc_raw, cyc_lru, cyc_akv = experiment_sweep("cyclic", seeds)
    br_table, br_raw, br_lru, br_akv = experiment_sweep("branching", seeds)

    cyc_series = {p: [r[p + "_hit_rate"] for r in cyc_raw] for p in ["LRU", "AgentKV", "Belady"]}
    br_series = {p: [r[p + "_hit_rate"] for r in br_raw] for p in ["LRU", "AgentKV", "Belady"]}

    sections += ["", "## Experiment 1+2: capacity sweep, 5 policies", "",
                 "### Cyclic: Planner -> Retriever -> Coder -> Reviewer -> Planner", "", cyc_table, "",
                 "### Branching: Reviewer -> Coder 30% / -> Planner 70%", "", br_table, "",
                 "```", ascii_chart("Cyclic workflow, block hit rate", CAPACITIES, cyc_series), "```"]

    print("Experiment 3: seeds ...")
    seed_table, seed_summary = experiment_seeds(300, list(range(10)))
    sections += ["", "## Experiment 3: branching workflow at 300 blocks (600 MiB), 10 seeds", "", seed_table, "",
                 f"AgentKV beat LRU on {seed_summary['agentkv_wins']}/{seed_summary['seeds']} seeds and was "
                 f"{seed_summary['gap_to_belady']:.1f} points below the Bélády oracle."]

    print("Experiment 4: workflows ...")
    wf_table, wf_raw = experiment_workflows(seeds)
    sections += ["", "## Experiment 4: every workflow at 75% of its working set", "", wf_table]

    print("Experiment 5: sensitivity ...")
    sens_table, sens_raw = experiment_sensitivity(seeds)
    sections += ["", "## Experiment 5: sensitivity (capacity = 75% of working set)", "",
                 sens_table]

    print("Experiment 6: models ...")
    sections += ["", "## Experiment 6: one scenario, four models (branching workflow, 360 blocks)", "",
                 experiment_models(seeds)]

    print("Experiment 7: GPU + CPU tiers ...")
    sections += ["", "## Experiment 7: GPU + CPU memory tiers with workflow-aware prefetching", "",
                 "Evicted agent prefixes move to a 1,024-block (2 GiB) CPU tier instead of being deleted; "
                 "one-time request tails are deleted. Stall = recomputing evicted blocks + copying CPU blocks "
                 "back while the request waits, modeled as A100 at 50% utilisation and PCIe 4.0 at 25 GB/s "
                 "(1 block: 1.64 ms to recompute vs 0.08 ms to copy). Prefetched copies overlap the previous "
                 "step, so they don't count as stall."]
    tiered = {}
    for workflow, gpu in [("cyclic", 300), ("branching", 300), ("branching", 360), ("fanout", 360)]:
        table, averages = experiment_tiered(workflow, gpu, seeds)
        tiered[f"{workflow}_{gpu}"] = averages
        sections += ["", f"### {workflow}, {gpu} GPU blocks ({blocks_to_mib(gpu):.0f} MiB)", "", table]

    # Sanity checks across the sweep
    all_rows = cyc_raw + br_raw
    never_worse = all(r["AgentKV_hit_rate"] >= r["LRU_hit_rate"] - 0.5 for r in all_rows)
    cyclic_equals_belady = all(abs(r["AgentKV_hit_rate"] - r["Belady_hit_rate"]) < 0.01 for r in cyc_raw)

    total_lru = cyc_lru + br_lru
    total_akv = cyc_akv + br_akv
    headline = {
        "cyclic_300": next(r for r in cyc_raw if r["capacity_blocks"] == 300),
        "branching_300": next(r for r in br_raw if r["capacity_blocks"] == 300),
        "sweep_recomputed_tokens_lru": total_lru,
        "sweep_recomputed_tokens_agentkv": total_akv,
        "sweep_recomputed_cut_percent": percent_cut(total_lru, total_akv),
        "sweep_tflops_saved": tokens_to_tflops(total_lru - total_akv),
        "sweep_kv_gib_saved": (total_lru - total_akv) * model_bytes_per_token("llama-3-8b") / GIB,
        "seeds_experiment": seed_summary,
        "agentkv_never_worse_than_lru": never_worse,
        "agentkv_equals_belady_on_cyclic": cyclic_equals_belady,
        "tiered": tiered,
    }

    sections += ["", "## Headline numbers", "",
                 f"- Across the full sweep (both workflows, 14 capacity points, per seed): recomputed prefill tokens "
                 f"{total_lru:,.0f} (LRU) -> {total_akv:,.0f} (AgentKV), a {headline['sweep_recomputed_cut_percent']:.0f}% cut, "
                 f"about {headline['sweep_tflops_saved']:,.0f} TFLOPs and {headline['sweep_kv_gib_saved']:.0f} GiB of KV not rebuilt.",
                 f"- Tiered, cyclic at 300 GPU blocks: modeled stall "
                 f"{tiered['cyclic_300']['LRU, GPU only']['stall_ms_per_request']:.1f} ms (LRU, GPU only) -> "
                 f"{tiered['cyclic_300']['LRU + CPU tier']['stall_ms_per_request']:.1f} ms (+ CPU tier) -> "
                 f"{tiered['cyclic_300']['AgentKV + CPU tier + prefetch']['stall_ms_per_request']:.2f} ms (AgentKV prefetch) per request.",
                 f"- Tiered, branching at 300 GPU blocks: GPU hit rate "
                 f"{tiered['branching_300']['AgentKV + CPU tier + prefetch']['gpu_hit_rate']:.1f}% (BFS prefetch) vs "
                 f"{tiered['branching_300']['AgentKV-Markov + CPU tier + prefetch']['gpu_hit_rate']:.1f}% (Markov prefetch).",
                 f"- Sanity check, AgentKV never below LRU in the sweep: {'PASS' if never_worse else 'FAIL'}",
                 f"- Sanity check, AgentKV == Bélády on the cyclic workflow: {'PASS' if cyclic_equals_belady else 'FAIL'}",
                 "", f"_Full run took {time.time() - start:.1f} s._"]

    report = "\n".join(sections)
    print()
    print(report)

    with open(os.path.join(args.out, "RESULTS.md"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"sweep": all_rows, "workflows": wf_raw, "sensitivity": sens_raw,
                   "headline": headline}, f, indent=2)
    with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
