"""
Tests for the GPU + CPU tiered engine.
Same small setup as test_eviction.py: system prompt 1 block, each agent 4 blocks,
GPU room for 3 agents (13 blocks).
"""

import numpy as np

from core.agent_dag import make_cycle
from core.cache_engine import make_policy
from core.config import BLOCK_SIZE
from core.tiered_engine import TieredCacheEngine

AGENTS = ["Planner", "Retriever", "Coder", "Reviewer"]
LABELS = {"Planner": 10, "Retriever": 20, "Coder": 30, "Reviewer": 40}


def prompt(agent):
    return [0] * BLOCK_SIZE + [LABELS[agent]] * (4 * BLOCK_SIZE)


def make_engine(policy_name, gpu=13, cpu=100):
    return TieredCacheEngine(gpu, cpu, make_policy(policy_name, make_cycle(AGENTS)))


def run_cycles(engine, n_cycles):
    for agent in AGENTS * n_cycles:
        engine.run_step([(agent, prompt(agent))])
        engine.check_invariants()


def test_evicted_data_goes_to_cpu_not_lost():
    engine = make_engine("LRU")
    run_cycles(engine, 1)              # LRU offloads Planner at step 4
    result = engine.run_request("Planner", prompt("Planner"))

    assert result["hit_tokens"] == 5 * BLOCK_SIZE    # everything reused...
    assert result["recomputed_tokens"] == 0          # ...nothing recomputed
    assert engine.cpu_hit_blocks == 4                # but 4 blocks were copied back while waiting


def test_lru_has_to_copy_back_on_every_step():
    engine = make_engine("LRU")
    run_cycles(engine, 4)
    assert engine.recomputed_tokens == 0
    assert engine.cpu_hit_blocks > 0
    assert engine.prefetched_blocks == 0             # LRU can't predict, so it never prefetches


def test_agentkv_prefetch_removes_the_waiting():
    engine = make_engine("AgentKV")
    run_cycles(engine, 1)                            # warm up
    copied_back_before = engine.cpu_hit_blocks
    run_cycles(engine, 3)

    assert engine.cpu_hit_blocks == copied_back_before   # no more on-demand copies after warm-up
    assert engine.prefetched_blocks > 0
    # every prefetch gets used, except the last two (4 blocks each): those were for
    # steps after the trace ends (AgentKV prefetches up to 2 agents ahead)
    assert engine.prefetched_blocks - engine.prefetch_used_blocks == 8
    assert engine.recomputed_tokens == 0


def test_small_cpu_drops_data_but_stays_consistent():
    engine = make_engine("LRU", gpu=13, cpu=4)
    run_cycles(engine, 4)
    assert engine.dropped_blocks > 0
    assert engine.recomputed_tokens > 0


def random_prompt(rng):
    tokens = []
    for _ in range(rng.integers(3, 11)):
        tokens.extend([int(rng.integers(0, 3))] * BLOCK_SIZE)
    return tokens


def test_random_workload_keeps_invariants():
    for policy in ["LRU", "AgentKV", "AgentKV-Markov"]:
        rng = np.random.default_rng(1)
        agents = ["A", "B", "C"]
        engine = TieredCacheEngine(12, 15, make_policy(policy, make_cycle(agents)))
        for i in range(300):
            if i % 5 == 0:
                step = [(a, random_prompt(rng)) for a in agents]     # parallel step
            else:
                step = [(agents[i % 3], random_prompt(rng))]
            engine.run_step(step)
            engine.check_invariants()
