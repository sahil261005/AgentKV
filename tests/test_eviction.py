"""
End-to-end eviction tests: the KVFlow "cyclic LRU thrashing" scenario.

Small version of the real setup, so the numbers are easy to check by hand:
  - shared system prompt: 1 block
  - each agent's prefix:   4 blocks
  - cache capacity: 1 + 3 * 4 = 13 blocks -> room for 3 agents, not 4
"""

from core.agent_dag import make_cycle
from core.cache_engine import CacheEngine, make_policy, run_workload
from core.config import BLOCK_SIZE

AGENTS = ["Planner", "Retriever", "Coder", "Reviewer"]
LABELS = {"Planner": 10, "Retriever": 20, "Coder": 30, "Reviewer": 40}


def prompt(agent):
    tokens = [0] * BLOCK_SIZE                       # system prompt
    tokens += [LABELS[agent]] * (4 * BLOCK_SIZE)    # agent prefix
    return tokens


def is_cached(engine, agent):
    _, _, matched = engine.tree.match_prefix(prompt(agent))
    return matched == len(prompt(agent))


def run_first_4_steps(policy_name):
    dag = make_cycle(AGENTS)
    prompts = [prompt(a) for a in AGENTS * 3]
    engine = CacheEngine(13, make_policy(policy_name, dag, prompts))
    for agent in AGENTS:
        engine.run_request(agent, prompt(agent))
    return engine


def test_lru_evicts_planner_at_step_4():
    engine = run_first_4_steps("LRU")
    # Planner has the oldest timestamp, so LRU throws it out right before it is needed again
    assert not is_cached(engine, "Planner")
    assert is_cached(engine, "Reviewer")


def test_agentkv_keeps_planner_at_step_4():
    engine = run_first_4_steps("AgentKV")
    # From Reviewer: Planner is 1 hop away (keep), Coder is 3 hops away (evict)
    assert is_cached(engine, "Planner")
    assert not is_cached(engine, "Coder")


def cycle_steps(n_cycles):
    return [[(a, prompt(a))] for a in AGENTS * n_cycles]


def test_lru_thrashes_on_every_step():
    result = run_workload(cycle_steps(5), make_cycle(AGENTS), "LRU", 13)
    # Only the system prompt ever hits: 1 block per request, except the very first one.
    assert result["hit_blocks"] == 20 - 1


def test_agentkv_beats_lru_and_matches_belady_on_cycle():
    dag = make_cycle(AGENTS)
    lru = run_workload(cycle_steps(5), dag, "LRU", 13)
    agentkv = run_workload(cycle_steps(5), dag, "AgentKV", 13)
    belady = run_workload(cycle_steps(5), dag, "Belady", 13)

    assert agentkv["hit_rate"] > lru["hit_rate"]
    assert agentkv["recomputed_tokens"] < lru["recomputed_tokens"]
    # on a fixed cycle "hops until next run" is exactly "steps until next use"
    assert agentkv["hit_blocks"] == belady["hit_blocks"]


def test_everything_fits_means_policies_tie():
    dag = make_cycle(AGENTS)
    results = [run_workload(cycle_steps(5), dag, p, 17)["hit_blocks"] for p in ["LRU", "AgentKV", "Belady"]]
    assert results[0] == results[1] == results[2]


def test_recomputed_tokens_counts_evicted_then_needed_again():
    engine = run_first_4_steps("LRU")
    result = engine.run_request("Planner", prompt("Planner"))
    # Planner's 4 prefix blocks were cached at step 1, evicted at step 4, needed at step 5
    assert result["recomputed_tokens"] == 4 * BLOCK_SIZE
    assert result["hit_tokens"] == BLOCK_SIZE   # only the system prompt


def test_parallel_requests_are_not_evicted_from_under_each_other():
    # capacity 9: system (1) + two agents (4 + 4). Both run at the same time.
    engine = CacheEngine(9, make_policy("LRU"))
    step = [("Planner", prompt("Planner")), ("Coder", prompt("Coder"))]
    engine.run_step(step)
    engine.check_invariants()
    assert is_cached(engine, "Planner")
    assert is_cached(engine, "Coder")


def test_memory_full_of_locked_requests_does_not_crash():
    # capacity 9 but three requests at once need 13 blocks: the third can't be fully cached
    engine = CacheEngine(9, make_policy("LRU"))
    step = [("Planner", prompt("Planner")), ("Coder", prompt("Coder")), ("Reviewer", prompt("Reviewer"))]
    engine.run_step(step)
    engine.check_invariants()
    assert engine.uncached_blocks == 4
    assert engine.evictions == 0     # nothing was evictable, everything was locked
