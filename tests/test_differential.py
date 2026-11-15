"""
Differential tests: check the radix tree against a simple brute-force version.

The radix tree has the tricky logic (edge splitting, walking, locking), so we compare
it with code that is too simple to be wrong:

  - No eviction: the reference is a set of every block-prefix ever seen.
  - With eviction: before each request we list every prefix stored in the tree by
    brute force (all root-to-node paths) and compute the expected hit from that.

Prompts are built from a tiny "vocabulary" of blocks so they share prefixes a lot,
which forces many splits.
"""

import numpy as np

from core.agent_dag import make_cycle
from core.cache_engine import CacheEngine, make_policy
from core.config import BLOCK_SIZE


def random_prompt(rng):
    """3 to 10 blocks, each block picked from only 3 options, plus a partial block sometimes."""
    tokens = []
    for _ in range(rng.integers(3, 11)):
        tokens.extend([int(rng.integers(0, 3))] * BLOCK_SIZE)
    tokens.extend([7] * int(rng.integers(0, BLOCK_SIZE)))
    return tokens


def brute_force_hit_without_eviction(seen, tokens):
    hit = 0
    while (hit + 1) * BLOCK_SIZE <= len(tokens) and tuple(tokens[:(hit + 1) * BLOCK_SIZE]) in seen:
        hit += 1
    return hit


def all_cached_prefixes(tree):
    """Every block-aligned prefix currently stored in the tree, found by brute force."""
    prefixes = set()
    stack = [(child, []) for child in tree.root.children.values()]
    while stack:
        node, path = stack.pop()
        full = path + node.tokens
        for end in range(len(path) + BLOCK_SIZE, len(full) + 1, BLOCK_SIZE):
            prefixes.add(tuple(full[:end]))
        for child in node.children.values():
            stack.append((child, full))
    return prefixes


def test_matches_brute_force_without_eviction():
    rng = np.random.default_rng(123)
    engine = CacheEngine(100000, make_policy("LRU"))
    seen = set()

    for _ in range(300):
        tokens = random_prompt(rng)
        expected = brute_force_hit_without_eviction(seen, tokens)

        result = engine.run_request("A", tokens)
        assert result["hit_tokens"] == expected * BLOCK_SIZE

        for end in range(BLOCK_SIZE, len(tokens) // BLOCK_SIZE * BLOCK_SIZE + 1, BLOCK_SIZE):
            seen.add(tuple(tokens[:end]))
        engine.check_invariants()


def check_with_eviction(policy_name, capacity, seed):
    rng = np.random.default_rng(seed)
    agents = ["A", "B", "C"]
    engine = CacheEngine(capacity, make_policy(policy_name, make_cycle(agents)))

    for i in range(300):
        tokens = random_prompt(rng)
        cached = all_cached_prefixes(engine.tree)
        expected = brute_force_hit_without_eviction(cached, tokens)

        result = engine.run_request(agents[i % 3], tokens)
        assert result["hit_tokens"] == expected * BLOCK_SIZE
        engine.check_invariants()


def test_matches_brute_force_with_lru_eviction():
    for seed in range(5):
        check_with_eviction("LRU", capacity=12, seed=seed)


def test_matches_brute_force_with_agentkv_eviction():
    for seed in range(5):
        check_with_eviction("AgentKV", capacity=12, seed=seed)


def test_parallel_steps_keep_invariants():
    rng = np.random.default_rng(7)
    agents = ["A", "B", "C"]
    engine = CacheEngine(20, make_policy("AgentKV", make_cycle(agents)))
    for _ in range(200):
        step = [(a, random_prompt(rng)) for a in agents]
        engine.run_step(step)
        engine.check_invariants()
