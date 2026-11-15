"""
Unit tests for the three policies on a hand-built tree.
(The full end-to-end "Planner is kept at step 4" test lives in test_eviction.py.)
"""

import math

from core.agent_dag import make_cycle
from core.block_pool import BlockPool
from core.config import BLOCK_SIZE
from core.eviction_policy import (
    LRUPolicy, AgentKVPolicy, BeladyPolicy, choose_victim, block_hashes,
)
from core.radix_tree import RadixTree

AGENTS = ["Planner", "Retriever", "Coder", "Reviewer"]


def make_tokens(block_labels):
    tokens = []
    for label in block_labels:
        tokens.extend([label] * BLOCK_SIZE)
    return tokens


def build_tree():
    """
    root -> [system] -> Planner prefix   (used at time 1)
                     -> Retriever prefix (used at time 2)
                     -> Coder prefix     (used at time 3)
    """
    tree = RadixTree()
    pool = BlockPool(50)
    leaves = {}
    for time, (agent, label) in enumerate([("Planner", 10), ("Retriever", 20), ("Coder", 30)], start=1):
        tokens = make_tokens([1, label, label])
        _, node, pos = tree.match_prefix(tokens)
        rest = tokens[pos:]
        leaf = tree.insert_after(node, rest, pool.allocate(len(rest) // BLOCK_SIZE), agent, time)
        tree.touch(leaf, time, agent)
        leaves[agent] = leaf
    return tree, leaves


def test_lru_evicts_oldest():
    tree, leaves = build_tree()
    policy = LRUPolicy()
    policy.start_step(4, ["Reviewer"])
    # Planner was used longest ago, so LRU throws it away (the KVFlow problem)
    assert choose_victim(policy, tree.evictable_leaves()) is leaves["Planner"]


def test_agentkv_evicts_furthest_agent():
    tree, leaves = build_tree()
    policy = AgentKVPolicy(make_cycle(AGENTS))
    policy.start_step(4, ["Reviewer"])
    # From Reviewer: Planner is 1 hop away, Retriever 2, Coder 3 -> evict Coder
    assert choose_victim(policy, tree.evictable_leaves()) is leaves["Coder"]


def test_agentkv_shared_node_uses_closest_agent():
    tree, leaves = build_tree()
    policy = AgentKVPolicy(make_cycle(AGENTS))
    policy.start_step(4, ["Reviewer"])
    system_node = leaves["Planner"].parent
    # the system prompt is used by Planner (1 hop), Retriever (2) and Coder (3)
    assert policy.distance_of(system_node) == 1


def test_agentkv_ties_broken_by_lru():
    tree, leaves = build_tree()
    policy = AgentKVPolicy(make_cycle(AGENTS))
    policy.distances = {"Planner": 5, "Retriever": 5, "Coder": 5}   # all equal
    assert choose_victim(policy, tree.evictable_leaves()) is leaves["Planner"]


def test_agentkv_unknown_agent_is_evicted_first():
    tree, leaves = build_tree()
    policy = AgentKVPolicy(make_cycle(["Retriever", "Coder", "Reviewer"]))  # no Planner
    policy.start_step(4, ["Reviewer"])
    assert policy.distance_of(leaves["Planner"]) == math.inf
    assert choose_victim(policy, tree.evictable_leaves()) is leaves["Planner"]


def test_block_hash_depends_on_previous_blocks():
    a = block_hashes(make_tokens([1, 2]))
    b = block_hashes(make_tokens([9, 2]))
    assert len(a) == 2
    # same second block, different first block -> different hash
    assert a[1] != b[1]
    assert block_hashes(make_tokens([1, 2])) == a


def test_belady_evicts_node_used_furthest_in_future():
    tree, leaves = build_tree()
    future = [
        make_tokens([1, 10, 10]),   # pos 0..3: already happened
        make_tokens([1, 20, 20]),
        make_tokens([1, 30, 30]),
        make_tokens([1, 40, 40]),
        make_tokens([1, 20, 20]),   # pos 4: Retriever soon
        make_tokens([1, 10, 10]),   # pos 5: Planner
        make_tokens([1, 30, 30]),   # pos 6: Coder last
    ]
    policy = BeladyPolicy(future)
    policy.start_step(3, ["Reviewer"])

    assert policy.steps_until_next_use(leaves["Retriever"]) == 1
    assert policy.steps_until_next_use(leaves["Planner"]) == 2
    assert policy.steps_until_next_use(leaves["Coder"]) == 3
    assert choose_victim(policy, tree.evictable_leaves()) is leaves["Coder"]


def test_belady_never_used_again_is_infinite():
    tree, leaves = build_tree()
    prompts = [make_tokens([1, 10, 10]), make_tokens([1, 20, 20]), make_tokens([1, 30, 30])]
    policy = BeladyPolicy(prompts)
    policy.start_step(2, ["Coder"])
    assert policy.steps_until_next_use(leaves["Planner"]) == math.inf
