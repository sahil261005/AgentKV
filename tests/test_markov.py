import math

from core.agent_dag import AgentDAG, make_cycle, dag_probabilities, expected_steps
from core.block_pool import BlockPool
from core.config import BLOCK_SIZE
from core.eviction_policy import AgentKVPolicy, MarkovAgentKVPolicy, choose_victim
from core.radix_tree import RadixTree

AGENTS = ["Planner", "Retriever", "Coder", "Reviewer"]


def branching_dag():
    dag = AgentDAG()
    dag.add_edge("Planner", "Retriever")
    dag.add_edge("Retriever", "Coder")
    dag.add_edge("Coder", "Reviewer")
    dag.add_edge("Reviewer", "Coder", 0.3)
    dag.add_edge("Reviewer", "Planner", 0.7)
    return dag


def test_markov_equals_bfs_on_a_fixed_cycle():
    dag = make_cycle(AGENTS)
    markov = expected_steps(dag_probabilities(dag), "Reviewer")
    bfs = dag.steps_to_execution("Reviewer")
    for agent in AGENTS:
        assert abs(markov[agent] - bfs[agent]) < 1e-9


def test_markov_on_branching_graph():
    # worked out by hand:
    #   h(Planner)   = 1 + 0.3 * (1 + h)        -> 1.857
    #   h(Coder)     = 1 + 0.7 * 2              -> 2.4
    #   h(Retriever) = 1 + 0.7 + 0.3 * (1 + h)  -> 2.857
    d = expected_steps(dag_probabilities(branching_dag()), "Reviewer")
    assert abs(d["Planner"] - 1.3 / 0.7) < 1e-9
    assert abs(d["Coder"] - 2.4) < 1e-9
    assert abs(d["Retriever"] - 2.0 / 0.7) < 1e-9
    assert d["Reviewer"] == 0


def test_unreachable_agent_is_infinite():
    probs = {"A": {"B": 1.0}, "B": {"A": 1.0}, "C": {"A": 1.0}}
    d = expected_steps(probs, "A")
    assert d["B"] == 1
    assert d["C"] == math.inf


def make_tokens(labels):
    tokens = []
    for label in labels:
        tokens.extend([label] * BLOCK_SIZE)
    return tokens


def build_tree():
    """Planner prefix used at time 1, Coder prefix used at time 3."""
    tree = RadixTree()
    pool = BlockPool(20)
    leaves = {}
    for time, (agent, label) in [(1, ("Planner", 10)), (3, ("Coder", 30))]:
        tokens = make_tokens([1, label])
        _, node, pos = tree.match_prefix(tokens)
        rest = tokens[pos:]
        leaf = tree.insert_after(node, rest, pool.allocate(len(rest) // BLOCK_SIZE), agent, time)
        tree.touch(leaf, time, agent)
        leaves[agent] = leaf
    return tree, leaves


def test_markov_breaks_the_bfs_tie_using_probabilities():
    tree, leaves = build_tree()

    # BFS: Planner and Coder are both 1 hop from Reviewer -> tie -> LRU picks older Planner
    bfs = AgentKVPolicy(branching_dag())
    bfs.start_step(0, ["Reviewer"])
    assert choose_victim(bfs, tree.evictable_leaves()) is leaves["Planner"]

    # Markov: Planner comes next 70% of the time (1.86 steps) vs Coder (2.4) -> evict Coder
    markov = MarkovAgentKVPolicy(branching_dag())
    markov.start_step(0, ["Reviewer"])
    assert choose_victim(markov, tree.evictable_leaves()) is leaves["Coder"]


def test_learned_policy_starts_like_lru_then_learns_the_cycle():
    policy = MarkovAgentKVPolicy()
    policy.start_step(0, ["Reviewer"])
    assert policy.distances.get("Planner", math.inf) == math.inf   # knows nothing yet

    for agent in AGENTS * 3:
        policy.observe_step([agent])
    policy.start_step(0, ["Reviewer"])
    assert abs(policy.distances["Planner"] - 1) < 1e-9
    assert abs(policy.distances["Retriever"] - 2) < 1e-9
    assert abs(policy.distances["Coder"] - 3) < 1e-9


def test_parallel_workers_are_one_step_away():
    # Planner -> (3 workers in parallel) -> Merger -> Planner
    policy = MarkovAgentKVPolicy()
    workers = ["Worker1", "Worker2", "Worker3"]
    for _ in range(3):
        policy.observe_step(["Planner"])
        policy.observe_step(workers)
        policy.observe_step(["Merger"])

    policy.start_step(0, ["Planner"])
    # all three workers run in the very next step, not "1 of 3 per step"
    assert policy.distances["Worker1"] == 1
    assert policy.distances["Worker3"] == 1
    assert policy.distances["Merger"] == 2
