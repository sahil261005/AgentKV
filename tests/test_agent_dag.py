import math

import numpy as np
import pytest

from core.agent_dag import AgentDAG, make_cycle


def branching_dag():
    dag = AgentDAG()
    dag.add_edge("Planner", "Retriever")
    dag.add_edge("Retriever", "Coder")
    dag.add_edge("Coder", "Reviewer")
    dag.add_edge("Reviewer", "Coder", 0.3)     # failed review: go back to Coder
    dag.add_edge("Reviewer", "Planner", 0.7)   # passed: next task
    return dag


def test_cycle_distances_from_reviewer():
    dag = make_cycle(["Planner", "Retriever", "Coder", "Reviewer"])
    d = dag.steps_to_execution("Reviewer")
    assert d == {"Reviewer": 0, "Planner": 1, "Retriever": 2, "Coder": 3}


def test_cycle_distances_from_planner():
    dag = make_cycle(["Planner", "Retriever", "Coder", "Reviewer"])
    d = dag.steps_to_execution("Planner")
    assert d == {"Planner": 0, "Retriever": 1, "Coder": 2, "Reviewer": 3}


def test_branching_distances_from_reviewer():
    d = branching_dag().steps_to_execution("Reviewer")
    # both possible next agents are 1 hop away
    assert d["Planner"] == 1
    assert d["Coder"] == 1
    assert d["Retriever"] == 2


def test_unreachable_agent_is_infinite():
    dag = AgentDAG()
    dag.add_edge("A", "B")
    dag.add_agent("Lonely")
    d = dag.steps_to_execution("A")
    assert d["B"] == 1
    assert d["Lonely"] == math.inf


def test_sample_next_follows_probabilities():
    dag = branching_dag()
    rng = np.random.default_rng(0)
    picks = [dag.sample_next("Reviewer", rng) for _ in range(10000)]
    coder_share = picks.count("Coder") / len(picks)
    assert 0.28 < coder_share < 0.32


def test_sample_next_same_seed_same_result():
    dag = branching_dag()
    a = [dag.sample_next("Reviewer", np.random.default_rng(42)) for _ in range(5)]
    b = [dag.sample_next("Reviewer", np.random.default_rng(42)) for _ in range(5)]
    assert a == b


def test_bad_probabilities_raise():
    dag = AgentDAG()
    dag.add_edge("A", "B", 0.5)
    dag.add_edge("A", "C", 0.2)
    with pytest.raises(ValueError):
        dag.sample_next("A", np.random.default_rng(0))
