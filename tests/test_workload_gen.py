from benchmarks.workload_gen import (
    WorkloadConfig, make_trace, save_trace_jsonl, load_trace_jsonl, infer_dag,
    working_set_blocks, WORKFLOWS,
)


def test_same_seed_same_trace():
    _, a = make_trace("branching", n_steps=50, seed=3)
    _, b = make_trace("branching", n_steps=50, seed=3)
    assert a == b


def test_different_seed_different_trace():
    _, a = make_trace("branching", n_steps=50, seed=1)
    _, b = make_trace("branching", n_steps=50, seed=2)
    assert a != b


def test_prompt_structure():
    config = WorkloadConfig()
    _, steps = make_trace("cyclic", n_steps=8, seed=0, config=config)
    planner_1 = steps[0][0][1]
    planner_2 = steps[4][0][1]
    retriever = steps[1][0][1]

    shared = config.system_tokens + config.prefix_tokens
    assert steps[0][0][0] == "Planner" and steps[4][0][0] == "Planner"
    assert planner_1[:shared] == planner_2[:shared]          # same agent: same prefix
    assert planner_1[shared:] != planner_2[shared:]          # but a fresh tail
    assert planner_1[:config.system_tokens] == retriever[:config.system_tokens]  # shared system prompt
    assert planner_1[:shared] != retriever[:shared]


def test_cyclic_order():
    _, steps = make_trace("cyclic", n_steps=8)
    agents = [step[0][0] for step in steps]
    assert agents == ["Planner", "Retriever", "Coder", "Reviewer"] * 2


def test_fanout_has_parallel_steps():
    _, steps = make_trace("fanout", n_steps=10)
    assert [a for a, _ in steps[1]] == ["Worker1", "Worker2", "Worker3"]


def test_all_workflows_build():
    for workflow in WORKFLOWS:
        dag, steps = make_trace(workflow, n_steps=20)
        assert len(steps) > 0
        for step in steps:
            for agent, tokens in step:
                assert agent in dag.agents()


def test_working_set():
    dag, _ = make_trace("cyclic", n_steps=4)
    assert working_set_blocks(dag, WorkloadConfig()) == 16 + 4 * 96


def test_jsonl_round_trip(tmp_path):
    _, steps = make_trace("fanout", n_steps=12)
    path = tmp_path / "trace.jsonl"
    save_trace_jsonl(steps, path)
    assert load_trace_jsonl(path) == steps


def test_infer_dag_from_cyclic_trace():
    _, steps = make_trace("cyclic", n_steps=40)
    dag = infer_dag(steps)
    assert dag.steps_to_execution("Reviewer") == {"Planner": 1, "Retriever": 2, "Coder": 3, "Reviewer": 0}


def test_infer_dag_learns_probabilities():
    _, steps = make_trace("branching", n_steps=4000, seed=0)
    dag = infer_dag(steps)
    probs = dict((dst, p) for dst, p in dag.edges["Reviewer"])
    assert 0.25 < probs["Coder"] < 0.35
