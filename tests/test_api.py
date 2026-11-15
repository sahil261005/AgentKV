from fastapi.testclient import TestClient

from api.main import app, state
from core.config import BLOCK_SIZE

client = TestClient(app)

CYCLE = [
    {"src": "Planner", "dst": "Retriever"},
    {"src": "Retriever", "dst": "Coder"},
    {"src": "Coder", "dst": "Reviewer"},
    {"src": "Reviewer", "dst": "Planner"},
]
LABELS = {"Planner": 10, "Retriever": 20, "Coder": 30, "Reviewer": 40}


def prompt(agent):
    return [0] * BLOCK_SIZE + [LABELS[agent]] * (4 * BLOCK_SIZE)


def register(policy="AgentKV", num_blocks=13):
    return client.post("/v1/workflow", json={"edges": CYCLE, "num_blocks": num_blocks, "policy": policy})


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_step_before_workflow_is_rejected():
    state["engine"] = None
    response = client.post("/v1/step", json={"agent": "Planner", "tokens": [1, 2, 3]})
    assert response.status_code == 409


def test_register_workflow():
    response = register()
    assert response.status_code == 200
    body = response.json()
    assert body["agents"] == ["Planner", "Retriever", "Coder", "Reviewer"]
    assert body["memory_mib"] == 26.0


def test_bad_probabilities_are_rejected():
    edges = [{"src": "A", "dst": "B", "probability": 0.5}]
    response = client.post("/v1/workflow", json={"edges": edges})
    assert response.status_code == 422


def test_belady_is_not_allowed_online():
    response = client.post("/v1/workflow", json={"edges": CYCLE, "policy": "Belady"})
    assert response.status_code == 422


def test_unknown_agent():
    register()
    response = client.post("/v1/step", json={"agent": "Nobody", "tokens": [1] * 32})
    assert response.status_code == 400


def test_second_request_hits_cache():
    register()
    first = client.post("/v1/step", json={"agent": "Planner", "tokens": prompt("Planner")}).json()
    second = client.post("/v1/step", json={"agent": "Planner", "tokens": prompt("Planner")}).json()
    assert first["hit_tokens"] == 0
    assert second["hit_tokens"] == 5 * BLOCK_SIZE


def test_kvflow_scenario_through_the_api():
    # AgentKV keeps Planner at step 4, so step 5 is a full hit
    register("AgentKV")
    for agent in ["Planner", "Retriever", "Coder", "Reviewer"]:
        client.post("/v1/step", json={"agent": agent, "tokens": prompt(agent)})
    step5 = client.post("/v1/step", json={"agent": "Planner", "tokens": prompt("Planner")}).json()
    assert step5["hit_tokens"] == 5 * BLOCK_SIZE

    # LRU evicted Planner at step 4, so step 5 only hits the system prompt
    register("LRU")
    for agent in ["Planner", "Retriever", "Coder", "Reviewer"]:
        client.post("/v1/step", json={"agent": agent, "tokens": prompt(agent)})
    step5 = client.post("/v1/step", json={"agent": "Planner", "tokens": prompt("Planner")}).json()
    assert step5["hit_tokens"] == BLOCK_SIZE
    assert step5["recomputed_tokens"] == 4 * BLOCK_SIZE


def test_stats_and_metrics():
    register()
    client.post("/v1/step", json={"agent": "Planner", "tokens": prompt("Planner")})
    client.post("/v1/step", json={"agent": "Planner", "tokens": prompt("Planner")})

    stats = client.get("/v1/stats").json()
    assert stats["requests"] == 2
    assert stats["hit_rate"] == 50.0
    assert stats["used_blocks"] == 5

    text = client.get("/metrics").text
    assert 'agentkv_requests_total{policy="AgentKV"} 2' in text
    assert "# TYPE agentkv_block_hit_rate gauge" in text


def test_benchmark_endpoint():
    response = client.post("/v1/benchmark", json={"workflow": "cyclic", "capacities": [360], "seeds": 1, "n_steps": 40})
    assert response.status_code == 200
    row = response.json()["rows"][0]
    assert row["hit_rate"]["AgentKV"] > row["hit_rate"]["LRU"]


def test_markov_policy_through_the_api():
    response = register("AgentKV-Markov")
    assert response.status_code == 200
    for agent in ["Planner", "Retriever", "Coder", "Reviewer"]:
        client.post("/v1/step", json={"agent": agent, "tokens": prompt(agent)})
    step5 = client.post("/v1/step", json={"agent": "Planner", "tokens": prompt("Planner")}).json()
    # it starts from the registered graph, so it keeps Planner just like AgentKV
    assert step5["hit_tokens"] == 5 * BLOCK_SIZE
