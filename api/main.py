"""
A small HTTP API around the cache engine.

    uvicorn api.main:app --reload
    open http://127.0.0.1:8000/docs

Endpoints
    POST /v1/workflow   register the agent graph, cache size and policy (resets the cache)
    POST /v1/step       send one agent request (token IDs) through the cache
    GET  /v1/stats      counters so far
    POST /v1/benchmark  run a small policy comparison on a generated workload
    GET  /metrics       the same counters in Prometheus text format
    GET  /health
"""

import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from api.schemas import (
    WorkflowRequest, WorkflowResponse, StepRequest, StepResponse, StatsResponse,
    BenchmarkRequest, BenchmarkResponse, BenchmarkRow,
)
from benchmarks.run_eval import run_policies, mean
from core.agent_dag import AgentDAG
from core.cache_engine import CacheEngine, make_policy, POLICY_NAMES
from core.config import blocks_to_mib, BYTES_PER_BLOCK, MIB
from core.eviction_policy import MarkovAgentKVPolicy

app = FastAPI(title="AgentKV", version="1.0.0",
              description="DAG-aware KV-cache simulator for multi-agent LLM workflows")

# One engine for the whole server. FastAPI runs normal (non-async) endpoints in a
# thread pool, so two requests could touch the engine at once. The lock prevents that.
engine_lock = threading.Lock()
state = {"engine": None, "dag": None, "static_tokens": None}


def get_engine():
    if state["engine"] is None:
        raise HTTPException(status_code=409, detail="register a workflow first: POST /v1/workflow")
    return state["engine"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/workflow", response_model=WorkflowResponse)
def register_workflow(req: WorkflowRequest):
    dag = AgentDAG()
    for edge in req.edges:
        dag.add_edge(edge.src, edge.dst, edge.probability)

    with engine_lock:
        state["dag"] = dag
        state["static_tokens"] = req.static_tokens
        if req.policy == "AgentKV-Markov":
            policy = MarkovAgentKVPolicy(dag)    # start from the registered graph, keep learning
        else:
            policy = make_policy(req.policy, dag)
        state["engine"] = CacheEngine(req.num_blocks, policy)

    return WorkflowResponse(agents=dag.agents(), num_blocks=req.num_blocks,
                            memory_mib=blocks_to_mib(req.num_blocks), policy=req.policy)


@app.post("/v1/step", response_model=StepResponse)
def run_step(req: StepRequest):
    with engine_lock:
        engine = get_engine()
        if req.agent not in state["dag"].agents():
            raise HTTPException(status_code=400, detail=f"unknown agent '{req.agent}'")
        result = engine.run_request(req.agent, req.tokens, state["static_tokens"])
        return StepResponse(**result, free_blocks=engine.pool.num_free())


@app.get("/v1/stats", response_model=StatsResponse)
def stats():
    with engine_lock:
        engine = get_engine()
        s = engine.summary()
        return StatsResponse(
            policy=s["policy"],
            capacity_blocks=s["capacity_blocks"],
            used_blocks=engine.pool.num_used(),
            free_blocks=engine.pool.num_free(),
            used_mib=engine.pool.num_used() * BYTES_PER_BLOCK / MIB,
            requests=s["requests"],
            hit_rate=s["hit_rate"],
            hit_blocks=s["hit_blocks"],
            total_blocks=s["total_blocks"],
            computed_tokens=s["computed_tokens"],
            recomputed_tokens=s["recomputed_tokens"],
            evictions=s["evictions"],
            evicted_blocks=s["evicted_blocks"],
        )


@app.post("/v1/benchmark", response_model=BenchmarkResponse)
def benchmark(req: BenchmarkRequest):
    rows = []
    for cap in req.capacities:
        res = run_policies(req.workflow, cap, list(range(req.seeds)), n_steps=req.n_steps)
        rows.append(BenchmarkRow(
            capacity_blocks=cap,
            capacity_mib=blocks_to_mib(cap),
            hit_rate={p: round(mean(res, p, "hit_rate"), 2) for p in POLICY_NAMES},
            recomputed_tokens={p: mean(res, p, "recomputed_tokens") for p in POLICY_NAMES},
        ))
    return BenchmarkResponse(workflow=req.workflow, seeds=req.seeds, rows=rows)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus text format, written by hand (no extra library needed)."""
    with engine_lock:
        engine = state["engine"]
        if engine is None:
            return "# no workflow registered yet\n"
        s = engine.summary()
        values = [
            ("agentkv_requests_total", "counter", "Requests processed", s["requests"]),
            ("agentkv_hit_blocks_total", "counter", "Prompt blocks served from cache", s["hit_blocks"]),
            ("agentkv_prompt_blocks_total", "counter", "Full prompt blocks seen", s["total_blocks"]),
            ("agentkv_computed_tokens_total", "counter", "Tokens that needed prefill", s["computed_tokens"]),
            ("agentkv_recomputed_tokens_total", "counter", "Tokens recomputed after eviction", s["recomputed_tokens"]),
            ("agentkv_evicted_blocks_total", "counter", "Blocks evicted", s["evicted_blocks"]),
            ("agentkv_block_hit_rate", "gauge", "Block hit rate in percent", round(s["hit_rate"], 4)),
            ("agentkv_used_blocks", "gauge", "Blocks currently in use", engine.pool.num_used()),
            ("agentkv_free_blocks", "gauge", "Blocks currently free", engine.pool.num_free()),
        ]
    lines = []
    for name, kind, help_text, value in values:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        lines.append(f'{name}{{policy="{s["policy"]}"}} {value}')
    return "\n".join(lines) + "\n"
