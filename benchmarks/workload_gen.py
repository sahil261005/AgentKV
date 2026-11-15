"""
Workload generator: builds multi-agent traces as token IDs.

A trace is a list of steps. A step is a list of (agent, tokens) pairs that run at
the same time (usually just one; the fan-out workflow runs 3 in parallel).

Every prompt looks like what an agent framework actually sends:

    [ shared system prompt ][ agent's own prefix: role, tools, examples ][ fresh task tail ]
          256 tokens               ~1536 tokens (96 blocks)                64-256 tokens

The first two parts repeat every time that agent runs, so they can be cached.
The tail is new every time (latest messages / tool results), so it always misses.

Token IDs are random integers from the Llama-3 vocabulary, made with a seeded
NumPy generator, so the same seed always gives the exact same trace.
"""

import json
from dataclasses import dataclass

import numpy as np

from core.agent_dag import AgentDAG, make_cycle

VOCAB_SIZE = 128256     # Llama-3 tokenizer vocabulary size

FOUR_AGENTS = ["Planner", "Retriever", "Coder", "Reviewer"]
EXTRA_AGENTS = ["Tester", "Critic", "Summarizer", "Deployer"]

WORKFLOWS = ["cyclic", "branching", "react", "fanout", "random"]


@dataclass
class WorkloadConfig:
    system_tokens: int = 256
    prefix_tokens: int = 1536
    tail_min: int = 64
    tail_max: int = 256
    num_agents: int = 4          # only used by the "cyclic" workflow


# ----------------------------------------------------------------------
# Workflow graphs
# ----------------------------------------------------------------------
def cyclic_dag(num_agents=4):
    """Planner -> Retriever -> Coder -> Reviewer -> Planner (more agents if asked)."""
    names = (FOUR_AGENTS + EXTRA_AGENTS)[:num_agents]
    return make_cycle(names)


def branching_dag():
    """Like the cycle, but a failed review sends the work back to Coder 30% of the time."""
    dag = AgentDAG()
    dag.add_edge("Planner", "Retriever")
    dag.add_edge("Retriever", "Coder")
    dag.add_edge("Coder", "Reviewer")
    dag.add_edge("Reviewer", "Coder", 0.3)
    dag.add_edge("Reviewer", "Planner", 0.7)
    return dag


def react_dag():
    """ReAct-style hub: Planner calls a tool, the tool always returns to Planner."""
    dag = AgentDAG()
    dag.add_edge("Planner", "Search", 0.5)
    dag.add_edge("Planner", "Calculator", 0.3)
    dag.add_edge("Planner", "Writer", 0.2)
    dag.add_edge("Search", "Planner")
    dag.add_edge("Calculator", "Planner")
    dag.add_edge("Writer", "Planner")
    return dag


def fanout_dag():
    """Planner starts 3 workers in parallel, a Merger combines their answers."""
    dag = AgentDAG()
    for worker in ["Worker1", "Worker2", "Worker3"]:
        dag.add_edge("Planner", worker)
        dag.add_edge(worker, "Merger")
    dag.add_edge("Merger", "Planner")
    return dag


def random_dag():
    """Any agent can follow any other with equal chance. The graph tells us nothing."""
    dag = AgentDAG()
    for src in FOUR_AGENTS:
        others = [a for a in FOUR_AGENTS if a != src]
        for dst in others:
            dag.add_edge(src, dst, 1.0 / len(others))
    return dag


def make_dag(workflow, config):
    if workflow == "cyclic":
        return cyclic_dag(config.num_agents)
    if workflow == "branching":
        return branching_dag()
    if workflow == "react":
        return react_dag()
    if workflow == "fanout":
        return fanout_dag()
    if workflow == "random":
        return random_dag()
    raise ValueError(f"unknown workflow {workflow}")


# ----------------------------------------------------------------------
# Which agent runs at each step
# ----------------------------------------------------------------------
def agent_schedule(workflow, dag, n_steps, rng):
    """Returns a list of steps, each step a list of agent names."""
    if workflow == "fanout":
        schedule = []
        requests = 0
        while requests < n_steps:
            for group in [["Planner"], ["Worker1", "Worker2", "Worker3"], ["Merger"]]:
                schedule.append(group)
                requests += len(group)
        return schedule

    current = dag.agents()[0]
    schedule = []
    for _ in range(n_steps):
        schedule.append([current])
        current = dag.sample_next(current, rng)
    return schedule


# ----------------------------------------------------------------------
# Building the token trace
# ----------------------------------------------------------------------
def make_trace(workflow, n_steps=200, seed=0, config=None):
    """Returns (dag, steps) where steps is a list of [(agent, tokens), ...]."""
    if config is None:
        config = WorkloadConfig()

    dag = make_dag(workflow, config)
    rng = np.random.default_rng(seed)

    system = rng.integers(0, VOCAB_SIZE, config.system_tokens).tolist()
    prefixes = {}
    for agent in dag.agents():
        prefixes[agent] = rng.integers(0, VOCAB_SIZE, config.prefix_tokens).tolist()

    steps = []
    for group in agent_schedule(workflow, dag, n_steps, rng):
        step = []
        for agent in group:
            tail_len = int(rng.integers(config.tail_min, config.tail_max + 1))
            tail = rng.integers(0, VOCAB_SIZE, tail_len).tolist()
            step.append((agent, system + prefixes[agent] + tail))
        steps.append(step)
    return dag, steps


def working_set_blocks(dag, config):
    """Blocks needed to keep the system prompt and every agent's prefix cached at once."""
    return config.system_tokens // 16 + len(dag.agents()) * (config.prefix_tokens // 16)


# ----------------------------------------------------------------------
# Replaying real traces
# ----------------------------------------------------------------------
def save_trace_jsonl(steps, path):
    """One JSON object per request: {"step": 0, "agent": "Planner", "tokens": [...]}."""
    with open(path, "w") as f:
        for i, step in enumerate(steps):
            for agent, tokens in step:
                f.write(json.dumps({"step": i, "agent": agent, "tokens": tokens}) + "\n")


def load_trace_jsonl(path):
    """Read a trace saved by save_trace_jsonl (or logged from a real agent run)."""
    steps = []
    last_step = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["step"] != last_step:
                steps.append([])
                last_step = record["step"]
            steps[-1].append((record["agent"], record["tokens"]))
    return steps


def infer_dag(steps):
    """
    Learn the workflow graph from a trace by counting which agent follows which.
    Useful for real logs where nobody wrote the graph down.
    """
    counts = {}
    for i in range(len(steps) - 1):
        for src, _ in steps[i]:
            for dst, _ in steps[i + 1]:
                counts.setdefault(src, {})
                counts[src][dst] = counts[src].get(dst, 0) + 1

    dag = AgentDAG()
    for step in steps:
        for agent, _ in step:
            dag.add_agent(agent)
    for src, targets in counts.items():
        total = sum(targets.values())
        for dst, c in targets.items():
            dag.add_edge(src, dst, c / total)
    return dag
