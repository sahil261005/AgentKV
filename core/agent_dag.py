"""
AgentDAG: the multi-agent workflow as a directed graph.

Each agent is a node. An edge A -> B means "after A runs, B can run next".
Edges can have a probability (e.g. Reviewer -> Coder 30%, Reviewer -> Planner 70%).

The graph can have cycles (Planner -> ... -> Reviewer -> Planner), so it is not
strictly a DAG. We keep the name because the KVFlow paper uses it.

The key function is steps_to_execution(): starting from the agent that is running
now, a breadth-first search (BFS) tells us how many hops away every other agent is.
An agent 1 hop away will run very soon, so its KV cache is valuable.
An agent 3 hops away runs much later, so it is the best thing to evict.
"""

import math
from collections import deque

import numpy as np


class AgentDAG:
    def __init__(self):
        # agent name -> list of (next agent name, probability)
        self.edges = {}

    def add_agent(self, name):
        if name not in self.edges:
            self.edges[name] = []

    def add_edge(self, src, dst, probability=1.0):
        self.add_agent(src)
        self.add_agent(dst)
        self.edges[src].append((dst, probability))

    def agents(self):
        return list(self.edges.keys())

    def next_agents(self, agent):
        return [dst for dst, _ in self.edges[agent]]

    def steps_to_execution(self, current):
        """
        BFS from the current agent.
        Returns {agent: number of hops}. The current agent is 0.
        Agents that can never be reached get math.inf.
        """
        distances = {name: math.inf for name in self.edges}
        distances[current] = 0

        queue = deque([current])
        while queue:
            agent = queue.popleft()
            for nxt in self.next_agents(agent):
                if distances[nxt] == math.inf:      # not visited yet
                    distances[nxt] = distances[agent] + 1
                    queue.append(nxt)
        return distances

    def sample_next(self, current, rng):
        """Pick the next agent using the edge probabilities. rng is a NumPy Generator."""
        options = self.edges[current]
        if not options:
            return None

        names = [dst for dst, _ in options]
        probs = [p for _, p in options]
        if abs(sum(probs) - 1.0) > 1e-9:
            raise ValueError(f"probabilities out of {current} must add up to 1, got {sum(probs)}")

        index = rng.choice(len(names), p=probs)
        return names[index]


def make_cycle(agent_names):
    """Helper: A -> B -> C -> ... -> A."""
    dag = AgentDAG()
    for i in range(len(agent_names)):
        src = agent_names[i]
        dst = agent_names[(i + 1) % len(agent_names)]
        dag.add_edge(src, dst)
    return dag


# ----------------------------------------------------------------------
# Expected steps until each agent runs (Markov chain version)
# ----------------------------------------------------------------------
#
# BFS counts hops and ignores probabilities: if Reviewer goes to Planner 70% of
# the time and to Coder 30% of the time, BFS says both are "1 step away".
#
# Instead we treat the workflow as a Markov chain and compute the EXPECTED number
# of steps until each agent runs next (its "hitting time"). For a target agent T:
#
#     h(T) = 0
#     h(u) = 1 + sum over next agents w of  P(u -> w) * h(w)
#
# That is one linear equation per agent, so we solve them all at once with NumPy:
#     (I - Q) h = 1      where Q = transition probabilities between non-target agents
#
# Example from Reviewer: Planner = 1.86 steps, Coder = 2.40, Retriever = 2.86.

def dag_probabilities(dag):
    """Turn an AgentDAG into {src: {dst: probability}}."""
    probs = {}
    for src, edges in dag.edges.items():
        probs[src] = {}
        for dst, p in edges:
            probs[src][dst] = probs[src].get(dst, 0) + p
    return probs


def all_agents(probs, start):
    names = set(probs.keys())
    names.add(start)
    for targets in probs.values():
        names.update(targets.keys())
    return sorted(names)


def agents_that_can_reach(probs, agents, target):
    """Walk the graph backwards from target: which agents can ever get there?"""
    can_reach = {target}
    changed = True
    while changed:
        changed = False
        for src in agents:
            if src in can_reach:
                continue
            for dst in probs.get(src, {}):
                if dst in can_reach:
                    can_reach.add(src)
                    changed = True
                    break
    return can_reach


def expected_steps_to(probs, agents, start, target):
    """Expected number of steps to go from start to target (math.inf if impossible)."""
    if start == target:
        return 0.0

    can_reach = agents_that_can_reach(probs, agents, target)
    if start not in can_reach:
        return math.inf

    # Only agents that can reach the target take part. (Edges into agents that can
    # never reach it are dropped, which keeps the system solvable.)
    others = [a for a in agents if a in can_reach and a != target]
    index = {a: i for i, a in enumerate(others)}

    n = len(others)
    A = np.eye(n)
    b = np.ones(n)
    for a in others:
        for dst, p in probs.get(a, {}).items():
            if dst in index:
                A[index[a], index[dst]] -= p

    h = np.linalg.solve(A, b)
    return float(h[index[start]])


def expected_steps(probs, start):
    """{agent: expected steps from start until that agent runs}. start itself is 0."""
    agents = all_agents(probs, start)
    return {target: expected_steps_to(probs, agents, start, target) for target in agents}
