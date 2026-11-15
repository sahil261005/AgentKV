"""
Eviction policies: which cached leaf to throw away when memory is full.

Every policy has one method that matters: score(node).
The cache engine evicts the leaf with the HIGHEST score:

    victim = max(leaves, key=policy.score)

    LRU      score = -last_access                     -> oldest leaf first
    AgentKV  score = (steps until owner runs, -last_access)
                                                      -> agent that runs furthest away first
    AgentKV-Markov  score = (EXPECTED steps until owner runs, -last_access)
                                                      -> same idea, but uses branch probabilities
    Belady   score = (steps until this node is next used, -last_access)
                                                      -> cheats by looking at the future trace

Python compares tuples element by element, so the second value only breaks ties.

The engine calls two hooks on every policy:
    observe_step(agents)                     once per workflow step (the learning policy uses it)
    start_step(position, current_agents)     before each request

can_predict = True means the policy knows which agents run next, so the tiered
engine can use it to prefetch.
"""

import math
from bisect import bisect_right

from core.agent_dag import dag_probabilities, expected_steps
from core.radix_tree import block_hashes


def choose_victim(policy, leaves):
    return max(leaves, key=policy.score)


class LRUPolicy:
    """What SGLang does by default: evict the least recently used leaf."""

    name = "LRU"
    can_predict = False

    def observe_step(self, agents):
        pass

    def start_step(self, position, current_agents):
        pass  # LRU only needs last_access, which the tree already tracks

    def score(self, node):
        return -node.last_access


class TailFirstLRUPolicy:
    """
    Ablation baseline: LRU that also knows which nodes are one-time request tails.

    It evicts tails first (oldest first), then agent prefixes (oldest first).
    It gets the same "fixed prompt length" hint as AgentKV but does NOT use the
    workflow graph. Comparing it with AgentKV shows how much of the gain comes
    from the graph itself and how much just from "throw away tails first".
    """

    name = "LRU-TailFirst"
    can_predict = False

    def observe_step(self, agents):
        pass

    def start_step(self, position, current_agents):
        pass

    def score(self, node):
        is_tail = 1 if not node.agents else 0
        return (is_tail, -node.last_access)


class AgentKVPolicy:
    """
    Our policy (from the KVFlow paper): use the workflow graph.

    For every agent we know how many hops away it is from the agent(s) running now.
    A node can be shared by several agents (e.g. the system prompt), so we take the
    smallest distance: if ANY agent that uses it runs soon, keep it.

    Nodes that belong to no agent are one-time request tails. Their distance is
    infinity, so they are always evicted before any agent's fixed prompt.
    """

    name = "AgentKV"
    can_predict = True

    def __init__(self, dag):
        self.dag = dag
        self.distances = {}

    def observe_step(self, agents):
        pass

    def start_step(self, position, current_agents):
        # Usually one agent is running. With parallel agents (fan-out) several are,
        # so for each agent we keep the smallest distance from any of them.
        self.distances = {}
        for running in current_agents:
            for agent, d in self.dag.steps_to_execution(running).items():
                if d < self.distances.get(agent, math.inf):
                    self.distances[agent] = d

    def distance_of(self, node):
        best = math.inf
        for agent in node.agents:
            best = min(best, self.distances.get(agent, math.inf))
        return best

    def score(self, node):
        return (self.distance_of(node), -node.last_access)


class MarkovAgentKVPolicy(AgentKVPolicy):
    """
    AgentKV with expected steps instead of BFS hops, learned from the requests.

    1. Probabilities. BFS treats a 30% branch and a 70% branch the same ("1 hop").
       Here the workflow is a Markov chain and the distance is the EXPECTED number of
       steps until each agent runs (see expected_steps in agent_dag.py).

    2. Learned, not registered. It counts which step follows which while running,
       so nobody has to hand it a graph. With no data yet every distance is infinity
       and it behaves exactly like LRU. A registered graph can be given as a
       starting point; learning continues on top of it.

    3. States are STEPS, not single agents. A step is the group of agents that run
       together, e.g. ("Worker1", "Worker2", "Worker3") in a fan-out. A chain over
       single agents would assume one agent per step and think each parallel worker
       is 3x further away than it really is. An agent's distance is the expected
       steps until any step that contains it.
    """

    name = "AgentKV-Markov"

    def __init__(self, dag=None):
        self.distances = {}
        self.cache = {}              # start step -> expected steps to every step
        self.last_step = None
        self.counts = {}             # step -> {next step: count}
        if dag is not None:
            # starting point: every edge of the registered graph as a one-agent step
            for src, targets in dag_probabilities(dag).items():
                self.counts[(src,)] = {(dst,): p for dst, p in targets.items()}

    def observe_step(self, agents):
        step = tuple(sorted(agents))
        if self.last_step is not None:
            self.counts.setdefault(self.last_step, {})
            self.counts[self.last_step][step] = self.counts[self.last_step].get(step, 0) + 1
            self.cache = {}          # the chain changed, old answers are stale
        self.last_step = step

    def probabilities(self):
        probs = {}
        for src, targets in self.counts.items():
            total = sum(targets.values())
            probs[src] = {dst: c / total for dst, c in targets.items()}
        return probs

    def start_step(self, position, current_agents):
        start = tuple(sorted(current_agents))
        if start not in self.cache:
            self.cache[start] = expected_steps(self.probabilities(), start)

        # an agent's distance = expected steps until the first step that contains it
        self.distances = {}
        for step, d in self.cache[start].items():
            for agent in step:
                if d < self.distances.get(agent, math.inf):
                    self.distances[agent] = d


class BeladyPolicy:
    """
    Offline oracle (Belady's MIN): it is given the whole future trace and evicts
    the node whose next use is furthest away (or never).

    A real server cannot do this because it does not know the future. We only use
    it as a reference for how good an eviction policy could possibly be.
    (MIN is provably optimal only when every item has the same size. Our nodes
    have different sizes, so we call it an oracle reference, not "optimal".)
    """

    name = "Belady"
    can_predict = False

    def observe_step(self, agents):
        pass

    def __init__(self, prompts):
        # For every block hash: the list of trace positions whose prompt contains it.
        self.uses = {}
        for position, tokens in enumerate(prompts):
            for h in block_hashes(tokens):
                if h not in self.uses:
                    self.uses[h] = []
                self.uses[h].append(position)
        self.position = 0

    def start_step(self, position, current_agents):
        self.position = position

    def node_hash(self, node):
        """The chained hash of the node's first block (stored on the node by the tree)."""
        return node.hashes[0]

    def steps_until_next_use(self, node):
        positions = self.uses.get(self.node_hash(node), [])
        i = bisect_right(positions, self.position)   # first use strictly after now
        if i == len(positions):
            return math.inf
        return positions[i] - self.position

    def score(self, node):
        return (self.steps_until_next_use(node), -node.last_access)
