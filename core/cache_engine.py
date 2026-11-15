"""
CacheEngine: puts the pieces together.

    BlockPool       - physical memory (block IDs)
    RadixTree       - which token prefixes are cached and in which blocks
    eviction policy - which leaf to throw away when memory is full

For every incoming request:
    1. find the longest cached prefix (those blocks are cache HITS)
    2. lock that path so we can't evict what we are about to use
    3. evict leaves (the policy picks which) until the new blocks fit
    4. allocate blocks for the uncached part and insert it into the tree
    5. when the request finishes, unlock it (it stays cached for later)

Everything is counted, so we can compare policies with exact numbers.
"""

from core.block_pool import BlockPool
from core.config import BLOCK_SIZE, tokens_to_gib, tokens_to_tflops
from core.eviction_policy import (
    LRUPolicy, TailFirstLRUPolicy, AgentKVPolicy, MarkovAgentKVPolicy, BeladyPolicy, choose_victim,
)
from core.radix_tree import RadixTree, block_hashes


class CacheEngine:
    def __init__(self, num_blocks, policy, simulate_memory=False):
        self.pool = BlockPool(num_blocks, simulate_memory)
        self.tree = RadixTree()
        self.policy = policy

        self.clock = 0          # goes up by 1 for every request (our notion of "time")
        self.position = 0       # index of the request in the trace (Belady needs this)
        self.active = []        # leaves of requests that are still running

        # Hash of every block we have ever cached. If we miss on a block whose hash is
        # in here, we are recomputing work we already did once (it got evicted).
        self.seen_hashes = set()

        # counters
        self.requests = 0
        self.prompt_tokens = 0
        self.total_blocks = 0
        self.hit_blocks = 0
        self.computed_tokens = 0
        self.recomputed_tokens = 0
        self.evictions = 0
        self.evicted_blocks = 0
        self.uncached_blocks = 0   # blocks we could not cache because memory was full of locked data

    # ------------------------------------------------------------------
    # Running requests
    # ------------------------------------------------------------------
    def begin_request(self, agent, tokens, running_agents=None, static_tokens=None):
        """
        Start a request. Returns (handle, result). Call end_request(handle) when it finishes.

        running_agents = every agent running at the same time (for parallel steps).
        static_tokens  = length of this agent's fixed prompt (system prompt + agent prefix).
                         The workflow framework knows this. Everything after it is a
                         one-time tail. None means "treat the whole prompt as fixed".
        """
        tokens = list(tokens)
        if running_agents is None:
            running_agents = [agent]

        self.clock += 1
        self.policy.start_step(self.position, running_agents)
        self.position += 1

        # 1. longest cached prefix
        _, node, matched_tokens = self.tree.match_prefix(tokens)
        usable = len(tokens) // BLOCK_SIZE * BLOCK_SIZE      # only full blocks get cached
        hit = matched_tokens // BLOCK_SIZE
        need = (usable - matched_tokens) // BLOCK_SIZE

        # 2. lock the matched path while we make room
        self.tree.lock(node)

        # 3. evict until the new blocks fit (or nothing else can be evicted)
        evicted_now = self._make_room(need)

        # 4. cache as much of the new part as fits
        n_alloc = min(need, self.pool.num_free())
        leaf = self._insert_new_part(node, agent, tokens, matched_tokens, n_alloc, usable, static_tokens)

        # 5. move the lock down to the new leaf, update counters
        result = self._finish_begin(node, leaf, agent, tokens, usable, matched_tokens,
                                    need, n_alloc, static_tokens, evicted_now)
        return leaf, result

    def _insert_new_part(self, node, agent, tokens, matched_tokens, n_alloc, usable, static_tokens):
        """Allocate n_alloc blocks and insert them under node. Returns the new leaf."""
        new_blocks = self.pool.allocate(n_alloc)
        end = matched_tokens + n_alloc * BLOCK_SIZE

        # Insert in two pieces so no node mixes the agent's fixed prompt with the tail:
        #   piece 1: the uncached part of the fixed prompt
        #   piece 2: the one-time tail
        if static_tokens is None:
            static_end = usable
        else:
            static_end = static_tokens // BLOCK_SIZE * BLOCK_SIZE
        cut = min(max(static_end, matched_tokens), end)
        k = (cut - matched_tokens) // BLOCK_SIZE

        leaf = node
        if cut > matched_tokens:
            leaf = self.tree.insert_after(leaf, tokens[matched_tokens:cut], new_blocks[:k], agent, self.clock)
        if end > cut:
            leaf = self.tree.insert_after(leaf, tokens[cut:end], new_blocks[k:], agent, self.clock)
        return leaf

    def _finish_begin(self, node, leaf, agent, tokens, usable, reused_tokens, need, n_alloc,
                      static_tokens, evicted_now):
        """Shared end of begin_request: locks, tags, and counters."""
        # move our lock from the matched node down to the new leaf
        self.tree.lock(leaf)
        self.tree.unlock(node)
        self.tree.touch(leaf, self.clock, agent, static_tokens)

        # how much of the missed work was work we had already done before?
        hit = reused_tokens // BLOCK_SIZE
        hashes = block_hashes(tokens[:usable])
        recomputed_blocks = 0
        for h in hashes[hit:]:
            if h in self.seen_hashes:
                recomputed_blocks += 1
        self.seen_hashes.update(hashes[hit:hit + n_alloc])

        # counters
        computed = len(tokens) - reused_tokens
        self.requests += 1
        self.prompt_tokens += len(tokens)
        self.total_blocks += usable // BLOCK_SIZE
        self.hit_blocks += hit
        self.computed_tokens += computed
        self.recomputed_tokens += recomputed_blocks * BLOCK_SIZE
        self.uncached_blocks += need - n_alloc

        self.active.append(leaf)
        return {
            "agent": agent,
            "prompt_tokens": len(tokens),
            "hit_tokens": reused_tokens,
            "computed_tokens": computed,
            "recomputed_tokens": recomputed_blocks * BLOCK_SIZE,
            "evicted_blocks": evicted_now,
        }

    def end_request(self, handle):
        self.tree.unlock(handle)
        self.active.remove(handle)

    def run_request(self, agent, tokens, static_tokens=None):
        self.policy.observe_step([agent])
        handle, result = self.begin_request(agent, tokens, None, static_tokens)
        self.end_request(handle)
        return result

    def run_step(self, step, static_tokens=None):
        """
        A step is a list of (agent, tokens) that run at the same time.
        All of them are started (and locked) before any of them finishes.
        """
        running = [agent for agent, _ in step]
        self.policy.observe_step(running)
        handles = []
        results = []
        for agent, tokens in step:
            handle, result = self.begin_request(agent, tokens, running, static_tokens)
            handles.append(handle)
            results.append(result)
        for handle in handles:
            self.end_request(handle)
        return results

    def _make_room(self, need):
        """Evict leaves chosen by the policy until `need` blocks are free. Returns blocks evicted."""
        freed = 0
        while self.pool.num_free() < need:
            leaves = self.tree.evictable_leaves()
            if not leaves:
                break       # everything left is in use, we can't free more
            victim = choose_victim(self.policy, leaves)
            blocks = self.tree.remove_leaf(victim)
            self.pool.free(blocks)
            self.evictions += 1
            self.evicted_blocks += len(blocks)
            freed += len(blocks)
        return freed

    # ------------------------------------------------------------------
    # Checking and reporting
    # ------------------------------------------------------------------
    def check_invariants(self):
        """Sanity checks used by tests and the benchmark. Raises AssertionError on a bug."""
        nodes = self.tree.all_nodes()

        all_blocks = []
        for node in nodes:
            assert len(node.blocks) * BLOCK_SIZE == len(node.tokens), "node tokens/blocks mismatch"
            assert len(node.hashes) == len(node.blocks), "node hashes/blocks mismatch"
            all_blocks.extend(node.blocks)

        assert len(all_blocks) == len(set(all_blocks)), "a block is used by two nodes"
        assert len(all_blocks) == self.pool.num_used(), "leaked blocks: pool and tree disagree"

        if not self.active:
            for node in nodes:
                assert node.ref_count == 0, "ref_count not back to 0 after all requests finished"
            assert self.tree.root.ref_count == 0

    def summary(self):
        hit_rate = 100.0 * self.hit_blocks / self.total_blocks if self.total_blocks else 0.0
        return {
            "policy": self.policy.name,
            "capacity_blocks": self.pool.num_blocks,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "total_blocks": self.total_blocks,
            "hit_blocks": self.hit_blocks,
            "hit_rate": hit_rate,
            "computed_tokens": self.computed_tokens,
            "recomputed_tokens": self.recomputed_tokens,
            "recomputed_kv_gib": tokens_to_gib(self.recomputed_tokens),
            "recomputed_tflops": tokens_to_tflops(self.recomputed_tokens),
            "evictions": self.evictions,
            "evicted_blocks": self.evicted_blocks,
            "uncached_blocks": self.uncached_blocks,
        }


POLICY_NAMES = ["LRU", "LRU-TailFirst", "AgentKV", "AgentKV-Markov", "Belady"]


def make_policy(name, dag=None, prompts=None):
    if name == "LRU":
        return LRUPolicy()
    if name == "LRU-TailFirst":
        return TailFirstLRUPolicy()
    if name == "AgentKV":
        return AgentKVPolicy(dag)
    if name == "AgentKV-Markov":
        return MarkovAgentKVPolicy()         # learns the workflow from the requests, no graph given
    if name == "Belady":
        return BeladyPolicy(prompts)
    raise ValueError(f"unknown policy {name}")


def run_workload(steps, dag, policy_name, num_blocks, static_tokens=None):
    """Replay a whole trace through a fresh engine and return the summary."""
    prompts = []
    for step in steps:
        for _, tokens in step:
            prompts.append(tokens)

    engine = CacheEngine(num_blocks, make_policy(policy_name, dag, prompts))
    for step in steps:
        engine.run_step(step, static_tokens)
    engine.check_invariants()
    return engine.summary()
