"""
TieredCacheEngine: GPU memory + CPU memory, with workflow-aware prefetching.

In the basic engine an evicted block is gone. If it is needed again it must be
recomputed with a full prefill, which is expensive.

Real servers have far more CPU RAM than GPU memory, so here eviction MOVES a node
to CPU RAM instead of deleting it ("offloading"). If a request needs it later,
the node is copied back over PCIe, which is about 20x cheaper than recomputing it.
But that copy happens while the request waits.

Prefetching removes the wait. After each step, a policy that knows the workflow
(AgentKV) asks: "which agent runs next, and is its prefix sitting on the CPU?" If
so, it copies that prefix to the GPU before the agent arrives, making room by
offloading agents that run later. The KVFlow paper uses the same idea, with the
copy overlapping the current step's compute. Plain LRU can't prefetch because it
doesn't know what comes next.

Rules that keep the tree consistent:
  - GPU nodes always form the top of the tree: a node on the GPU always has its
    parent on the GPU too (a child's KV cache is useless without its parent's).
  - So GPU eviction picks "GPU leaves": GPU nodes with no children on the GPU.
  - If CPU RAM is full too, the oldest CPU leaves are deleted for real.
"""

import math

from core.block_pool import BlockPool
from core.cache_engine import CacheEngine, make_policy
from core.config import BLOCK_SIZE, recompute_ms, transfer_ms
from core.eviction_policy import choose_victim
from core.radix_tree import first_block_key


class TieredCacheEngine(CacheEngine):
    def __init__(self, gpu_blocks, cpu_blocks, policy, prefetch=True):
        super().__init__(gpu_blocks, policy)
        self.cpu_pool = BlockPool(cpu_blocks)
        # only policies that know the workflow can prefetch
        self.prefetch_enabled = prefetch and policy.can_predict

        self.gpu_hit_blocks = 0         # already on the GPU when the request arrived
        self.cpu_hit_blocks = 0         # on the CPU: copied back while the request waited
        self.prefetched_blocks = 0      # copied to the GPU ahead of time
        self.prefetch_used_blocks = 0   # prefetched blocks that a request then used
        self.offloaded_blocks = 0       # GPU -> CPU moves
        self.dropped_blocks = 0         # deleted because the CPU was full as well

    # ------------------------------------------------------------------
    # Which nodes can move where
    # ------------------------------------------------------------------
    def gpu_leaves(self):
        """GPU nodes nobody is using that have no children on the GPU."""
        result = []
        for node in self.tree.all_nodes():
            if not node.on_gpu or node.ref_count > 0:
                continue
            has_gpu_child = False
            for child in node.children.values():
                if child.on_gpu:
                    has_gpu_child = True
            if not has_gpu_child:
                result.append(node)
        return result

    def cpu_leaves(self):
        """CPU nodes nobody is using with no children at all."""
        return [n for n in self.tree.all_nodes()
                if not n.on_gpu and n.ref_count == 0 and not n.children]

    def path_to(self, node):
        """Nodes from the top of the tree down to node (root not included)."""
        path = []
        while node is not self.tree.root:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Moving data between GPU and CPU
    # ------------------------------------------------------------------
    def _offload(self, node):
        """Move a GPU leaf to CPU RAM. If the CPU is full, delete old CPU data first."""
        if not node.agents:
            # A one-time request tail: it will never be reused, so copying it to the
            # CPU would only waste PCIe bandwidth and CPU RAM. Delete it instead.
            # (This rule is the same for every policy.)
            self._drop_subtree(node)
            return

        size = len(node.blocks)
        while self.cpu_pool.num_free() < size:
            leaves = self.cpu_leaves()
            if not leaves:
                break
            oldest = min(leaves, key=lambda n: n.last_access)
            self.dropped_blocks += len(oldest.blocks)
            self.cpu_pool.free(self.tree.remove_leaf(oldest))

        if self.cpu_pool.num_free() >= size:
            cpu_blocks = self.cpu_pool.allocate(size)
            self.pool.free(node.blocks)
            node.blocks = cpu_blocks
            node.on_gpu = False
            node.prefetched = False
            self.offloaded_blocks += size
        else:
            self._drop_subtree(node)    # nowhere to put it: delete it (and its CPU children)

    def _drop_subtree(self, node):
        stack = [node]
        while stack:
            current = stack.pop()
            stack.extend(current.children.values())
            if current.on_gpu:
                self.pool.free(current.blocks)
            else:
                self.cpu_pool.free(current.blocks)
            self.dropped_blocks += len(current.blocks)
        del node.parent.children[first_block_key(node.tokens)]
        node.parent = None

    def _swap_in(self, node):
        """Copy a CPU node back to the GPU (caller must make sure there is room)."""
        gpu_blocks = self.pool.allocate(len(node.blocks))
        self.cpu_pool.free(node.blocks)
        node.blocks = gpu_blocks
        node.on_gpu = True

    def _make_room(self, need):
        """Same as the basic engine, but victims go to the CPU instead of being deleted."""
        freed = 0
        while self.pool.num_free() < need:
            leaves = self.gpu_leaves()
            if not leaves:
                break
            victim = choose_victim(self.policy, leaves)
            size = len(victim.blocks)
            self._offload(victim)
            self.evictions += 1
            self.evicted_blocks += size
            freed += size
        return freed

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------
    def begin_request(self, agent, tokens, running_agents=None, static_tokens=None):
        tokens = list(tokens)
        if running_agents is None:
            running_agents = [agent]

        self.clock += 1
        self.policy.start_step(self.position, running_agents)
        self.position += 1

        _, node, matched_tokens = self.tree.match_prefix(tokens)
        usable = len(tokens) // BLOCK_SIZE * BLOCK_SIZE
        self.tree.lock(node)

        # split the matched path into the part on the GPU and the part on the CPU
        path = self.path_to(node)
        gpu_part = [n for n in path if n.on_gpu]
        cpu_part = [n for n in path if not n.on_gpu]

        gpu_tokens = 0
        for n in gpu_part:
            gpu_tokens += len(n.tokens)
            if n.prefetched:                     # prefetch paid off
                self.prefetch_used_blocks += len(n.blocks)
                n.prefetched = False

        swap_need = sum(len(n.blocks) for n in cpu_part)
        new_need = (usable - matched_tokens) // BLOCK_SIZE
        evicted_now = self._make_room(swap_need + new_need)

        # copy the CPU part back to the GPU, top-down (the request waits for this)
        swapped = 0
        for n in cpu_part:
            if self.pool.num_free() < len(n.blocks):
                break
            self._swap_in(n)
            swapped += len(n.blocks)

        self.gpu_hit_blocks += gpu_tokens // BLOCK_SIZE
        self.cpu_hit_blocks += swapped

        if swapped == swap_need:
            # the whole prefix is on the GPU: cache the new part as usual
            reused_tokens = matched_tokens
            n_alloc = min(new_need, self.pool.num_free())
            leaf = self._insert_new_part(node, agent, tokens, matched_tokens, n_alloc, usable, static_tokens)
            need = new_need
        else:
            # not enough room to bring everything back: the rest is recomputed, not cached
            reused_tokens = gpu_tokens + swapped * BLOCK_SIZE
            n_alloc = 0
            leaf = node
            need = 0

        result = self._finish_begin(node, leaf, agent, tokens, usable, reused_tokens,
                                    need, n_alloc, static_tokens, evicted_now)
        return leaf, result

    def run_step(self, step, static_tokens=None):
        results = super().run_step(step, static_tokens)
        if self.prefetch_enabled:
            self.prefetch([agent for agent, _ in step])
        return results

    # ------------------------------------------------------------------
    # Prefetching
    # ------------------------------------------------------------------
    def prefetch(self, finished_agents):
        """
        Copy the prefixes of the agents that run next from CPU to GPU.
        We only ever make room by offloading things that run LATER than the agent
        we are prefetching for, so prefetching never hurts a closer agent.
        """
        self.policy.start_step(self.position, finished_agents)   # distances from "now"

        for _ in range(len(self.tree.all_nodes())):               # safety limit
            # CPU nodes whose parent is on the GPU (we can only add to the top part)
            candidates = []
            for node in self.tree.all_nodes():
                parent_on_gpu = node.parent is self.tree.root or node.parent.on_gpu
                if not node.on_gpu and parent_on_gpu and node.agents:
                    d = self.policy.distance_of(node)
                    if 0 < d < math.inf:
                        candidates.append(node)
            if not candidates:
                return

            target = min(candidates, key=lambda n: (self.policy.distance_of(n), -n.last_access))
            if not self._room_for_prefetch(len(target.blocks), self.policy.distance_of(target)):
                return
            self._swap_in(target)
            target.prefetched = True
            self.prefetched_blocks += len(target.blocks)

    def _room_for_prefetch(self, need, distance):
        while self.pool.num_free() < need:
            victims = [n for n in self.gpu_leaves() if self.policy.distance_of(n) > distance]
            if not victims:
                return False
            self._offload(choose_victim(self.policy, victims))
        return True

    # ------------------------------------------------------------------
    # Checking and reporting
    # ------------------------------------------------------------------
    def check_invariants(self):
        nodes = self.tree.all_nodes()
        gpu_blocks = []
        cpu_blocks = []
        for node in nodes:
            assert len(node.blocks) * BLOCK_SIZE == len(node.tokens), "node tokens/blocks mismatch"
            if node.on_gpu:
                gpu_blocks.extend(node.blocks)
                assert node.parent is self.tree.root or node.parent.on_gpu, "GPU node under a CPU node"
            else:
                cpu_blocks.extend(node.blocks)

        assert len(gpu_blocks) == len(set(gpu_blocks)), "a GPU block is used twice"
        assert len(cpu_blocks) == len(set(cpu_blocks)), "a CPU block is used twice"
        assert len(gpu_blocks) == self.pool.num_used(), "leaked GPU blocks"
        assert len(cpu_blocks) == self.cpu_pool.num_used(), "leaked CPU blocks"
        if not self.active:
            for node in nodes:
                assert node.ref_count == 0, "ref_count not back to 0"

    def summary(self):
        s = super().summary()
        total = self.total_blocks if self.total_blocks else 1
        s["cpu_capacity_blocks"] = self.cpu_pool.num_blocks
        s["gpu_hit_rate"] = 100.0 * self.gpu_hit_blocks / total
        s["cpu_hit_blocks"] = self.cpu_hit_blocks
        s["prefetched_blocks"] = self.prefetched_blocks
        s["prefetch_used_blocks"] = self.prefetch_used_blocks
        s["prefetch_accuracy"] = (100.0 * self.prefetch_used_blocks / self.prefetched_blocks
                                  if self.prefetched_blocks else 0.0)
        s["offloaded_blocks"] = self.offloaded_blocks
        s["dropped_blocks"] = self.dropped_blocks
        # time the requests spent waiting on work they shouldn't have needed:
        # recomputing evicted blocks + copying CPU blocks back on demand
        stall = recompute_ms(self.recomputed_tokens) + transfer_ms(self.cpu_hit_blocks)
        s["stall_ms_per_request"] = stall / self.requests if self.requests else 0.0
        return s


def run_workload_tiered(steps, dag, policy_name, gpu_blocks, cpu_blocks, static_tokens=None, prefetch=True):
    """Replay a trace through a fresh tiered engine and return the summary."""
    prompts = []
    for step in steps:
        for _, tokens in step:
            prompts.append(tokens)

    engine = TieredCacheEngine(gpu_blocks, cpu_blocks, make_policy(policy_name, dag, prompts), prefetch)
    for step in steps:
        engine.run_step(step, static_tokens)
    engine.check_invariants()
    return engine.summary()


def single_tier_stall_ms(summary):
    """Same stall metric for the basic GPU-only engine (only recompute, no copies)."""
    if not summary["requests"]:
        return 0.0
    return recompute_ms(summary["recomputed_tokens"]) / summary["requests"]
