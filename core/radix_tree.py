"""
RadixTree: the prefix cache (the RadixAttention idea from SGLang).

Many prompts start with the same tokens (the system prompt, an agent's tool
description, ...). Their KV cache is identical, so we store it once and share it.

The tree stores prompts as paths. Each node holds a run of tokens (an "edge")
and the physical block IDs that hold the KV cache for those tokens:

    root
     └── [system prompt]                      blocks 0,1,2
          ├── [planner prefix]                blocks 3,4
          │     └── [planner task tail]       blocks 5,6
          └── [coder prefix]                  blocks 7,8

Only FULL blocks of 16 tokens are cached. A block cannot be half shared, so every
node's token list has a length that is a multiple of 16, and two siblings always
differ somewhere in their first block. That is why a child is looked up by its
first 16 tokens.

When a new prompt matches only the first part of an edge, we split that node in
two. Splitting only moves list slices around. No KV block is copied.
"""

from core.config import BLOCK_SIZE


class Node:
    def __init__(self, tokens, blocks, parent):
        self.tokens = tokens          # list of token IDs, length is a multiple of 16
        self.blocks = blocks          # physical block IDs, len(tokens) // 16 of them
        self.hashes = []              # one chained hash per block (see block_hashes)
        self.parent = parent
        self.children = {}            # first 16 tokens (as a tuple) -> child Node
        self.ref_count = 0            # how many running requests are using this node
        self.last_access = 0          # time of the most recent request that touched it
        self.agents = set()           # which agents have used this node
        self.on_gpu = True            # False = moved to CPU memory (only in the tiered engine)
        self.prefetched = False       # True if the tiered engine prefetched it and nobody used it yet


def first_block_key(tokens):
    return tuple(tokens[:BLOCK_SIZE])


def block_hashes(tokens, prev=None):
    """
    One hash per full block. Each hash also covers every block before it
    (the same trick vLLM uses for prefix caching), so two prompts get the same
    hash for block k only if their first k+1 blocks are identical.

    prev is the hash of the block just before these tokens (None at the start).
    """
    hashes = []
    for start in range(0, len(tokens) // BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE):
        prev = hash((prev, tuple(tokens[start:start + BLOCK_SIZE])))
        hashes.append(prev)
    return hashes


class RadixTree:
    def __init__(self):
        self.root = Node([], [], None)

    # ------------------------------------------------------------------
    # Finding a prefix
    # ------------------------------------------------------------------
    def match_prefix(self, tokens):
        """
        Walk down the tree as far as the prompt matches.

        Returns (matched_blocks, last_node, matched_tokens).
        If the match ends in the middle of an edge, that edge is split first,
        so last_node is always exactly the end of the match.
        """
        tokens = list(tokens)
        usable = len(tokens) // BLOCK_SIZE * BLOCK_SIZE   # ignore the partial last block

        node = self.root
        pos = 0
        matched_blocks = []

        while pos < usable:
            child = node.children.get(tuple(tokens[pos:pos + BLOCK_SIZE]))
            if child is None:
                break

            shared = self._count_shared_blocks(child, tokens, pos, usable)
            if shared < len(child.blocks):
                child = self._split(child, shared)

            matched_blocks.extend(child.blocks)
            pos += len(child.tokens)
            node = child

        return matched_blocks, node, pos

    def _count_shared_blocks(self, child, tokens, pos, usable):
        """How many whole blocks at the start of this edge match the prompt."""
        count = 0
        for i in range(len(child.blocks)):
            start = pos + i * BLOCK_SIZE
            if start + BLOCK_SIZE > usable:
                break
            edge_part = child.tokens[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            if tokens[start:start + BLOCK_SIZE] != edge_part:
                break
            count += 1
        return count

    def _split(self, child, num_blocks):
        """
        Cut a node after its first num_blocks blocks.

        Before:  parent -> child[A B C D]
        After:   parent -> top[A B] -> child[C D]

        The block IDs are not copied, we only re-slice the lists.
        """
        cut = num_blocks * BLOCK_SIZE
        parent = child.parent

        top = Node(child.tokens[:cut], child.blocks[:num_blocks], parent)
        top.hashes = child.hashes[:num_blocks]
        # Every running request that was locking child was also using these
        # first blocks, so top starts with the same ref_count.
        top.ref_count = child.ref_count
        top.last_access = child.last_access
        top.agents = set(child.agents)
        top.on_gpu = child.on_gpu
        top.prefetched = child.prefetched

        child.tokens = child.tokens[cut:]
        child.blocks = child.blocks[num_blocks:]
        child.hashes = child.hashes[num_blocks:]
        child.parent = top

        # top has the same first block as the old child, so it takes the same slot
        parent.children[first_block_key(top.tokens)] = top
        top.children[first_block_key(child.tokens)] = child
        return top

    # ------------------------------------------------------------------
    # Adding to the tree
    # ------------------------------------------------------------------
    def insert_after(self, node, tokens, blocks, agent, now):
        """
        Add the part of a prompt that was NOT in the cache as a new child of node.
        Call match_prefix first to find the node.
        """
        if len(tokens) == 0:
            return node
        if len(tokens) != len(blocks) * BLOCK_SIZE:
            raise ValueError("tokens must be exactly 16 tokens per block")
        if first_block_key(tokens) in node.children:
            raise ValueError("this prefix is already in the tree, use match_prefix first")

        new_node = Node(list(tokens), list(blocks), node)
        # continue the hash chain from the end of the parent node
        prev = node.hashes[-1] if node.hashes else None
        new_node.hashes = block_hashes(tokens, prev)
        new_node.last_access = now
        # which agents own this node is decided in touch(), which knows where
        # the agent's fixed prompt ends and the one-time tail begins
        node.children[first_block_key(tokens)] = new_node
        return new_node

    def depth(self, node):
        """Number of tokens from the root to the end of this node."""
        total = 0
        while node is not self.root:
            total += len(node.tokens)
            node = node.parent
        return total

    def touch(self, node, now, agent, static_tokens=None):
        """
        Record that a request used node and all of its ancestors.

        static_tokens = length of the agent's fixed prompt (system + agent prefix).
        Only nodes inside that fixed part are tagged with the agent. Nodes after it
        hold the request's one-time tail, so no agent "owns" them.
        If static_tokens is None, the whole prompt counts as fixed.
        """
        end = self.depth(node)
        while node is not self.root:
            node.last_access = now
            if static_tokens is None or end <= static_tokens:
                node.agents.add(agent)
            end -= len(node.tokens)
            node = node.parent

    # ------------------------------------------------------------------
    # Locking (reference counting)
    # ------------------------------------------------------------------
    def lock(self, node):
        """A request is using this prompt. Pin every node from here up to the root."""
        while node is not None:
            node.ref_count += 1
            node = node.parent

    def unlock(self, node):
        """The request finished. Release the pins."""
        while node is not None:
            if node.ref_count <= 0:
                raise ValueError("unlock called more times than lock")
            node.ref_count -= 1
            node = node.parent

    # ------------------------------------------------------------------
    # Eviction support
    # ------------------------------------------------------------------
    def all_nodes(self):
        """Every node except the root."""
        result = []
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(node.children.values())
        return result

    def evictable_leaves(self):
        """
        Nodes we are allowed to evict: no children and nobody using them.
        We only evict leaves because a child's KV cache depends on its parent's
        (the child is useless without the parent), so parents go last.
        """
        return [n for n in self.all_nodes() if not n.children and n.ref_count == 0]

    def remove_leaf(self, node):
        """Remove an evictable leaf from the tree and return its block IDs to free."""
        if node.children:
            raise ValueError("cannot remove a node that still has children")
        if node.ref_count > 0:
            raise ValueError("cannot remove a node that is in use")

        del node.parent.children[first_block_key(node.tokens)]
        blocks = node.blocks
        node.parent = None
        # If the parent has no other children it is now a leaf too,
        # so it becomes a candidate for the next eviction.
        return blocks

    def count_blocks(self):
        """Total blocks held by the tree (used by tests to check nothing leaks)."""
        return sum(len(n.blocks) for n in self.all_nodes())
