"""
BlockPool: the physical KV-cache memory, split into fixed-size blocks.

This is the PagedAttention (vLLM) idea. Instead of giving each request one big
contiguous chunk of memory, GPU memory is cut into equal 16-token blocks and each
block gets an integer ID. A request just holds a list of block IDs. Because every
block is the same size there is no external fragmentation: any free block can be
used by any request.

Here a block is only an integer ID plus bookkeeping. If simulate_memory=True we
also allocate a real NumPy FP16 array with the exact Llama-3-8B shape, so we can
check that one block really is 2 MiB.
"""

import numpy as np

from core.config import BLOCK_SIZE, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, BYTES_PER_BLOCK


class OutOfBlocksError(Exception):
    """Raised when we ask for more blocks than are free."""
    pass


class BlockPool:
    def __init__(self, num_blocks, simulate_memory=False):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.num_blocks = num_blocks

        # Free list used as a stack. Reversed so that block 0 is handed out first,
        # which makes tests and debugging easier to follow.
        self.free_blocks = list(range(num_blocks - 1, -1, -1))

        # is_used[i] is True while block i belongs to someone.
        # We use it to catch bugs like freeing the same block twice.
        self.is_used = np.zeros(num_blocks, dtype=bool)

        # Optional real memory.
        # Shape: (block, K/V, layer, token in block, kv head, head dim)
        self.kv_buffer = None
        if simulate_memory:
            self.kv_buffer = np.zeros(
                (num_blocks, 2, NUM_LAYERS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
                dtype=np.float16,
            )

    def allocate(self, n):
        """Take n free blocks and return their IDs."""
        if n < 0:
            raise ValueError("cannot allocate a negative number of blocks")
        if n > len(self.free_blocks):
            raise OutOfBlocksError(
                f"asked for {n} blocks but only {len(self.free_blocks)} are free"
            )

        block_ids = []
        for _ in range(n):
            block_id = self.free_blocks.pop()
            self.is_used[block_id] = True
            block_ids.append(block_id)
        return block_ids

    def free(self, block_ids):
        """Give blocks back to the pool."""
        for block_id in block_ids:
            if not self.is_used[block_id]:
                raise ValueError(f"block {block_id} is already free (double free)")
            self.is_used[block_id] = False
            self.free_blocks.append(block_id)

    def num_free(self):
        return len(self.free_blocks)

    def num_used(self):
        return self.num_blocks - len(self.free_blocks)

    def used_bytes(self):
        """Memory in use, in bytes, using the Llama-3-8B block size."""
        return self.num_used() * BYTES_PER_BLOCK
