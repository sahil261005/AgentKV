import pytest

from core.block_pool import BlockPool, OutOfBlocksError
from core.config import BYTES_PER_TOKEN, BYTES_PER_BLOCK, KIB, MIB, blocks_to_mib


def test_llama3_8b_sizes():
    # 2 * 32 * 8 * 128 * 2 bytes
    assert BYTES_PER_TOKEN == 131072
    assert BYTES_PER_TOKEN == 128 * KIB
    assert BYTES_PER_BLOCK == 2 * MIB
    assert blocks_to_mib(300) == 600


def test_allocate_and_free():
    pool = BlockPool(10)
    blocks = pool.allocate(4)

    assert blocks == [0, 1, 2, 3]
    assert pool.num_used() == 4
    assert pool.num_free() == 6

    pool.free(blocks)
    assert pool.num_used() == 0
    assert pool.num_free() == 10


def test_allocated_blocks_are_unique():
    pool = BlockPool(8)
    a = pool.allocate(3)
    b = pool.allocate(5)
    assert len(set(a + b)) == 8


def test_out_of_blocks():
    pool = BlockPool(5)
    pool.allocate(3)
    with pytest.raises(OutOfBlocksError):
        pool.allocate(3)
    # a failed allocation must not take anything
    assert pool.num_free() == 2


def test_double_free_is_caught():
    pool = BlockPool(4)
    blocks = pool.allocate(2)
    pool.free(blocks)
    with pytest.raises(ValueError):
        pool.free(blocks)


def test_used_bytes():
    pool = BlockPool(10)
    pool.allocate(3)
    assert pool.used_bytes() == 3 * 2 * MIB


def test_real_numpy_buffer_is_2_mib_per_block():
    # allocate a real FP16 buffer with the Llama-3-8B shape and check its size
    pool = BlockPool(2, simulate_memory=True)
    assert pool.kv_buffer[0].nbytes == 2 * MIB
    assert pool.kv_buffer.nbytes == 2 * 2 * MIB
