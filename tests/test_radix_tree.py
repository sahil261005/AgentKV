import pytest

from core.block_pool import BlockPool
from core.radix_tree import RadixTree
from core.config import BLOCK_SIZE


def make_tokens(block_labels):
    """[1, 2, 3] -> three blocks of 16 tokens, filled with 1s, 2s and 3s."""
    tokens = []
    for label in block_labels:
        tokens.extend([label] * BLOCK_SIZE)
    return tokens


class Setup:
    """Small helper: a tree + a pool, and one method that behaves like a cache miss."""

    def __init__(self):
        self.tree = RadixTree()
        self.pool = BlockPool(100)
        self.time = 0

    def add(self, block_labels, agent="A"):
        self.time += 1
        tokens = make_tokens(block_labels)
        matched, node, pos = self.tree.match_prefix(tokens)
        rest = tokens[pos:]
        new_blocks = self.pool.allocate(len(rest) // BLOCK_SIZE)
        leaf = self.tree.insert_after(node, rest, new_blocks, agent, self.time)
        self.tree.touch(leaf, self.time, agent)
        return leaf


def test_empty_tree_matches_nothing():
    tree = RadixTree()
    blocks, node, matched = tree.match_prefix(make_tokens([1, 2]))
    assert blocks == []
    assert node is tree.root
    assert matched == 0


def test_exact_match():
    s = Setup()
    leaf = s.add([1, 2, 3])

    blocks, node, matched = s.tree.match_prefix(make_tokens([1, 2, 3]))
    assert blocks == leaf.blocks
    assert node is leaf
    assert matched == 3 * BLOCK_SIZE


def test_prompt_longer_than_cached_prefix():
    s = Setup()
    leaf = s.add([1, 2])

    blocks, node, matched = s.tree.match_prefix(make_tokens([1, 2, 3, 4]))
    assert blocks == leaf.blocks
    assert node is leaf
    assert matched == 2 * BLOCK_SIZE


def test_partial_last_block_is_ignored():
    s = Setup()
    s.add([1])
    # 16 tokens of "1" plus 5 extra tokens: only the full block counts
    tokens = make_tokens([1]) + [9] * 5
    blocks, node, matched = s.tree.match_prefix(tokens)
    assert matched == BLOCK_SIZE
    assert len(blocks) == 1


def test_mid_edge_split_does_not_copy_blocks():
    s = Setup()
    original = s.add([1, 2, 3, 4])
    original_blocks = list(original.blocks)

    # shares the first 2 blocks, then diverges
    blocks, node, matched = s.tree.match_prefix(make_tokens([1, 2, 9, 9]))

    assert matched == 2 * BLOCK_SIZE
    assert blocks == original_blocks[:2]          # same block IDs, nothing copied
    assert node.blocks == original_blocks[:2]     # node is the new top half
    # the old node is now the bottom half and hangs under the top half
    assert original.parent is node
    assert original.blocks == original_blocks[2:]
    assert s.tree.count_blocks() == 4             # still only 4 blocks in total


def test_two_prompts_share_one_prefix_node():
    s = Setup()
    s.add([1, 2, 3])
    s.add([1, 2, 4])

    # tree shape: root -> [1,2] -> {[3], [4]}
    top = s.tree.root.children[tuple([1] * BLOCK_SIZE)]
    assert len(top.blocks) == 2
    assert len(top.children) == 2
    assert s.tree.count_blocks() == 4


def test_split_keeps_ref_count():
    s = Setup()
    leaf = s.add([1, 2, 3])
    s.tree.lock(leaf)

    _, top, _ = s.tree.match_prefix(make_tokens([1, 9]))

    # the request that locked the leaf was using these blocks too
    assert top.ref_count == 1
    assert leaf.ref_count == 1


def test_lock_and_unlock_whole_path():
    s = Setup()
    s.add([1, 2, 3])
    leaf = s.add([1, 2, 4])
    top = leaf.parent

    s.tree.lock(leaf)
    assert leaf.ref_count == 1
    assert top.ref_count == 1

    s.tree.unlock(leaf)
    assert leaf.ref_count == 0
    assert top.ref_count == 0


def test_unlock_too_many_times_raises():
    s = Setup()
    leaf = s.add([1])
    with pytest.raises(ValueError):
        s.tree.unlock(leaf)


def test_locked_leaf_cannot_be_evicted():
    s = Setup()
    leaf_a = s.add([1, 2])
    leaf_b = s.add([1, 3])

    s.tree.lock(leaf_a)

    assert s.tree.evictable_leaves() == [leaf_b]
    with pytest.raises(ValueError):
        s.tree.remove_leaf(leaf_a)


def test_internal_nodes_are_not_evictable():
    s = Setup()
    s.add([1, 2, 3])
    s.add([1, 2, 4])
    top = s.tree.root.children[tuple([1] * BLOCK_SIZE)]

    leaves = s.tree.evictable_leaves()
    assert top not in leaves
    assert len(leaves) == 2


def test_cascading_cleanup():
    s = Setup()
    leaf_3 = s.add([1, 2, 3])
    leaf_4 = s.add([1, 2, 4])
    top = leaf_3.parent

    # remove one child: the parent still has another child
    s.tree.remove_leaf(leaf_3)
    assert s.tree.evictable_leaves() == [leaf_4]

    # remove the other: now the parent has no children, so it becomes evictable
    s.tree.remove_leaf(leaf_4)
    assert s.tree.evictable_leaves() == [top]

    # remove the parent: the tree is empty again
    s.tree.remove_leaf(top)
    assert s.tree.count_blocks() == 0
    assert s.tree.root.children == {}


def test_touch_updates_ancestors():
    s = Setup()
    leaf = s.add([1, 2], agent="Planner")
    s.tree.touch(leaf, now=50, agent="Coder")

    assert leaf.last_access == 50
    assert leaf.agents == {"Planner", "Coder"}


def test_touch_only_tags_the_fixed_prompt():
    s = Setup()
    s.add([1, 2])                 # fixed prompt: 2 blocks
    leaf = s.add([1, 2, 3])       # same fixed prompt + a 1-block tail
    s.tree.touch(leaf, now=9, agent="Coder", static_tokens=2 * BLOCK_SIZE)

    assert "Coder" not in leaf.agents        # the tail belongs to nobody
    assert "Coder" in leaf.parent.agents     # the fixed prompt belongs to Coder
    assert leaf.last_access == 9             # but the tail's time is still updated
