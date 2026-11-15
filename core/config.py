"""
Model and memory constants.

All numbers are for Llama-3-8B with an FP16 KV cache.
Llama-3-8B uses Grouped Query Attention (GQA): it has 32 query heads but
only 8 key/value heads, which is why the KV cache is smaller than you might expect.
"""

# ---- Llama-3-8B architecture ----
NUM_LAYERS = 32
NUM_KV_HEADS = 8          # GQA: 8 KV heads (not 32)
HEAD_DIM = 128
BYTES_PER_VALUE = 2       # FP16 = 2 bytes
MODEL_PARAMS = 8.0e9      # ~8 billion parameters

# ---- Paging (vLLM / PagedAttention) ----
BLOCK_SIZE = 16           # tokens per physical KV block

# For every token we store one K vector and one V vector, in every layer, for every KV head.
# 2 (K and V) * 32 layers * 8 heads * 128 dims * 2 bytes = 131,072 bytes = 128 KiB
BYTES_PER_TOKEN = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_VALUE

# 16 tokens * 128 KiB = 2 MiB per block
BYTES_PER_BLOCK = BLOCK_SIZE * BYTES_PER_TOKEN

# A forward pass costs roughly 2 FLOPs per parameter per token.
# We use this to estimate how much prefill compute a cache hit saves.
FLOPS_PER_TOKEN = 2 * MODEL_PARAMS

KIB = 1024
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


# ---- Other models, used to compare how much KV memory a token costs ----
# Same formula as above. Grouped Query Attention (fewer KV heads) makes a huge difference:
# Llama-2-7B has 32 KV heads (no GQA), so each token costs 4x more than Llama-3-8B.
MODEL_PRESETS = {
    "llama-3-8b":  {"layers": 32, "kv_heads": 8,  "head_dim": 128, "params": 8.0e9},
    "llama-3-70b": {"layers": 80, "kv_heads": 8,  "head_dim": 128, "params": 70.6e9},
    "qwen2.5-7b":  {"layers": 28, "kv_heads": 4,  "head_dim": 128, "params": 7.6e9},
    "llama-2-7b":  {"layers": 32, "kv_heads": 32, "head_dim": 128, "params": 6.7e9},
}


def model_bytes_per_token(model_name):
    m = MODEL_PRESETS[model_name]
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * BYTES_PER_VALUE


def model_bytes_per_block(model_name):
    return BLOCK_SIZE * model_bytes_per_token(model_name)


def blocks_to_mib(num_blocks):
    """How many MiB of GPU memory this many blocks would take."""
    return num_blocks * BYTES_PER_BLOCK / MIB


def tokens_to_gib(num_tokens):
    """How many GiB of KV cache this many tokens would take."""
    return num_tokens * BYTES_PER_TOKEN / GIB


def tokens_to_tflops(num_tokens):
    """Estimated prefill compute (in TFLOPs) needed to process this many tokens."""
    return num_tokens * FLOPS_PER_TOKEN / 1e12


# ---- Cost model for the GPU + CPU tiered engine ----
# These are ESTIMATES used only to turn exact block counts into rough time.
# A100 GPU: 312 TFLOPS peak (BF16), assume 50% utilisation during prefill.
# PCIe 4.0 x16: ~25 GB/s of real-world bandwidth between CPU RAM and GPU.
GPU_PEAK_FLOPS = 312e12
GPU_UTILISATION = 0.5
PCIE_BYTES_PER_SEC = 25e9


def recompute_ms(num_tokens):
    """Time to recompute the KV cache for num_tokens with prefill (estimate)."""
    return num_tokens * FLOPS_PER_TOKEN / (GPU_PEAK_FLOPS * GPU_UTILISATION) * 1000


def transfer_ms(num_blocks):
    """Time to copy num_blocks from CPU RAM to the GPU over PCIe (estimate)."""
    return num_blocks * BYTES_PER_BLOCK / PCIE_BYTES_PER_SEC * 1000
