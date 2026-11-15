# AgentKV benchmark results

200 requests per trace, 10 seeds, Llama-3-8B (2 MiB per 16-token block). Hit rate = cached blocks / full prompt blocks. Every number is an exact count from the simulator.

## Experiment 1+2: capacity sweep, 5 policies

### Cyclic: Planner -> Retriever -> Coder -> Reviewer -> Planner

| Blocks | MiB | LRU % | LRU-TailFirst % | AgentKV % | AgentKV-Markov % | Bélády % | Recomputed tokens LRU -> AgentKV | Cut |
|---|---|---|---|---|---|---|---|---|
| 120 | 240 | 13.1 | 13.1 | **13.1** | 13.1 | 13.1 | 301,056 -> 301,056 | 0% |
| 180 | 360 | 13.1 | 13.1 | **13.1** | 13.1 | 13.1 | 301,056 -> 301,056 | 0% |
| 240 | 480 | 13.1 | 13.1 | **39.2** | 38.8 | 39.2 | 301,056 -> 199,680 | 34% |
| 300 | 600 | 13.1 | 13.1 | **39.2** | 38.8 | 39.2 | 301,056 -> 199,680 | 34% |
| 360 | 720 | 13.1 | 13.1 | **64.9** | 64.5 | 64.9 | 301,056 -> 99,840 | 67% |
| 420 | 840 | 22.1 | 90.6 | **90.6** | 90.2 | 90.6 | 266,035 -> 0 | 100% |
| 480 | 960 | 90.6 | 90.6 | **90.6** | 90.6 | 90.6 | 0 -> 0 | 0% |

### Branching: Reviewer -> Coder 30% / -> Planner 70%

| Blocks | MiB | LRU % | LRU-TailFirst % | AgentKV % | AgentKV-Markov % | Bélády % | Recomputed tokens LRU -> AgentKV | Cut |
|---|---|---|---|---|---|---|---|---|
| 120 | 240 | 13.1 | 13.1 | **13.1** | 13.1 | 13.1 | 301,056 -> 301,056 | 0% |
| 180 | 360 | 13.1 | 13.1 | **13.1** | 13.1 | 13.1 | 301,056 -> 301,056 | 0% |
| 240 | 480 | 26.5 | 26.5 | **42.5** | 42.2 | 46.2 | 249,139 -> 186,931 | 25% |
| 300 | 600 | 26.5 | 26.5 | **42.5** | 42.2 | 46.2 | 249,139 -> 186,931 | 25% |
| 360 | 720 | 26.5 | 26.5 | **66.7** | 66.3 | 68.7 | 249,139 -> 93,082 | 63% |
| 420 | 840 | 33.4 | 90.6 | **90.6** | 90.2 | 90.6 | 222,413 -> 0 | 100% |
| 480 | 960 | 90.0 | 90.6 | **90.6** | 90.6 | 90.6 | 2,611 -> 0 | 100% |

```
Cyclic workflow, block hit rate
 120 blk  LRU      ##### 13.1%
 120 blk  AgentKV  ##### 13.1%
 120 blk  Belady   ##### 13.1%

 180 blk  LRU      ##### 13.1%
 180 blk  AgentKV  ##### 13.1%
 180 blk  Belady   ##### 13.1%

 240 blk  LRU      ##### 13.1%
 240 blk  AgentKV  ################ 39.2%
 240 blk  Belady   ################ 39.2%

 300 blk  LRU      ##### 13.1%
 300 blk  AgentKV  ################ 39.2%
 300 blk  Belady   ################ 39.2%

 360 blk  LRU      ##### 13.1%
 360 blk  AgentKV  ########################## 64.9%
 360 blk  Belady   ########################## 64.9%

 420 blk  LRU      ######### 22.1%
 420 blk  AgentKV  #################################### 90.6%
 420 blk  Belady   #################################### 90.6%

 480 blk  LRU      #################################### 90.6%
 480 blk  AgentKV  #################################### 90.6%
 480 blk  Belady   #################################### 90.6%

```

## Experiment 3: branching workflow at 300 blocks (600 MiB), 10 seeds

| Seed | LRU % | LRU-TailFirst % | AgentKV % | AgentKV-Markov % | Belady % |
|---|---|---|---|---|---|
| 0 | 22.5 | 22.5 | 39.4 | 39.8 | 43.8 |
| 1 | 29.0 | 29.0 | 44.4 | 44.0 | 47.6 |
| 2 | 22.6 | 22.6 | 39.6 | 38.9 | 44.4 |
| 3 | 27.5 | 27.5 | 43.4 | 43.4 | 46.5 |
| 4 | 27.4 | 27.4 | 43.2 | 43.2 | 46.8 |
| 5 | 28.1 | 28.1 | 43.6 | 43.6 | 46.7 |
| 6 | 27.3 | 27.3 | 43.1 | 42.7 | 47.1 |
| 7 | 25.8 | 25.8 | 42.1 | 42.1 | 45.6 |
| 8 | 30.5 | 30.5 | 45.4 | 44.3 | 49.0 |
| 9 | 24.2 | 24.2 | 40.8 | 40.0 | 44.7 |
| **mean ± std** | **26.5 ± 2.5** | **26.5 ± 2.5** | **42.5 ± 1.9** | **42.2 ± 1.8** | **46.2 ± 1.5** |

AgentKV beat LRU on 10/10 seeds and was 3.7 points below the Bélády oracle.

## Experiment 4: every workflow at 75% of its working set

| Workflow | Blocks (75% of working set) | LRU % | LRU-TailFirst % | AgentKV % | AgentKV-Markov % | Belady % | Result |
|---|---|---|---|---|---|---|---|
| cyclic | 300 | 13.1 | 13.1 | 39.2 | 38.8 | 39.2 | AgentKV wins |
| branching | 300 | 26.5 | 26.5 | 42.5 | 42.2 | 46.2 | AgentKV wins |
| react | 300 | 69.1 | 69.1 | 69.1 | 69.1 | 69.1 | tie |
| fanout | 372 | 13.1 | 13.1 | 44.0 | 44.0 | 44.0 | AgentKV wins |
| random | 300 | 39.2 | 39.2 | 39.2 | 39.1 | 53.9 | tie |

## Experiment 5: sensitivity (capacity = 75% of working set)

| Setting | Workflow | Blocks | LRU % | LRU-TailFirst % | AgentKV % | AgentKV-Markov % | Belady % | AgentKV - LRU (pts) |
|---|---|---|---|---|---|---|---|---|
| 4 agents | cyclic | 300 | 13.1 | 13.1 | 39.2 | 38.8 | 39.2 | +26.1 |
| 6 agents | cyclic | 444 | 13.1 | 13.1 | 59.3 | 58.9 | 59.3 | +46.2 |
| 8 agents | cyclic | 588 | 13.1 | 13.1 | 57.0 | 55.8 | 57.0 | +43.9 |
| agent prefix 512 tok | branching | 108 | 37.2 | 37.2 | 48.5 | 48.5 | 51.1 | +11.3 |
| agent prefix 1536 tok | branching | 300 | 26.5 | 26.5 | 42.5 | 42.2 | 46.2 | +16.0 |
| agent prefix 3072 tok | branching | 588 | 24.5 | 24.5 | 41.7 | 41.8 | 45.9 | +17.3 |
| tail 64-256 tok | branching | 300 | 26.5 | 26.5 | 42.5 | 42.2 | 46.2 | +16.0 |
| tail 256-1024 tok | branching | 300 | 21.2 | 21.2 | 34.0 | 33.8 | 37.0 | +12.8 |
| tail 1024-2048 tok | branching | 300 | 11.2 | 11.2 | 12.4 | 12.7 | 13.2 | +1.3 |

## Experiment 6: one scenario, four models (branching workflow, 360 blocks)

| Model | KV per token | Block size | Blocks in 600 MiB | Memory for 360 blocks | KV not rebuilt per trace (GiB) | Prefill TFLOPs saved per trace |
|---|---|---|---|---|---|---|
| llama-3-8b | 128 KiB | 2.00 MiB | 300 | 0.70 GiB | 19.1 | 2,497 |
| llama-3-70b | 320 KiB | 5.00 MiB | 120 | 1.76 GiB | 47.6 | 22,035 |
| qwen2.5-7b | 56 KiB | 0.88 MiB | 685 | 0.31 GiB | 8.3 | 2,372 |
| llama-2-7b | 512 KiB | 8.00 MiB | 75 | 2.81 GiB | 76.2 | 2,091 |

Same trace and same 360-block cache for every model: LRU 26.5% -> AgentKV 66.7%, 156,058 fewer recomputed tokens per trace. Llama-2-7B has no GQA (32 KV heads), so 600 MiB holds only 75 blocks, not even one agent prefix (96 blocks + 16 system).

## Experiment 7: GPU + CPU memory tiers with workflow-aware prefetching

Evicted agent prefixes move to a 1,024-block (2 GiB) CPU tier instead of being deleted; one-time request tails are deleted. Stall = recomputing evicted blocks + copying CPU blocks back while the request waits, modeled as A100 at 50% utilisation and PCIe 4.0 at 25 GB/s (1 block: 1.64 ms to recompute vs 0.08 ms to copy). Prefetched copies overlap the previous step, so they don't count as stall.

### cyclic, 300 GPU blocks (600 MiB)

| Setup | GPU hit % | Reused (GPU+CPU) % | Copied back on demand (blocks/request) | Prefetch accuracy % | Recomputed tokens | Modeled stall (ms/request) |
|---|---|---|---|---|---|---|
| LRU, GPU only | 13.1 | 13.1 | 0.0 | - | 301,056 | 154.39 |
| LRU + CPU tier | 13.1 | 90.6 | 94.1 | - | 0 | 7.89 |
| AgentKV + CPU tier, no prefetch | 39.2 | 90.6 | 62.4 | - | 0 | 5.23 |
| AgentKV + CPU tier + prefetch | 90.6 | 90.6 | 0.0 | 99.5 | 0 | 0.00 |
| AgentKV-Markov + CPU tier + prefetch | 90.2 | 90.6 | 0.5 | 99.5 | 0 | 0.04 |

### branching, 300 GPU blocks (600 MiB)

| Setup | GPU hit % | Reused (GPU+CPU) % | Copied back on demand (blocks/request) | Prefetch accuracy % | Recomputed tokens | Modeled stall (ms/request) |
|---|---|---|---|---|---|---|
| LRU, GPU only | 26.5 | 26.5 | 0.0 | - | 249,139 | 127.76 |
| LRU + CPU tier | 26.5 | 90.6 | 77.9 | - | 0 | 6.53 |
| AgentKV + CPU tier, no prefetch | 42.5 | 90.6 | 58.4 | - | 0 | 4.90 |
| AgentKV + CPU tier + prefetch | 74.6 | 90.6 | 19.5 | 99.9 | 0 | 1.63 |
| AgentKV-Markov + CPU tier + prefetch | 82.0 | 90.6 | 10.5 | 91.1 | 0 | 0.88 |

### branching, 360 GPU blocks (720 MiB)

| Setup | GPU hit % | Reused (GPU+CPU) % | Copied back on demand (blocks/request) | Prefetch accuracy % | Recomputed tokens | Modeled stall (ms/request) |
|---|---|---|---|---|---|---|
| LRU, GPU only | 26.5 | 26.5 | 0.0 | - | 249,139 | 127.76 |
| LRU + CPU tier | 26.5 | 90.6 | 77.9 | - | 0 | 6.53 |
| AgentKV + CPU tier, no prefetch | 66.7 | 90.6 | 29.1 | - | 0 | 2.44 |
| AgentKV + CPU tier + prefetch | 90.6 | 90.6 | 0.0 | 99.2 | 0 | 0.00 |
| AgentKV-Markov + CPU tier + prefetch | 89.3 | 90.6 | 1.6 | 97.4 | 0 | 0.14 |

### fanout, 360 GPU blocks (720 MiB)

| Setup | GPU hit % | Reused (GPU+CPU) % | Copied back on demand (blocks/request) | Prefetch accuracy % | Recomputed tokens | Modeled stall (ms/request) |
|---|---|---|---|---|---|---|
| LRU, GPU only | 13.1 | 13.1 | 0.0 | - | 299,520 | 153.60 |
| LRU + CPU tier | 13.1 | 90.3 | 93.6 | - | 0 | 7.85 |
| AgentKV + CPU tier, no prefetch | 44.0 | 90.3 | 56.2 | - | 0 | 4.71 |
| AgentKV + CPU tier + prefetch | 59.4 | 90.3 | 37.4 | 98.7 | 0 | 3.14 |
| AgentKV-Markov + CPU tier + prefetch | 59.0 | 90.3 | 37.9 | 98.7 | 0 | 3.18 |

## Headline numbers

- Across the full sweep (both workflows, 14 capacity points, per seed): recomputed prefill tokens 3,345,869 (LRU) -> 2,170,368 (AgentKV), a 35% cut, about 18,808 TFLOPs and 143 GiB of KV not rebuilt.
- Tiered, cyclic at 300 GPU blocks: modeled stall 154.4 ms (LRU, GPU only) -> 7.9 ms (+ CPU tier) -> 0.00 ms (AgentKV prefetch) per request.
- Tiered, branching at 300 GPU blocks: GPU hit rate 74.6% (BFS prefetch) vs 82.0% (Markov prefetch).
- Sanity check, AgentKV never below LRU in the sweep: PASS
- Sanity check, AgentKV == Bélády on the cyclic workflow: PASS

_Full run took 27.6 s._
