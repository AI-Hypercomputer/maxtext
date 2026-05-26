# Indexer Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the indexer implementations in `attention_mla.py` and `attention_compressed.py` to use a shared `BaseIndexer` base class.

**Architecture:** We will create a `BaseIndexer` class containing the shared `prepare_query` and `compute_topk` logic. The existing `Indexer` and `DeepSeekV4Indexer` will inherit from this class and delegate their query preparation and scoring logic to it, while retaining their unique Key preparation and masking logic.

**Tech Stack:** JAX, Flax NNX.

---

### Task 1: Implement BaseIndexer and MLAIndexer

**Files:**
- Modify: `src/maxtext/layers/attention_mla.py`
- Test: `tests/unit/paged_attention_test.py` (Assuming this exercises MLA, otherwise rely on compilation/syntax checks for this purely structural refactor).

- [ ] **Step 1: Write BaseIndexer**
Add the `BaseIndexer` class to `src/maxtext/layers/attention_mla.py` (above `Indexer`).
```python
class BaseIndexer(nnx.Module):
  def __init__(self, config: Any, kernel_init, rngs: Optional[nnx.Rngs] = None):
    self.config = config
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.n_heads = config.indexer_n_heads
    self.head_dim = config.indexer_head_dim
    self.indexer_topk = config.indexer_topk
    self.q_lora_rank = config.q_lora_rank
    self.softmax_scale = self.head_dim**-0.5
    self.weights_scaling = self.n_heads**-0.5 # Optional, used mostly by V4
    
    # Common projections
    self.wq_b = DenseGeneral(
        in_features_shape=self.q_lora_rank,
        out_features_shape=(self.n_heads, self.head_dim),
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("q_lora", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        rngs=rngs,
    )
    self.weights_proj = DenseGeneral(
        in_features_shape=config.emb_dim, # V3 uses emb_dim, V4 uses hidden_size
        out_features_shape=self.n_heads,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "q_heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        rngs=rngs,
    )

  def prepare_query(self, low_rank_q: jnp.ndarray, inputs_positions: jnp.ndarray, apply_partial_rope) -> jnp.ndarray:
    bsz, seqlen, _ = low_rank_q.shape
    q = self.wq_b(low_rank_q)
    q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim)
    q = apply_partial_rope(q, inputs_positions=inputs_positions)
    return q

  def compute_topk(self, q: jnp.ndarray, k: jnp.ndarray, inputs_q: jnp.ndarray, mask: Optional[jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    logits = jnp.einsum("bthd, bsd -> btsh", q, k, precision=self.config.matmul_precision)
    logits = jax.nn.relu(logits)
    weights = self.weights_proj(inputs_q)
    weights = weights * self.weights_scaling * self.softmax_scale
    indexer_score = jnp.einsum("btsh, bth -> bts", logits, weights, precision=self.config.matmul_precision)
    
    if mask is not None:
      indexer_score += mask
      
    _, topk_indices = jax.lax.top_k(indexer_score, k=self.indexer_topk)
    return topk_indices, indexer_score
```

- [ ] **Step 2: Refactor Indexer (V3) to use BaseIndexer**
Update `Indexer` to inherit from `BaseIndexer`. Remove duplicated initialization and scoring logic. Delegate to `self.prepare_query` and `self.compute_topk`.

- [ ] **Step 3: Test and Commit**
Verify syntax and structural correctness. (Will verify via tests later).
```bash
git add src/maxtext/layers/attention_mla.py
git commit -m "refactor: extract BaseIndexer and refactor MLA Indexer"
```

### Task 2: Refactor DeepSeekV4Indexer

**Files:**
- Modify: `src/maxtext/layers/attention_compressed.py`
- Test: `tests/unit/deepseek_v4_vs_reference_test.py`

- [ ] **Step 1: Refactor DeepSeekV4Indexer**
Update `DeepSeekV4Indexer` to inherit from `BaseIndexer` (imported from `attention_mla`).
Remove `q_b_proj` and `weights_proj` initialization.
Refactor `__call__` to use `self.prepare_query` and `self.compute_topk`.

- [ ] **Step 2: Run test to verify it passes**
Run: `pytest tests/unit/deepseek_v4_vs_reference_test.py::DeepseekV4AttentionTest::test_indexer_parity -v`
Expected: PASS

- [ ] **Step 3: Commit**
```bash
git add src/maxtext/layers/attention_compressed.py
git commit -m "refactor: update DeepSeekV4Indexer to use BaseIndexer"
```