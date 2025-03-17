import jax

def mla_attention_block(
    x: jax.Array,
    segment_ids: jax.Array,
    attn_layer: AttentionLayer,
    sin: jax.Array,
    cos: jax.Array,
    cfg: Config,
    cache: KVCache | None = None,
    idx: int = 0,
) -> jax.Array:
    dtype = cfg.weight_dtype
    with jax.named_scope("q_embed"):
        q_lora = einsum("btd,dr->btr", x, attn_layer.q_a).astype(dtype)
        q_lora = rms_norm(q_lora, attn_layer.q_gamma).astype(dtype)
        q = einsum("btr,rhq->bhtq", q_lora, attn_layer.q_b).astype(dtype)
        q_nope = q[..., : cfg.qk_nope_head_dim]
        q_pe = apply_rotary_embedding(q[..., cfg.qk_nope_head_dim :], sin, cos).astype(dtype)

    with jax.named_scope("kv_compressed_embed"):
        kv_compressed = einsum("btd,dr->btr", x, attn_layer.kv_a).astype(dtype)
        kv_compressed = rms_norm(kv_compressed, attn_layer.kv_gamma).astype(dtype)
        k_pe = einsum("btd,dq->btq", x, attn_layer.k_pe)
        k_pe = apply_rotary_embedding(k_pe[..., None, :, :], sin, cos)[..., 0, :, :].astype(dtype)

    with jax.named_scope("kv_embed"):
        k_nope = einsum("btr,rhq->bhtq", kv_compressed, attn_layer.k_b)
        v = einsum("btr,rhv->bhtv", kv_compressed, attn_layer.v_b)

    with jax.named_scope("full_cache_update"):
        if cache is not None:
            k_nope = update_slice(cache.k_nope[idx], k_nope, cache.length, update_axis=cache.time_axis)
            k_pe = update_slice(cache.k_pe[idx], k_pe, cache.length, update_axis=cache.time_axis - 1)
            v = update_slice(cache.v[idx], v, cache.length, update_axis=cache.time_axis)
            cache_updates = (k_nope, k_pe, v)
        else:
            cache_updates = None

    # constrain the sharding of intermediates for the attention operation
    lsc = partial(logical_sharding_constraint, mesh=cfg.mesh, rules=cfg.rules)
    spec = ("batch", "act_heads", "sequence", "head_dim")
    q_nope, q_pe = lsc(q_nope, spec), lsc(q_pe, spec)
    k_nope, k_pe, v = lsc(k_nope, spec), lsc(k_pe, ("batch", "sequence", "head_dim")), lsc(v, spec)

    # create position embeddings
    if cache is not None:
        time_indices = jnp.arange(0, v.shape[-2])[None, :]  # [1, T]
        q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
        incremental_position = jnp.max(_count_length_from_left(segment_ids))
        # i.e. valid below where we've written things [B, T]
        k_segment_ids = (
            (time_indices >= cache.starts[:, None]) & (time_indices < (cache.length + incremental_position))
        ).astype(jnp.int32)

        q_offset = cache.length[None]
        starts, lengths = cache.starts, (cache.length + incremental_position)[None]
    else:
        q_segment_ids, k_segment_ids = segment_ids, segment_ids
        q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)
        starts = jnp.sum(jnp.cumsum(k_segment_ids != 0, axis=-1) == 0, axis=-1)
        lengths = _count_length_from_left(k_segment_ids)

    # Compute attention
    with jax.named_scope("attention"):
        if (cfg.use_prefill_attn_kernel and q.shape[-2] != 1) or (cfg.use_decode_attn_kernel and q.shape[-2] == 1):
            raise NotImplementedError
            attn_out = attention_kernel(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                v,
                q_segment_ids,
                k_segment_ids,
                q_offset,
                starts=starts,
                lengths=lengths,
                cfg=cfg,
            )
        else:
            attn_out = attention(q_nope, q_pe, k_nope, k_pe, v, q_segment_ids, k_segment_ids, q_offset, cfg)

    with jax.named_scope("o_proj"):
        attn_out = einsum("bhtv,hvd->btd", attn_out, attn_layer.o)
    attn_out = lsc(attn_out.astype(cfg.weight_dtype), ("batch", "sequence", "act_embed"))
    return attn_out, cache_updates
