import jax 

from base import _Init, ArrayInfo, Config, jax_pytree_struct

@jax_pytree_struct
class KVCache(_Init):
    k_nope: list[jax.Array]  # [batch_size, max_seq_len, kv_lora]
    k_pe: list[jax.Array]  # [batch_size, max_seq_len, qk_rope_head_dim]
    v: list[jax.Array]  # [batch_size, max_seq_len, kv_lora]
    length: jax.Array  # []  # sequences are right-aligned for slice udpate performance
    starts: jax.Array  # [batch_size]  # sequences are right-aligned, we need start indices

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int, max_seq_len: int, dtype: int = jnp.bfloat16):
        _init = jax.nn.initializers.zeros
        k_nope_info = ArrayInfo(
            (batch_size, cfg.num_heads, max_seq_len, cfg.qk_nope_head_dim),
            dtype,
            ("batch", "qkv_heads", "sequence", "head_dim"),
            _init,
        )
        k_pe_info = ArrayInfo(
            (batch_size, max_seq_len, cfg.qk_rope_head_dim),
            dtype,
            ("batch", "sequence", "head_dim"),
            _init,
        )
        v_info = ArrayInfo(
            (batch_size, cfg.num_heads, max_seq_len, cfg.v_head_dim),
            dtype,
            ("batch", "qkv_heads", "sequence", "head_dim"),
            _init,
        )
        cache = KVCache(
            k_nope=[k_nope_info for _ in range(cfg.num_layers)],
            k_pe=[k_pe_info for _ in range(cfg.num_layers)],
            v=[v_info for _ in range(cfg.num_layers)],
            length=ArrayInfo((), jnp.int32, (), _init),
            starts=ArrayInfo((batch_size,), jnp.int32, ("batch",), _init),
        )
        return cache

    @property
    def time_axis(self) -> int:
        return 2
