import sys


import common_types

from flax.core import freeze
import jax
import jax.numpy as jnp
import max_utils
from jax.experimental import checkify

import pyconfig

from layers import attentions

Mesh = jax.sharding.Mesh
Attention = attentions.Attention

def get_data(rng, global_batch_size, max_target_length, embed_dim, dtype):
    lnx = jax.random.normal(
        rng,
        shape=(global_batch_size, max_target_length, embed_dim),
        dtype=dtype,
    )

    decoder_segment_ids = jax.random.randint(rng, (global_batch_size, max_target_length), 0, 4)
    decoder_positions = jax.random.randint(rng, (global_batch_size, max_target_length), 0, max_target_length)

    return lnx, decoder_segment_ids, decoder_positions

def main(cfg):

    global_batch_size = cfg.global_batch_size_to_train_on
    num_kv_heads = cfg.num_kv_heads
    num_query_heads = cfg.num_query_heads
    max_target_length = cfg.max_target_length
    head_dim = cfg.head_dim
    embed_dim = cfg.base_emb_dim
    dtype = cfg.dtype
    rng = jax.random.PRNGKey(0)

    devices_array = max_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    attention_as_mha_flash = Attention(
        config=cfg,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        mesh=mesh,
        attention_kernel = "flash",
        dtype=dtype,
        dropout_rate=cfg.dropout_rate,
        name='self_attention',
    )

    lnx, decoder_segment_ids, decoder_positions = get_data(rng, global_batch_size, 
                                                           max_target_length, embed_dim, dtype)

    attention_as_mha_flash_variable = attention_as_mha_flash.init(
        {'params': rng, 'aqt': rng},
        jnp.ones(
            (global_batch_size, max_target_length, embed_dim)),
        jnp.ones(
            (global_batch_size, max_target_length, embed_dim)),
        jnp.ones(
            (global_batch_size, max_target_length)),
    )

    mha_generic_flash_output = attention_as_mha_flash.apply(
        attention_as_mha_flash_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={'aqt': rng},
    )

    print(f"{mha_generic_flash_output.shape=}")

if __name__=='__main__':
    pyconfig.initialize([sys.argv[0], 'MaxText/configs/base.yml'], per_device_batch_size = 1.0, run_name='test', enable_checkpointing=False, max_target_length=128, max_prefill_predict_length=16 )
    cfg = pyconfig.config
    main(cfg)