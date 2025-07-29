"""test_logit.py
cd ~/maxtext
python -m MaxText.test_logit
"""

import jax
from flax.linen import partitioning as nn_partitioning
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.layers import models, quantizations

Transformer = models.Transformer


def debug_sharding(array, prefix=""):
  global_shape = array.shape
  jax.debug.inspect_array_sharding(
      array,
      callback=lambda sharding_obj: print(
          prefix + f"\tGlobal Shape: {global_shape}\n"
          f"\tLocal Shape: {sharding_obj.shard_shape(global_shape)}\n"
          f"\tSharding Object: {sharding_obj}\n"
      ),
  )


class DataIter:

  def __init__(self, rng, max_target_length, global_batch_size_to_train_on, base_emb_dim):
    self.max_target_length = max_target_length
    self.global_batch_size = global_batch_size_to_train_on
    self.embed_dim = base_emb_dim
    self.rng = rng

  def get_data(self):
    """get data"""
    lnx = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 100, dtype=jax.numpy.int32)
    decoder_segment_ids = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        self.rng, (self.global_batch_size, self.max_target_length), 0, self.max_target_length
    )
    return lnx, decoder_segment_ids, decoder_positions


def init(argv, rng):
  config = pyconfig.initialize(argv)
  quant = quantizations.configure_quantization(config)
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  maxtext_model = models.Transformer(config, mesh, quant=quant)
  maxtext_state, _ = maxtext_utils.setup_decode_state(maxtext_model, config, rng, mesh, None)
  return maxtext_model, maxtext_state, mesh, config


def cp_forward(cfg_cp, mesh_cp, rng, maxtext_model, maxtext_state, lnx, decoder_positions, decoder_segment_ids):
  # If load balanced cp, shuffle along seq dim for input
  # This correponds to the pre-shuffle step in training
  context_parallel_size = cfg_cp.ici_context_parallelism
  # ep acts like cp
  if cfg_cp.expert_shard_attention_option == "context":
    context_parallel_size = context_parallel_size * cfg_cp.ici_expert_parallelism
  if context_parallel_size > 1 and cfg_cp.context_parallel_load_balance:
    batch = {"inputs": lnx, "inputs_segmentation": decoder_segment_ids, "inputs_position": decoder_positions}
    with mesh_cp:
      reordered_batch = max_utils.get_reorder_callable(context_parallel_size)(batch)
    lnx = reordered_batch["inputs"]
    decoder_segment_ids = reordered_batch["inputs_segmentation"]
    decoder_positions = reordered_batch["inputs_position"]
  # apply attention with sharding
  with mesh_cp, nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
    out2 = maxtext_model.apply(
        maxtext_state.params,
        decoder_input_tokens=lnx,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        # train mode
        model_mode=MODEL_MODE_TRAIN,
        # deterministic
        enable_dropout=False,
        rngs={"aqt": rng},
    )
  # If load balanced cp, de-shuffle and gather along seq dim for output
  # Note training does not need post-shuffle. Since the target seq is also pre-shuffled, the loss remains correct
  if context_parallel_size > 1 and cfg_cp.context_parallel_load_balance:
    out2 = max_utils.reorder_sequence(tensor=out2, cp_size=context_parallel_size, seq_dim=1, to_contiguous=True)
  return out2


def main(model_name="mixtral-8x7b") -> None:
  rng = jax.random.PRNGKey(0)
  argv = [
      "something.py",
      "MaxText/configs/base.yml",
      "per_device_batch_size=1",
      "max_target_length=8192",
      "skip_jax_distributed_system=true",
      f"model_name={model_name}",
  ]
  config = pyconfig.initialize(argv)
  data = DataIter(rng, config.max_target_length, config.global_batch_size_to_train_on, config.base_emb_dim)
  lnx, decoder_segment_ids, decoder_positions = data.get_data()

  # model1: dot_product
  argv1 = argv + ["attention=dot_product"]
  maxtext_model, maxtext_state, mesh, config = init(argv1, rng)
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    out1 = maxtext_model.apply(
        maxtext_state.params,
        decoder_input_tokens=lnx,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        # train mode
        model_mode=MODEL_MODE_TRAIN,
        # deterministic
        enable_dropout=False,
        rngs={"aqt": rng},
    )

  # model2: cp
  argv2 = argv + ["attention=flash", "ici_context_parallelism=4", "context_parallel_load_balance=True"]
  maxtext_model, maxtext_state, mesh_cp, cfg_cp = init(argv2, rng)
  out2 = cp_forward(cfg_cp, mesh_cp, rng, maxtext_model, maxtext_state, lnx, decoder_positions, decoder_segment_ids)
  assert jax.numpy.allclose(out1, out2, rtol=1e-01, atol=1e-01, equal_nan=False), 2

  # model3: cp
  argv3 = argv + ["attention=flash", "ici_context_parallelism=4", "context_parallel_load_balance=False"]
  maxtext_model, maxtext_state, mesh_cp, cfg_cp = init(argv3, rng)
  out3 = cp_forward(cfg_cp, mesh_cp, rng, maxtext_model, maxtext_state, lnx, decoder_positions, decoder_segment_ids)
  assert jax.numpy.allclose(out1, out3, rtol=1e-01, atol=1e-01, equal_nan=False), 3

  # model4: cp + ep_as_cp
  argv4 = argv + [
      "attention=flash",
      "ici_context_parallelism=2",
      "context_parallel_load_balance=True",
      "ici_expert_parallelism=2",
      "expert_shard_attention_option=context",
  ]
  maxtext_model, maxtext_state, mesh_cp, cfg_cp = init(argv4, rng)
  out4 = cp_forward(cfg_cp, mesh_cp, rng, maxtext_model, maxtext_state, lnx, decoder_positions, decoder_segment_ids)
  # debug_sharding(out1)
  # debug_sharding(out4)
  # ValueError: Received incompatible devices for jitted computation.
  # Got argument a of allclose with shape float32[4,8192,32000] and device ids [0, 2, 1, 3] on platform TPU
  # and argument b of allclose with shape float32[4,8192,32000] and device ids [0, 1, 2, 3] on platform TPU
  out4 = jax.device_put(out4, out1.sharding)  # need reshard
  assert jax.numpy.allclose(out1, out4, rtol=1e-01, atol=1e-01, equal_nan=False), 4

  # model5: cp + ep_as_fsdp
  argv5 = argv + [
      "attention=flash",
      "ici_context_parallelism=2",
      "context_parallel_load_balance=True",
      "ici_expert_parallelism=2",
      "expert_shard_attention_option=fsdp",
  ]
  maxtext_model, maxtext_state, mesh_cp, cfg_cp = init(argv5, rng)
  out5 = cp_forward(cfg_cp, mesh_cp, rng, maxtext_model, maxtext_state, lnx, decoder_positions, decoder_segment_ids)
  out5 = jax.device_put(out5, out1.sharding)  # need reshard
  assert jax.numpy.allclose(out1, out5, rtol=1e-01, atol=1e-01, equal_nan=False), 5

  # model6: ep_as_cp
  argv6 = argv + [
      "attention=flash",
      "ici_context_parallelism=1",
      "context_parallel_load_balance=True",
      "ici_expert_parallelism=4",
      "expert_shard_attention_option=context",
  ]
  maxtext_model, maxtext_state, mesh_cp, cfg_cp = init(argv6, rng)
  out6 = cp_forward(cfg_cp, mesh_cp, rng, maxtext_model, maxtext_state, lnx, decoder_positions, decoder_segment_ids)
  assert jax.numpy.allclose(out1, out6, rtol=1e-01, atol=1e-01, equal_nan=False), 6

  # model7: ep_as_fsdp
  argv7 = argv + [
      "attention=flash",
      "ici_context_parallelism=1",
      "context_parallel_load_balance=True",
      "ici_expert_parallelism=4",
      "expert_shard_attention_option=fsdp",
  ]
  maxtext_model, maxtext_state, mesh_cp, cfg_cp = init(argv7, rng)
  out7 = cp_forward(cfg_cp, mesh_cp, rng, maxtext_model, maxtext_state, lnx, decoder_positions, decoder_segment_ids)
  assert jax.numpy.allclose(out1, out7, rtol=1e-01, atol=1e-01, equal_nan=False), 7


if __name__ == "__main__":
  # base_num_decoder_layers: 2
  main("mixtral-8x7b")
  # base_emb_dim: 512, base_moe_mlp_dim: 512, base_num_decoder_layers: 2, first_num_dense_layers: 2
  main("deepseek3-671b")
