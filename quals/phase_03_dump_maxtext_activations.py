# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Phase 3 Modular Block 2: MaxText Logit/Activation Dumper.
Loads un-scanned MaxText SFT baseline, applies class monkeypatches eagerly, and dumps activations.
"""

import os
import shutil
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from orbax import checkpoint as ocp
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils

# Disable JAX rematerialization at root module level
def no_op_remat(target, *args, **kwargs):
  return target

nn.remat = no_op_remat

# Dynamic capture target and call counters
active_activations = {}
q_call_count = 0
kv_call_count = 0
o_call_count = 0
mlp_call_count = 0

def unscan_checkpoint(scanned_dir, target_dir):
  """Loads GCS scanned parameter checkpoint, converts it to un-scanned, and saves it locally."""
  print("\n=== Step 3.1.5: Dynamically Converting Scanned Checkpoint ===")
  print(f"Source: {scanned_dir}")
  print(f"Target: {target_dir}")
  
  if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
    
  ckptr = ocp.StandardCheckpointer()
  scanned_pytree = ckptr.restore(scanned_dir)
  
  # Extract parameters sub-tree
  scanned_params = scanned_pytree['params']['params']
  unscanned_params = {
      'token_embedder': scanned_params['token_embedder'],
      'decoder': {
          'decoder_norm': scanned_params['decoder']['decoder_norm']
      }
  }
  
  # Un-stack 28 transformer layers using dynamic shape-based axis auto-detection
  scanned_layers = scanned_params['decoder']['layers']
  
  def slice_scanned_axis(x, lyr):
    axes_with_28 = [i for i, size in enumerate(x.shape) if size == 28]
    if not axes_with_28:
      return x
    scan_axis = axes_with_28[0]
    indexer = [slice(None)] * len(x.shape)
    indexer[scan_axis] = lyr
    return x[tuple(indexer)]

  for lyr in range(28):
    layer_key = f"layers_{lyr}"
    unscanned_params['decoder'][layer_key] = jax.tree.map(
        lambda x, l=lyr: slice_scanned_axis(x, l),
        scanned_layers
    )
    
  unscanned_pytree = {
      'params': {
          'params': unscanned_params
      }
  }
  
  ocp.PyTreeCheckpointer().save(target_dir, unscanned_pytree, force=True)
  print("Dynamic un-scanned parameter checkpoint successfully written.")


def main():
  print("\n=== Step 3.2: Extracting MaxText Logit Baselines & Layer 0 Activations ===")
  
  # Dynamically encode identical strings to guarantee 100% input parity
  # pylint: disable=import-outside-toplevel
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
  
  prompt = "What is DPO?"
  chosen = "DPO stands for Direct Preference Optimization, an algorithm for aligning LLMs."
  rejected = "DPO is a marketing strategy used to target customers' preferences."

  chosen_ids = tokenizer.encode(prompt + chosen)
  rejected_ids = tokenizer.encode(prompt + rejected)

  temp_unscanned_path = "/tmp/unscanned_sft_baseline"
  
  # Note: Dynamic unscan conversion executes once externally under simulated 16-device config
  if not os.path.exists(temp_unscanned_path):
    scanned_checkpoint_path = "gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_mcjax_v2/0/items"
    unscan_checkpoint(scanned_checkpoint_path, temp_unscanned_path)

  # 1. Initialize MaxText configurations pointing to unrolled local CPU temporary checkpoint
  argv = [
      "src/maxtext/configs/base.yml",
      "model_name=qwen2.5-1.5b",
      "tokenizer_path=Qwen/Qwen2.5-1.5B-Instruct",
      f"load_parameters_path={temp_unscanned_path}",
      "scan_layers=False",  # Required to unroll layers in Python for monkeypatching
      "attention=dot_product",
      "per_device_batch_size=1",
      "max_target_length=1024",
      "skip_jax_distributed_system=True",
      "enable_nnx=True",
      "pure_nnx=True",
      "pure_nnx_decoder=False",  # Required for Qwen2 block compatibility
      "remat_policy=full",       # Matches no-op JAX rematerialization
      "log_config=0"
  ]
  config = pyconfig.initialize_pydantic(argv)
  
  # 2. Create mesh and restore parameters using from_pretrained
  model, mesh = model_creation_utils.from_pretrained(config)
  
  # 3. Class-Level Monkeypatching (safely intercepts unrolled Linen submodules)
  from maxtext.layers.attentions import Attention
  from maxtext.layers.linears import MlpBlock

  original_query_proj = Attention.query_projection
  original_kv_proj = Attention.kv_projection
  original_out_proj = Attention.out_projection
  original_mlp = MlpBlock.__call__

  def custom_query_proj(self, inputs_q, out_sharding=None):
    global q_call_count
    out = original_query_proj(self, inputs_q, out_sharding=out_sharding)
    if q_call_count == 0:
      active_activations['q_proj'] = np.array(out)[0].astype(np.float32)
    q_call_count += 1
    return out

  def custom_kv_proj(self, inputs_kv, proj_name, out_sharding=None):
    global kv_call_count
    out = original_kv_proj(self, inputs_kv, proj_name, out_sharding=out_sharding)
    if kv_call_count == 0 and proj_name == "key":
      active_activations['k_proj'] = np.array(out)[0].astype(np.float32)
    elif kv_call_count == 0 and proj_name == "value":
      active_activations['v_proj'] = np.array(out)[0].astype(np.float32)
      kv_call_count += 1
    return out

  def custom_out_proj(self, out_attn, out_sharding=None):
    global o_call_count
    out = original_out_proj(self, out_attn, out_sharding=out_sharding)
    if o_call_count == 0:
      active_activations['o_proj'] = np.array(out)[0].astype(np.float32)
    o_call_count += 1
    return out

  def custom_mlp(self, inputs, deterministic=False):
    global mlp_call_count
    out = original_mlp(self, inputs, deterministic=deterministic)
    if mlp_call_count == 0:
      active_activations['mlp_out'] = np.array(out)[0].astype(np.float32)
    mlp_call_count += 1
    return out

  Attention.query_projection = custom_query_proj
  Attention.kv_projection = custom_kv_proj
  Attention.out_projection = custom_out_proj
  MlpBlock.__call__ = custom_mlp

  # 4. Execute eager JAX forward passes inside the mesh context
  chosen_activations = {}
  rejected_activations = {}
  
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    chosen_jax = jnp.array([chosen_ids], dtype=jnp.int32)
    rejected_jax = jnp.array([rejected_ids], dtype=jnp.int32)
    
    chosen_pos = jnp.arange(len(chosen_ids), dtype=jnp.int32)[None, :]
    rejected_pos = jnp.arange(len(rejected_ids), dtype=jnp.int32)[None, :]
    
    # Extract embedding layer lookup outputs directly
    jax_embeds = model.token_embedder(chosen_jax)
    jax_embeds_np = np.array(jax_embeds)[0].astype(np.float32)

    # 1. Execute Chosen Pass statefully
    global active_activations, q_call_count, kv_call_count, o_call_count, mlp_call_count
    active_activations = chosen_activations
    q_call_count = kv_call_count = o_call_count = mlp_call_count = 0
    
    print("Running MaxText forward pass for chosen sequence...")
    chosen_logits_jax = model(chosen_jax, chosen_pos)
    chosen_logits_np = np.array(chosen_logits_jax)[0].astype(np.float32)

    # 2. Execute Rejected Pass statefully
    active_activations = rejected_activations
    q_call_count = kv_call_count = o_call_count = mlp_call_count = 0
    
    print("Running MaxText forward pass for rejected sequence...")
    rejected_logits_jax = model(rejected_jax, rejected_pos)
    rejected_logits_np = np.array(rejected_logits_jax)[0].astype(np.float32)

  # Save MaxText output arrays locally for modular auditing
  os.makedirs("quals/logs", exist_ok=True)
  np.save("quals/logs/maxtext_chosen_embeds.npy", jax_embeds_np)
  np.save("quals/logs/maxtext_chosen_logits.npy", chosen_logits_np)
  np.save("quals/logs/maxtext_rejected_logits.npy", rejected_logits_np)
  
  for name, val in chosen_activations.items():
    np.save(f"quals/logs/maxtext_chosen_{name}.npy", val)
  for name, val in rejected_activations.items():
    np.save(f"quals/logs/maxtext_rejected_{name}.npy", val)
    
  print(f"MaxText Chosen embeddings shape: {jax_embeds_np.shape}")
  print(f"MaxText Chosen logits shape: {chosen_logits_np.shape}")
  print("SUCCESS: MaxText baseline activations successfully dumped to quals/logs/")

if __name__ == "__main__":
  main()
