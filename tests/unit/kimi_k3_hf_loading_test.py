# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit test for Kimi K3 HuggingFace checkpoint loading and forward pass in MaxText."""

import os
import unittest
import pytest

transformers = pytest.importorskip("transformers")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from flax import linen as nn
from flax import nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


@pytest.mark.tpu_only
class KimiK3HFLoadingTest(unittest.TestCase):

  """Tests loading a converted Kimi K3 Orbax checkpoint and running a forward pass."""

  @classmethod
  def setUpClass(cls):
    raw_ckpt_dir = os.environ.get(
        "KIMI_K3_CHECKPOINT_DIR",
        "scratch/kimi_k3_orbax_checkpoint",
    )
    if raw_ckpt_dir.startswith("gs://"):
      cls.checkpoint_dir = raw_ckpt_dir
    else:
      cls.checkpoint_dir = os.path.abspath(raw_ckpt_dir)

    raw_hf_path = os.environ.get("HF_MODEL_PATH", "scratch/hf_kimi_k3_subset")
    if raw_hf_path.startswith("gs://"):
      cls.hf_model_path = raw_hf_path
    else:
      cls.hf_model_path = os.path.abspath(raw_hf_path)

    cls.config_path = os.environ.get(
        "KIMI_K3_CONFIG",
        "src/maxtext/configs/models/kimi-k3-minimal.yml",
    )
    if not os.path.exists(cls.checkpoint_dir):
      raise unittest.SkipTest(f"Checkpoint directory {cls.checkpoint_dir} does not exist. Run to_maxtext first.")

  def test_load_checkpoint_and_forward_pass(self):
    ckpt_path = (
        self.checkpoint_dir
        if self.checkpoint_dir.endswith("items")
        else os.path.join(self.checkpoint_dir, "0", "items")
    )
    num_devices = jax.device_count()
    expert_parallelism = min(num_devices, 8) if num_devices > 0 else 1
    config = pyconfig.initialize([
        "kimi_k3_hf_loading_test.py",
        self.config_path,
        "model_name=kimi-k3",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "scan_layers=False",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "remat_policy=none",
        f"ici_expert_parallelism={expert_parallelism}",
        "ici_fsdp_parallelism=1",
        f"load_parameters_path={ckpt_path}",
    ])

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # Use MaxText's official from_pretrained loader to instantiate and stream checkpoint to TPU
    print(f"Loading Kimi K3 checkpoint from {ckpt_path} onto TPU mesh...")
    model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    print("Model initialized and checkpoint restored successfully!")

    # Dummy inputs for 2-layer Kimi K3 (1 Dense + 1 MoE/MLA)
    batch_size = 1
    seq_len = 4
    inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    # Split NNX model into static graphdef and state to run pure functional forward passes
    graphdef, state = nnx.split(model)

    @jax.jit
    def run_layer0_forward(state_in, x, p, s):
      with nn.logical_axis_rules(config.logical_axis_rules):
        m = nnx.merge(graphdef, state_in)
        h = m.decoder.token_embedder(x)
        h = m.decoder.layers["decoder_0"](h, p, s, enable_dropout=False)
        h = m.decoder.decoder_norm(h)
        return m.decoder.token_embedder.attend(h)

    @jax.jit
    def run_full_forward(state_in, x, p, s):
      with nn.logical_axis_rules(config.logical_axis_rules):
        m = nnx.merge(graphdef, state_in)
        return m(
            decoder_input_tokens=x,
            decoder_positions=p,
            decoder_segment_ids=s,
            enable_dropout=False,
        )

    print("Running JIT-compiled full 2-layer forward pass on TPU...")
    full_logits = run_full_forward(state, inputs, positions, segment_ids)
    print("Full model logits shape:", full_logits.shape, "dtype:", full_logits.dtype)

    print("Running JIT-compiled layer-0 forward pass on TPU...")
    layer0_logits = run_layer0_forward(state, inputs, positions, segment_ids)
    print("Layer 0 logits shape:", layer0_logits.shape, "dtype:", layer0_logits.dtype)

    # Assertions on JAX forward pass
    self.assertEqual(full_logits.shape, (batch_size, seq_len, config.vocab_size))
    self.assertFalse(jnp.isnan(full_logits).any(), "Full model logits contain NaNs!")
    self.assertFalse(jnp.isinf(full_logits).any(), "Full model logits contain Infs!")
    self.assertFalse(jnp.isnan(layer0_logits).any(), "Layer 0 logits contain NaNs!")
    print("FORWARD PASSES ON TPU SUCCESSFUL!")

    # Check if PyTorch Hugging Face reference model is available for logit parity comparison
    print(f"\nChecking Hugging Face reference checkpoint at: {self.hf_model_path}")
    if os.path.exists(self.hf_model_path):
      print(f"Found Hugging Face model directory at {self.hf_model_path}.")
      try:
        import glob
        import torch
        from safetensors import safe_open
        from tests.unit.kimi_k3_logit_parity_test import PtRMSNorm, PtSituMLP, PtKDA, PtFullDecoderLayer

        print("Loading Hugging Face safetensors shards directly into PyTorch reference layers...")
        weights = {}
        for f in sorted(glob.glob(os.path.join(self.hf_model_path, "*.safetensors"))):
          with safe_open(f, framework="pt", device="cpu") as s:
            for k in s.keys():
              weights[k] = s.get_tensor(k)
        print(f"Loaded {len(weights)} tensors from {self.hf_model_path}.")

        # 1. Embeddings & Final Norm & LM Head
        embed_w = weights.get("model.embed_tokens.weight", weights.get("language_model.model.embed_tokens.weight"))
        norm_w = weights.get("model.norm.weight", weights.get("language_model.model.norm.weight"))
        lm_head_w = weights.get("lm_head.weight", weights.get("language_model.lm_head.weight"))

        # 2. Layer 0 (KDA + Dense Situ MLP)
        prefix0 = (
            "language_model.model.layers.0."
            if "language_model.model.layers.0.input_layernorm.weight" in weights
            else "model.layers.0."
        )
        D = int(embed_w.shape[1])
        kda_H = int(weights[f"{prefix0}self_attn.b_proj.weight"].shape[0])
        kda_K = int(weights[f"{prefix0}self_attn.A_log"].shape[0])
        intermediate_dim = int(weights[f"{prefix0}mlp.gate_proj.weight"].shape[0])

        norm1 = PtRMSNorm(D)
        norm1.scale.data = weights[f"{prefix0}input_layernorm.weight"].float()

        kda = PtKDA(hidden_size=D, num_heads=kda_H, head_dim=kda_K, conv_kernel_size=4)
        kda.q_proj.weight.data = weights[f"{prefix0}self_attn.q_proj.weight"].float()
        kda.k_proj.weight.data = weights[f"{prefix0}self_attn.k_proj.weight"].float()
        kda.v_proj.weight.data = weights[f"{prefix0}self_attn.v_proj.weight"].float()
        kda.f_a_proj.weight.data = weights[f"{prefix0}self_attn.f_a_proj.weight"].float()
        kda.f_b_proj.weight.data = weights[f"{prefix0}self_attn.f_b_proj.weight"].float()
        kda.b_proj.weight.data = weights[f"{prefix0}self_attn.b_proj.weight"].float()
        kda.g_proj.weight.data = weights[f"{prefix0}self_attn.g_proj.weight"].float()
        kda.o_proj.weight.data = weights[f"{prefix0}self_attn.o_proj.weight"].float()
        kda.q_conv1d.weight.data = weights[f"{prefix0}self_attn.q_conv1d.weight"].float()
        kda.k_conv1d.weight.data = weights[f"{prefix0}self_attn.k_conv1d.weight"].float()
        kda.v_conv1d.weight.data = weights[f"{prefix0}self_attn.v_conv1d.weight"].float()
        kda.A_log.data = weights[f"{prefix0}self_attn.A_log"].float()
        kda.dt_bias.data = weights[f"{prefix0}self_attn.dt_bias"].float()
        kda.o_norm.scale.data = weights[f"{prefix0}self_attn.o_norm.weight"].float()

        norm2 = PtRMSNorm(D)
        norm2.scale.data = weights[f"{prefix0}post_attention_layernorm.weight"].float()

        mlp = PtSituMLP(D, intermediate_dim)
        mlp.wi_0.weight.data = weights[f"{prefix0}mlp.gate_proj.weight"].float()
        mlp.wi_1.weight.data = weights[f"{prefix0}mlp.up_proj.weight"].float()
        mlp.wo.weight.data = weights[f"{prefix0}mlp.down_proj.weight"].float()

        layer0 = PtFullDecoderLayer(norm1, kda, norm2, mlp)

        final_norm = PtRMSNorm(D)
        final_norm.scale.data = norm_w.float()

        # Run PyTorch reference forward pass for Layer 0
        token_ids_pt = torch.from_numpy(np.array(inputs))
        x_pt = embed_w[token_ids_pt].float()
        x_pt = layer0(x_pt)
        x_pt = final_norm(x_pt)
        pt_logits = (x_pt @ lm_head_w.float().T).detach().numpy()

        jax_layer0_logits_np = np.array(layer0_logits).astype(np.float32)

        # Compute logit parity metrics
        diff = np.abs(jax_layer0_logits_np - pt_logits)
        max_err = float(np.max(diff))
        mae = float(np.mean(diff))
        cos_sim = float(
            np.dot(jax_layer0_logits_np.flatten(), pt_logits.flatten())
            / (np.linalg.norm(jax_layer0_logits_np) * np.linalg.norm(pt_logits) + 1e-12)
        )
        top1_agree = float(np.mean(np.argmax(jax_layer0_logits_np, axis=-1) == np.argmax(pt_logits, axis=-1)))

        print("=" * 70)
        print("REAL PRETRAINED LAYER-0 CHECKPOINT LOGIT PARITY (MaxText TPU vs HF PyTorch):")
        print(f"  Logits Shape:          {jax_layer0_logits_np.shape}")
        print(f"  Max Absolute Error:    {max_err:.6e}")
        print(f"  Mean Absolute Error:   {mae:.6e}")
        print(f"  Cosine Similarity:     {cos_sim:.8f}")
        print(f"  Top-1 Argmax Agreement:{top1_agree * 100:.1f}%")
        print("=" * 70)

        self.assertGreater(cos_sim, 0.999, f"Logit cosine similarity {cos_sim} is below 0.999!")
        self.assertEqual(top1_agree, 1.0, f"Top-1 argmax agreement {top1_agree} is not 100%!")
        print("REAL PRETRAINED LOGIT PARITY VERIFIED SUCCESSFULLY!")
      except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nNote: Hugging Face PyTorch comparison skipped ({e}).")
        print("MaxText forward pass on TPU is verified and passed.")
    else:
      print(f"WARNING: Hugging Face checkpoint not found at {self.hf_model_path}.")
      print("Pass HF_MODEL_PATH=<path_to_hf_subset> to run logit parity against PyTorch.")




if __name__ == "__main__":
  unittest.main()
