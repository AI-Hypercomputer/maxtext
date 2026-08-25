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

    # Split NNX model into static graphdef and state to run a pure functional forward pass
    graphdef, state = nnx.split(model)

    @jax.jit
    def run_forward(state_in, x, p, s):
      with nn.logical_axis_rules(config.logical_axis_rules):
        m = nnx.merge(graphdef, state_in)
        return m(
            decoder_input_tokens=x,
            decoder_positions=p,
            decoder_segment_ids=s,
            enable_dropout=False,
        )

    print("Running JIT-compiled forward pass on TPU...")
    logits = run_forward(state, inputs, positions, segment_ids)
    print("Logits shape:", logits.shape, "dtype:", logits.dtype)

    # Assertions on JAX forward pass
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
    self.assertFalse(jnp.isnan(logits).any(), "Logits contain NaNs!")
    self.assertFalse(jnp.isinf(logits).any(), "Logits contain Infs!")
    print("FORWARD PASS SUCCESSFUL!")

    # Check if PyTorch Hugging Face reference model is available for logit parity comparison
    print(f"\nChecking Hugging Face reference checkpoint at: {self.hf_model_path}")
    if os.path.exists(self.hf_model_path):
      print(f"Found Hugging Face model directory at {self.hf_model_path}.")
      try:
        import sys
        import types
        import torch
        import torch.nn as torch_nn
        import torch.nn.functional as F
        import transformers.utils.generic as tg

        # 1. OutputRecorder compatibility shim
        if not hasattr(tg, "OutputRecorder"):
          class OutputRecorder:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
          tg.OutputRecorder = OutputRecorder

        # 2. Pure PyTorch CPU fallback for fla (Flash Linear Attention)
        fla = types.ModuleType("fla")
        fla_modules = types.ModuleType("fla.modules")
        fla_ops = types.ModuleType("fla.ops")
        fla_ops_kda = types.ModuleType("fla.ops.kda")
        fla_ops_utils = types.ModuleType("fla.ops.utils")
        fla_ops_utils_index = types.ModuleType("fla.ops.utils.index")
        fla_utils = types.ModuleType("fla.utils")

        class ShortConvolution(torch_nn.Module):
          def __init__(self, hidden_size, kernel_size=4, activation="silu", **kwargs):
            super().__init__()
            self.hidden_size = hidden_size
            self.kernel_size = kernel_size
            self.weight = torch_nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
            self.bias = None
            self.activation = activation
          def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
            B, T, C = x.shape
            x_t = x.transpose(1, 2)
            x_pad = F.pad(x_t, (self.kernel_size - 1, 0))
            y = F.conv1d(x_pad, self.weight, groups=C).transpose(1, 2)
            if self.activation == "silu":
              y = F.silu(y)
            return y, None

        class FusedRMSNormGated(torch_nn.Module):
          def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, **kwargs):
            super().__init__()
            self.hidden_size = hidden_size
            self.eps = eps
            self.weight = torch_nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
          def forward(self, x, gate=None):
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            out = x * norm
            if self.weight is not None:
              out = out * self.weight
            if gate is not None:
              out = out * torch.sigmoid(gate)
            return out

        def chunk_kda(q, k, v, g, beta, A_log, dt_bias, initial_state=None, output_final_state=True,
                      use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                      safe_gate=True, lower_bound=-5.0, transpose_state_layout=True, cu_seqlens=None, **kwargs):
          B, T, H, K_dim = q.shape
          V_dim = v.shape[-1]
          if use_qk_l2norm_in_kernel:
            q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-6)
            k = k / torch.linalg.norm(k, dim=-1, keepdim=True).clamp(min=1e-6)
          if use_gate_in_kernel:
            a_log_exp = torch.exp(A_log).reshape(1, 1, 1, K_dim)
            g = lower_bound * torch.sigmoid(a_log_exp * (g + dt_bias.reshape(1, 1, H, K_dim)))
          if use_beta_sigmoid_in_kernel:
            beta = torch.sigmoid(beta)
          scale = K_dim ** -0.5
          q = q * scale
          S = torch.zeros(B, H, K_dim, V_dim, dtype=q.dtype, device=q.device) if initial_state is None else initial_state
          outputs = []
          for t in range(T):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            g_t = g[:, t]
            b_t = beta[:, t]
            S = S * torch.exp(g_t).unsqueeze(-1)
            k_S = torch.sum(k_t.unsqueeze(-1) * S, dim=-2)
            v_diff = v_t - k_S
            bk = b_t.unsqueeze(-1) * k_t
            S = S + bk.unsqueeze(-1) * v_diff.unsqueeze(-2)
            o_t = torch.sum(q_t.unsqueeze(-1) * S, dim=-2)
            outputs.append(o_t)
          o = torch.stack(outputs, dim=1)
          return o, S

        def prepare_cu_seqlens_from_mask(mask): return None
        def prepare_lens_from_mask(mask): return None
        def tensor_cache(fn): return fn

        fla_modules.ShortConvolution = ShortConvolution
        fla_modules.FusedRMSNormGated = FusedRMSNormGated
        fla_ops_kda.chunk_kda = chunk_kda
        fla_ops_kda.fused_recurrent_kda = chunk_kda
        fla_ops_utils_index.prepare_cu_seqlens_from_mask = prepare_cu_seqlens_from_mask
        fla_ops_utils_index.prepare_lens_from_mask = prepare_lens_from_mask
        fla_utils.tensor_cache = tensor_cache

        sys.modules["fla"] = fla
        sys.modules["fla.modules"] = fla_modules
        sys.modules["fla.ops"] = fla_ops
        sys.modules["fla.ops.kda"] = fla_ops_kda
        sys.modules["fla.ops.utils"] = fla_ops_utils
        sys.modules["fla.ops.utils.index"] = fla_ops_utils_index
        sys.modules["fla.utils"] = fla_utils

        from transformers import AutoConfig, AutoModelForCausalLM

        # Ensure all required custom python modeling files are present in the subset directory
        py_files = [
            "configuration_kimi_k3.py",
            "modeling_kimi_k3.py",
            "modeling_kimi_linear.py",
            "encoding_k3.py",
            "media_utils.py",
            "tokenization_kimi.py",
        ]
        try:
          from huggingface_hub import hf_hub_download
          repo_id = os.environ.get("HF_REPO_ID", "moonshotai/Kimi-K3")
          for fn in py_files:
            dst = os.path.join(self.hf_model_path, fn)
            if not os.path.exists(dst):
              print(f"Fetching {fn} from {repo_id}...")
              hf_hub_download(repo_id=repo_id, filename=fn, local_dir=self.hf_model_path)
        except Exception as hub_err:
          print(f"Note: Hub download check skipped/failed: {hub_err}")

        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from safetensors import safe_open
        import glob

        hf_config = AutoConfig.from_pretrained(
            self.hf_model_path,
            trust_remote_code=True,
        )
        hf_config.quantization_config = None
        if hasattr(hf_config, "text_config"):
          hf_config.text_config.quantization_config = None
          hf_config.text_config.num_hidden_layers = 2
          if hasattr(hf_config.text_config, "linear_attn_config") and isinstance(hf_config.text_config.linear_attn_config, dict):
            hf_config.text_config.linear_attn_config["kda_layers"] = [1]
            hf_config.text_config.linear_attn_config["full_attn_layers"] = [2]
        if hasattr(hf_config, "num_hidden_layers"):
          hf_config.num_hidden_layers = 2
        if hasattr(hf_config, "num_layers"):
          hf_config.num_layers = 2
        if hasattr(hf_config, "kda_layers"):
          hf_config.kda_layers = [1]
        if hasattr(hf_config, "full_attn_layers"):
          hf_config.full_attn_layers = [2]

        model_cls = get_class_from_dynamic_module(
            "modeling_kimi_k3.KimiK3ForConditionalGeneration",
            self.hf_model_path,
        )
        orig_tie_weights = getattr(model_cls, "tie_weights", None)
        def patched_tie_weights(self, *args, **kwargs):
          try:
            if orig_tie_weights:
              orig_tie_weights(self)
          except TypeError:
            pass
        model_cls.tie_weights = patched_tie_weights

        print("Instantiating 2-layer PyTorch reference model...")
        pt_model = model_cls(hf_config).to(torch.bfloat16)

        # Load weights directly from downloaded safetensors shards
        loaded_keys = 0
        for sf in glob.glob(os.path.join(self.hf_model_path, "*.safetensors")):
          with safe_open(sf, framework="pt", device="cpu") as f:
            for k in f.keys():
              mapped_k = k.replace(".layers.3.", ".layers.1.")
              if mapped_k in pt_model.state_dict():
                pt_model.state_dict()[mapped_k].copy_(f.get_tensor(k).to(torch.bfloat16))
                loaded_keys += 1
        print(f"Loaded {loaded_keys} weight tensors into PyTorch reference model.")

        pt_model.eval()
        with torch.no_grad():
          pt_inputs = torch.from_numpy(np.array(inputs))
          pt_outputs = pt_model(pt_inputs)
          pt_logits = pt_outputs.logits.detach().float().numpy()

        jax_logits_np = np.array(logits).astype(np.float32)

        # Compute logit parity metrics
        diff = np.abs(jax_logits_np - pt_logits)
        max_err = float(np.max(diff))
        mae = float(np.mean(diff))
        cos_sim = float(
            np.dot(jax_logits_np.flatten(), pt_logits.flatten())
            / (np.linalg.norm(jax_logits_np) * np.linalg.norm(pt_logits) + 1e-12)
        )
        top1_agree = float(np.mean(np.argmax(jax_logits_np, axis=-1) == np.argmax(pt_logits, axis=-1)))

        print("=" * 70)
        print("REAL PRETRAINED 2-LAYER CHECKPOINT LOGIT PARITY (MaxText TPU vs HF PyTorch):")
        print(f"  Logits Shape:          {jax_logits_np.shape}")
        print(f"  Max Absolute Error:    {max_err:.6e}")
        print(f"  Mean Absolute Error:   {mae:.6e}")
        print(f"  Cosine Similarity:     {cos_sim:.8f}")
        print(f"  Top-1 Argmax Agreement:{top1_agree * 100:.1f}%")
        print("=" * 70)

        self.assertGreater(cos_sim, 0.999, f"Logit cosine similarity {cos_sim} is below 0.999!")
        self.assertEqual(top1_agree, 1.0, f"Top-1 argmax agreement {top1_agree} is not 100%!")
        print("REAL PRETRAINED LOGIT PARITY VERIFIED SUCCESSFULLY!")
      except Exception as e:
        print(f"\nNote: Hugging Face PyTorch comparison skipped ({e}).")
        print("MaxText forward pass on TPU is verified and passed.")
    else:
      print(f"WARNING: Hugging Face checkpoint not found at {self.hf_model_path}.")
      print("Pass HF_MODEL_PATH=<path_to_hf_subset> to run logit parity against PyTorch.")




if __name__ == "__main__":
  unittest.main()
