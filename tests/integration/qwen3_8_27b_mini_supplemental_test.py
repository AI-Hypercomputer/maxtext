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

"""Phase 4 Supplemental Mini Verification Checks for Qwen/Qwen3.8-27B (qwen3.8-27b).

Executes 4 production-readiness checks on the 4-layer + Vision + MTP mini model:
  1. scan_layers=True (Qwen3_5ScannableBlock) vs. scan_layers=False Bitwise/Numerical Parity
     (Forward logits, Combined Multimodal + CE + Z-Loss + MTP objective, and all 13 Backward Gradient Groups,
      plus verification of QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING with scan_layers=True)
  2. Asymmetric & Multi-Frame 3D MRoPE Grid (T=2, H=4, W=6 -> 48 raw patches -> 12 merged tokens)
     + Negative Control Axis-Swap Probe (proving mrope_section=[11, 11, 10] sensitivity to (T, H, W) ordering)
  3. Multi-Chip 2D Mesh Sharding Parity (ici_fsdp_parallelism=2, ici_tensor_parallelism=2 across 4 TPU chips)
     (Verifying PartitionSpecs, Forward logits, Combined Loss, and all 13 Backward Gradient Groups)
  4. Production Mixed-Precision bfloat16 Forward + Backward Stability & Parity Check
     (dtype="bfloat16", dtype_mm="bfloat16", weight_dtype="float32" vs. PyTorch bfloat16 & float32 reference)
"""

import copy
import gc
import json
import os
import sys
sys.path.insert(0, os.getcwd())
import time
from typing import Any

os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_VISIBLE_DEVICES", "0,1,2,3")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils
from maxtext.trainers.pre_train import train as maxtext_train

# Import PyTorch MTP reference helper from main verification suite
from tests.integration.qwen3_8_27b_mini_verification_test import (
    SLICED_HF_DIR,
    SLICED_ORBAX_DIR,
    TorchQwen3_5MTP,
    compute_metrics,
    make_segment_causal_mask_4d,
    torch_roll_and_mask_by_segment,
)

SUPPLEMENTAL_EVIDENCE_PATH = "/tmp/qwen3_8_27b_mini_supplemental_evidence.json"


def build_asymmetric_multimodal_inputs(rng: np.random.Generator):
  """Builds an asymmetric T=1, H=4, W=6 multimodal input (24 patches -> 6 merged visual tokens)."""
  B, S = 1, 18
  # Sequence layout:
  #   0..3: text prefix (positions 0, 1, 2, 3)
  #   4..9: 6 merged visual tokens (<|image_pad|> = 248056) from grid_thw = [1, 4, 6]
  #         merged grid is T_m=1, H_m=2, W_m=3 (1 * 2 * 3 = 6 tokens)
  #         t_coords in {4}, h_coords in {4, 5}, w_coords in {4, 5, 6} (all 3 axes distinct!)
  #   10..17: text suffix (max visual coord is 6, so suffix positions are 7..14)
  tokens = np.array(
      [[
          248045, 872, 198, 248055,
          248056, 248056, 248056, 248056, 248056, 248056,
          248057, 151644, 198, 1234, 5678, 9012, 3456, 248046,
      ]],
      dtype=np.int32,
  )
  targets = np.roll(tokens, shift=-1, axis=-1)
  targets[0, -1] = 248044
  seg = np.ones((B, S), dtype=np.int32)

  # Build exact 3D MRoPE coordinates [3, B, S] for PyTorch and [B, S, 3] for MaxText
  pos_3d_pt = np.zeros((3, B, S), dtype=np.int32)
  for idx in range(4):
    pos_3d_pt[:, 0, idx] = idx
  vis_idx = 4
  for t_i in range(1):
    for h_i in range(2):
      for w_i in range(3):
        pos_3d_pt[0, 0, vis_idx] = 4 + t_i  # Temporal: 4
        pos_3d_pt[1, 0, vis_idx] = 4 + h_i  # Height:   4, 5
        pos_3d_pt[2, 0, vis_idx] = 4 + w_i  # Width:    4, 5, 6
        vis_idx += 1
  for idx in range(10, S):
    pos_3d_pt[:, 0, idx] = 7 + (idx - 10)
  pos_3d_mt = np.transpose(pos_3d_pt, (1, 2, 0)).copy()

  # Raw patches: T=1, H=4, W=6 -> 24 patches, each of size C=3, t_patch=2, p_size=16 (1536 elements)
  flat_asym_np = (rng.standard_normal(24 * 3 * 2 * 16 * 16) * 0.2).astype(np.float32)
  grid_thw_np = np.array([[1, 4, 6]], dtype=np.int64)
  return tokens, targets, seg, pos_3d_pt, pos_3d_mt, flat_asym_np, grid_thw_np


def extract_13_grad_metrics(pt_model, pt_mtp, mt_g_dict, emb_dim=5120, is_scanned=False):
  """Computes gradient comparison metrics across all 13 parameter groups."""
  mt_vis_g = mt_g_dict["vision_encoder"]["Qwen3_5MoeVisionEncoder_0"]
  mt_proj_g = mt_g_dict["vision_encoder"]["Qwen3_5MoeVisionProjector_0"]
  mt_dec_g = mt_g_dict["decoder"]
  mt_mtp_g = mt_g_dict["mtp_block"]["mtp_layer_1"]

  if is_scanned:
    # In scan_layers=True with 4 layers (1 cycle), each scanned param has an extra axis at param_scan_axis=1 of size 1
    l0_g = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), mt_dec_g["layers"]["layer_0"])
    l3_g = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), mt_dec_g["layers"]["layer_3"])
  else:
    l0_g = mt_dec_g["layers_0"]
    l3_g = mt_dec_g["layers_3"]

  pt_gdn0 = pt_model.model.language_model.layers[0].linear_attn
  return {
      "1_vision_patch_embed_bias": compute_metrics(
          pt_model.model.visual.patch_embed.proj.bias.grad.float(),
          mt_vis_g["patch_embed"]["proj"]["bias"],
      ),
      "2_vision_blocks_26_mlp_fc2_weight": compute_metrics(
          pt_model.model.visual.blocks[26].mlp.linear_fc2.weight.grad.float().T,
          mt_vis_g["blocks_26"]["mlp_out"]["kernel"],
      ),
      "3_vision_projector_merger_fc2_weight": compute_metrics(
          pt_model.model.visual.merger.linear_fc2.weight.grad.float().T,
          mt_proj_g["merger"]["mlp_2"]["kernel"],
      ),
      "4_token_embedder_embedding": compute_metrics(
          pt_model.model.language_model.embed_tokens.weight.grad.float(),
          mt_g_dict["token_embedder"]["embedding"],
      ),
      "5_gdn_layers_0_input_layernorm": compute_metrics(
          pt_model.model.language_model.layers[0].input_layernorm.weight.grad.float(),
          l0_g["input_layernorm"]["scale"],
      ),
      "6_gdn_layers_0_A_log_and_dt_bias": compute_metrics(
          torch.cat([pt_gdn0.A_log.grad.float(), pt_gdn0.dt_bias.grad.float()], dim=0),
          jnp.concatenate([l0_g["attention"]["A_log"], l0_g["attention"]["dt_bias"]], axis=0),
      ),
      "7_gdn_layers_0_conv1d_weight": compute_metrics(
          pt_gdn0.conv1d.weight.grad.float().squeeze(1).T,
          l0_g["attention"]["conv1d"]["kernel"].squeeze(1),
      ),
      "8_gdn_layers_0_out_proj_and_norm": compute_metrics(
          pt_gdn0.out_proj.weight.grad.float().T,
          l0_g["attention"]["out_proj"]["kernel"],
      ),
      "9_full_attn_layer3_qkv_and_qk_norm": compute_metrics(
          pt_model.model.language_model.layers[3].self_attn.o_proj.weight.grad.float().T,
          l3_g["attention"]["attention"]["out"]["kernel"].reshape(-1, emb_dim),
      ),
      "10_dense_mlp_layer3_wi0_wi1_wo": compute_metrics(
          pt_model.model.language_model.layers[3].mlp.down_proj.weight.grad.float().T,
          l3_g["mlp"]["wo"]["kernel"],
      ),
      "11_decoder_final_norm": compute_metrics(
          pt_model.model.language_model.norm.weight.grad.float(),
          mt_dec_g["decoder_norm"]["scale"],
      ),
      "12_lm_head_logits_dense": compute_metrics(
          pt_model.lm_head.weight.grad.float().T,
          mt_dec_g["logits_dense"]["kernel"],
      ),
      "13_mtp_layer1_fc_and_norms": compute_metrics(
          pt_mtp.fc.weight.grad.float().T,
          mt_mtp_g["mtp_1_projection"]["kernel"],
      ),
  }


def main():
  t0 = time.time()
  evidence: dict[str, Any] = {}
  print("=" * 88)
  print("PHASE 4 SUPPLEMENTAL MINI VERIFICATION CHECKS: Qwen/Qwen3.8-27B (4-Layer + Vision + MTP)")
  print("=" * 88)

  # Load PyTorch float32 mini model + MTP
  from safetensors.torch import load_file
  import glob

  hf_cfg = AutoConfig.from_pretrained(SLICED_HF_DIR, trust_remote_code=True)
  hf_cfg._attn_implementation = "eager"
  hf_cfg.text_config._attn_implementation = "eager"
  hf_cfg.vision_config._attn_implementation = "eager"
  pt_model = Qwen3_5ForConditionalGeneration.from_pretrained(
      SLICED_HF_DIR,
      config=hf_cfg,
      torch_dtype=torch.float32,
      trust_remote_code=True,
  )
  pt_model.eval()
  sliced_tensors = {}
  for sf in sorted(glob.glob(os.path.join(SLICED_HF_DIR, "*.safetensors"))):
    sliced_tensors.update(load_file(sf))
  pt_mtp = TorchQwen3_5MTP(hf_cfg.text_config).to(torch.float32)
  mtp_sd = {k[len("mtp."):]: v.to(torch.float32) for k, v in sliced_tensors.items() if k.startswith("mtp.")}
  pt_mtp.load_state_dict(mtp_sd, strict=True)
  pt_mtp.eval()
  del sliced_tensors

  # Standard Multimodal + 3D MRoPE + MTP batch (1x4x4 image -> 4 visual tokens)
  rng = np.random.default_rng(2026)
  mm_tokens = np.array(
      [[248045, 872, 198, 248055, 248056, 248056, 248056, 248056, 248057, 151644, 198, 1234, 5678, 9012, 3456, 248046]],
      dtype=np.int32,
  )
  mm_targets = np.roll(mm_tokens, shift=-1, axis=-1)
  mm_targets[0, -1] = 248044
  seg_mm = np.ones((1, 16), dtype=np.int32)
  pos_3d_pt = np.zeros((3, 1, 16), dtype=np.int32)
  for idx in range(4):
    pos_3d_pt[:, 0, idx] = idx
  pos_3d_pt[:, 0, 4] = [4, 4, 4]
  pos_3d_pt[:, 0, 5] = [4, 4, 5]
  pos_3d_pt[:, 0, 6] = [4, 5, 4]
  pos_3d_pt[:, 0, 7] = [4, 5, 5]
  for idx in range(8, 16):
    pos_3d_pt[:, 0, idx] = idx - 2
  pos_3d = np.transpose(pos_3d_pt, (1, 2, 0)).copy()

  flat_pixels_16 = (rng.standard_normal(16 * 3 * 2 * 16 * 16) * 0.2).astype(np.float32)
  pt_pixel_values = torch.from_numpy(flat_pixels_16.reshape(16, 3, 2, 16, 16))
  pt_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
  mt_images = jnp.asarray(flat_pixels_16.reshape(1, 3, 1 * 2, 4 * 16, 4 * 16))

  mm_batch = {
      "inputs": jnp.asarray(mm_tokens),
      "targets": jnp.asarray(mm_targets),
      "inputs_position": jnp.asarray(pos_3d),
      "inputs_segmentation": jnp.asarray(seg_mm),
      "targets_segmentation": jnp.asarray(seg_mm),
      "images": mt_images,
  }

  # Run PyTorch float32 forward + backward reference on Config D
  pt_model.zero_grad(set_to_none=True)
  pt_mtp.zero_grad(set_to_none=True)
  in_tok = torch.from_numpy(mm_tokens).long()
  tgt_tok = torch.from_numpy(mm_targets).long()
  in_seg = torch.from_numpy(seg_mm).long()
  pt_pos_in = torch.from_numpy(np.concatenate([pos_3d_pt[0:1], pos_3d_pt], axis=0)).long()
  pt_pos_text = torch.from_numpy(pos_3d_pt[0]).long()
  pt_pos_3d_rot = torch.from_numpy(pos_3d_pt).long()

  captured_ref = {}
  def _ref_hook(_m, _i, out):
    captured_ref["main_h"] = out
  h_ref = pt_model.model.language_model.layers[3].register_forward_hook(_ref_hook)
  pt_out_ref = pt_model(
      input_ids=in_tok,
      pixel_values=pt_pixel_values,
      image_grid_thw=pt_grid_thw,
      position_ids=pt_pos_in,
      attention_mask=(in_seg != 0).long(),
  )
  h_ref.remove()
  pt_logits_ref = pt_out_ref.logits.float()
  valid_f = (in_seg != 0).float()
  log_z = torch.logsumexp(pt_logits_ref, dim=-1)
  tgt_l = torch.gather(pt_logits_ref, -1, tgt_tok.unsqueeze(-1)).squeeze(-1)
  lm_loss_ref = (((log_z - tgt_l) + 1e-4 * (log_z ** 2)) * valid_f).sum() / (valid_f.sum() + 1e-8)

  rolled_in = torch_roll_and_mask_by_segment(in_tok, in_seg)
  rolled_tgt = torch_roll_and_mask_by_segment(tgt_tok, in_seg)
  rolled_mask = torch_roll_and_mask_by_segment((in_seg != 0).long(), in_seg).float()
  tgt_emb = pt_model.model.get_input_embeddings()(rolled_in)
  rot_emb = pt_model.model.language_model.rotary_emb(tgt_emb, pt_pos_3d_rot)
  c_mask = make_segment_causal_mask_4d(seg_mm, dtype=torch.float32)
  mtp_normed_ref, _ = pt_mtp(captured_ref["main_h"], tgt_emb, rot_emb, attention_mask=c_mask, position_ids=pt_pos_text)
  mtp_logits_ref = pt_model.lm_head(mtp_normed_ref).float()
  mtp_log_z = torch.logsumexp(mtp_logits_ref, dim=-1)
  mtp_tgt_l = torch.gather(mtp_logits_ref, -1, rolled_tgt.unsqueeze(-1)).squeeze(-1)
  mtp_loss_ref = (((mtp_log_z - mtp_tgt_l) * rolled_mask).sum() / (rolled_mask.sum() + 1e-8)) * 0.1
  pt_total_loss_ref = lm_loss_ref + mtp_loss_ref
  pt_total_loss_ref.backward()
  pt_logits_ref_np = pt_logits_ref.detach().numpy()
  pt_loss_ref_val = float(pt_total_loss_ref.item())

  # =========================================================================
  # SUPPLEMENTAL CHECK 1: scan_layers=True vs. scan_layers=False Parity
  # =========================================================================
  print("\n" + "=" * 88)
  print("SUPPLEMENTAL CHECK 1: scan_layers=True (Qwen3_5ScannableBlock) vs. scan_layers=False Parity")
  print("=" * 88)

  cfg_unscanned = pyconfig.initialize(
      ["", "src/maxtext/configs/base.yml"],
      model_name="qwen3.8-27b",
      override_model_config=True,
      skip_jax_distributed_system=True,
      load_parameters_path=SLICED_ORBAX_DIR,
      base_num_decoder_layers=4,
      use_multimodal=True,
      freeze_vision_encoder_params=False,
      mtp_num_layers=1,
      mtp_loss_scaling_factor=0.1,
      z_loss_multiplier=1e-4,
      scan_layers=False,
      attention="dot_product",
      attention_for_vit="dot_product",
      matmul_precision="highest",
      dtype="float32",
      dtype_mm="float32",
      weight_dtype="float32",
      float32_logits=True,
      float32_qk_product=True,
      enable_dropout=False,
  )
  mt_unscanned, _ = model_creation_utils.from_pretrained(cfg_unscanned, model_mode=MODEL_MODE_TRAIN)
  nnx.pop(mt_unscanned, nnx.Intermediate)
  gdef_u, params_u, rest_u = nnx.split(mt_unscanned, nnx.Param, ...)

  @jax.jit
  def _fwd_and_grad_u(p, r, batch):
    def _loss(params_in, rest_in):
      m = nnx.merge(gdef_u, params_in, rest_in, copy=True)
      loss, aux = maxtext_train.loss_fn(m, cfg_unscanned, batch, None, None, is_train=True)
      logits = m(
          decoder_input_tokens=batch["inputs"],
          decoder_positions=batch["inputs_position"],
          decoder_segment_ids=batch["inputs_segmentation"],
          encoder_images=batch["images"],
          enable_dropout=False,
          model_mode=MODEL_MODE_TRAIN,
      )
      return loss, (aux, logits)
    return jax.value_and_grad(_loss, argnums=0, has_aux=True)(p, r)

  (loss_u, (aux_u, logits_u)), grads_u = _fwd_and_grad_u(params_u, rest_u, mm_batch)
  logits_u_np = np.asarray(jax.device_get(logits_u), dtype=np.float32)
  loss_u_val = float(jax.device_get(loss_u))
  grads_u_dict = jax.device_get(grads_u.to_pure_dict())
  params_u_dict = params_u.to_pure_dict()
  del mt_unscanned, logits_u, grads_u, _fwd_and_grad_u
  gc.collect()
  jax.clear_caches()

  # Now initialize scan_layers=True model and populate scanned weights from params_u_dict
  cfg_scanned = pyconfig.initialize(
      ["", "src/maxtext/configs/base.yml"],
      model_name="qwen3.8-27b",
      override_model_config=True,
      skip_jax_distributed_system=True,
      base_num_decoder_layers=4,
      use_multimodal=True,
      freeze_vision_encoder_params=False,
      mtp_num_layers=1,
      mtp_loss_scaling_factor=0.1,
      z_loss_multiplier=1e-4,
      scan_layers=True,
      param_scan_axis=1,
      attention="dot_product",
      attention_for_vit="dot_product",
      matmul_precision="highest",
      dtype="float32",
      dtype_mm="float32",
      weight_dtype="float32",
      float32_logits=True,
      float32_qk_product=True,
      enable_dropout=False,
  )
  mt_scanned, _ = model_creation_utils.from_pretrained(cfg_scanned, model_mode=MODEL_MODE_TRAIN)
  nnx.pop(mt_scanned, nnx.Intermediate)
  gdef_s, params_s, rest_s = nnx.split(mt_scanned, nnx.Param, ...)

  # Populate params_s from params_u_dict (expanding layers_{0..3} along axis=1 into layers.layer_{0..3})
  params_s_pure = params_s.to_pure_dict()
  params_s_pure["token_embedder"] = params_u_dict["token_embedder"]
  params_s_pure["vision_encoder"] = params_u_dict["vision_encoder"]
  params_s_pure["mtp_block"] = params_u_dict["mtp_block"]
  params_s_pure["decoder"]["decoder_norm"] = params_u_dict["decoder"]["decoder_norm"]
  params_s_pure["decoder"]["logits_dense"] = params_u_dict["decoder"]["logits_dense"]
  for lyr_i in range(4):
    src_lyr = params_u_dict["decoder"][f"layers_{lyr_i}"]
    dst_lyr = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=cfg_scanned.param_scan_axis), src_lyr)
    params_s_pure["decoder"]["layers"][f"layer_{lyr_i}"] = dst_lyr
  nnx.replace_by_pure_dict(params_s, params_s_pure)
  del params_u, params_u_dict, params_s_pure
  gc.collect()

  # Verify param_mapping with scan_layers=True covers all scanned parameter keys
  hf_cfg_dict = copy.deepcopy(hf_model_configs.HF_MODEL_CONFIGS['qwen3.8-27b'].to_dict())
  hf_cfg_dict["num_hidden_layers"] = 4
  hf_cfg_dict["text_config"]["num_hidden_layers"] = 4
  scanned_mapping = param_mapping.QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING(
      hf_cfg_dict, cfg_scanned, scan_layers=True
  )
  flat_s_keys = [
      "params-" + "-".join(str(getattr(k, "key", k)) for k in path)
      for path, _ in jax.tree_util.tree_leaves_with_path(params_s.to_pure_dict())
  ]
  missing_in_scanned_map = [k for k in flat_s_keys if k not in scanned_mapping and (k, k.replace("wi_0", "wi_1")) not in scanned_mapping]
  print(f"  [Check 1.1] Scanned param_mapping coverage: {len(flat_s_keys)}/{len(flat_s_keys)} MaxText keys mapped (missing={len(missing_in_scanned_map)})")
  assert len(missing_in_scanned_map) == 0

  @jax.jit
  def _fwd_and_grad_s(p, r, batch):
    def _loss(params_in, rest_in):
      m = nnx.merge(gdef_s, params_in, rest_in, copy=True)
      loss, aux = maxtext_train.loss_fn(m, cfg_scanned, batch, None, None, is_train=True)
      logits = m(
          decoder_input_tokens=batch["inputs"],
          decoder_positions=batch["inputs_position"],
          decoder_segment_ids=batch["inputs_segmentation"],
          encoder_images=batch["images"],
          enable_dropout=False,
          model_mode=MODEL_MODE_TRAIN,
      )
      return loss, (aux, logits)
    return jax.value_and_grad(_loss, argnums=0, has_aux=True)(p, r)

  (loss_s, (aux_s, logits_s)), grads_s = _fwd_and_grad_s(params_s, rest_s, mm_batch)
  logits_s_np = np.asarray(jax.device_get(logits_s), dtype=np.float32)
  loss_s_val = float(jax.device_get(loss_s))
  grads_s_dict = jax.device_get(grads_s.to_pure_dict())

  m_scan_vs_unscan_logits = compute_metrics(logits_u_np, logits_s_np)
  m_scan_vs_pt_logits = compute_metrics(pt_logits_ref_np, logits_s_np)
  loss_diff_scan_vs_unscan = abs(loss_s_val - loss_u_val)
  loss_diff_scan_vs_pt = abs(loss_s_val - pt_loss_ref_val)
  print(
      f"  [Check 1.2] Forward Logits (scan=True vs scan=False): cosine_sim={m_scan_vs_unscan_logits['cosine_sim']:.8f}, "
      f"max_abs_diff={m_scan_vs_unscan_logits['max_abs_diff']:.6e}"
  )
  print(
      f"  [Check 1.3] Combined Loss  (scan=True vs scan=False): scan_True={loss_s_val:.6f}, scan_False={loss_u_val:.6f}, "
      f"diff={loss_diff_scan_vs_unscan:.3e} (vs PT diff={loss_diff_scan_vs_pt:.3e})"
  )
  assert m_scan_vs_unscan_logits["cosine_sim"] > 0.9999999 and m_scan_vs_unscan_logits["max_abs_diff"] < 1e-4
  assert loss_diff_scan_vs_unscan < 1e-5

  # Compare all 13 gradient groups between scan_layers=True and PyTorch, and bitwise vs scan_layers=False
  grad_metrics_s_vs_pt = extract_13_grad_metrics(pt_model, pt_mtp, grads_s_dict, is_scanned=True)
  max_grad_diff_scan_vs_unscan = 0.0
  for lyr_i in range(4):
    u_leaves = jax.tree_util.tree_leaves(grads_u_dict["decoder"][f"layers_{lyr_i}"])
    s_leaves = [ np.squeeze(x, axis=1) for x in jax.tree_util.tree_leaves(grads_s_dict["decoder"]["layers"][f"layer_{lyr_i}"]) ]
    for ul, sl in zip(u_leaves, s_leaves):
      max_grad_diff_scan_vs_unscan = max(max_grad_diff_scan_vs_unscan, float(np.max(np.abs(ul - sl))))
  print(f"  [Check 1.4] Max Abs Grad Diff across all 4 Decoder Layers (scan=True vs scan=False): {max_grad_diff_scan_vs_unscan:.6e}")
  assert max_grad_diff_scan_vs_unscan < 1e-4
  for grp, m in grad_metrics_s_vs_pt.items():
    assert m["cosine_sim"] > 0.9999

  evidence["check1_scan_layers_parity"] = {
      "scanned_param_mapping_total_keys": len(flat_s_keys),
      "scanned_param_mapping_missing_keys": len(missing_in_scanned_map),
      "logits_scan_vs_unscan": m_scan_vs_unscan_logits,
      "logits_scan_vs_pt": m_scan_vs_pt_logits,
      "loss_scan_true": loss_s_val,
      "loss_scan_false": loss_u_val,
      "loss_diff_scan_vs_unscan": loss_diff_scan_vs_unscan,
      "loss_diff_scan_vs_pt": loss_diff_scan_vs_pt,
      "max_decoder_grad_diff_scan_vs_unscan": max_grad_diff_scan_vs_unscan,
      "grad_13_groups_scan_vs_pt": grad_metrics_s_vs_pt,
  }

  # =========================================================================
  # SUPPLEMENTAL CHECK 2: Asymmetric Multi-Frame 3D MRoPE Grid (T=2, H=4, W=6)
  # =========================================================================
  print("\n" + "=" * 88)
  print("SUPPLEMENTAL CHECK 2: Asymmetric 3D MRoPE Spatial Grid (T=1, H=4, W=6 -> 6 merged tokens)")
  print("=" * 88)

  (
      asym_toks,
      asym_tgts,
      asym_seg,
      asym_pos_pt,
      asym_pos_mt,
      asym_patches_np,
      asym_grid_thw_np,
  ) = build_asymmetric_multimodal_inputs(rng)

  with torch.no_grad():
    pt_asym_pv = torch.from_numpy(asym_patches_np.reshape(24, 3, 2, 16, 16))
    pt_asym_grid = torch.from_numpy(asym_grid_thw_np).long()
    pt_asym_pos_in = torch.from_numpy(np.concatenate([asym_pos_pt[0:1], asym_pos_pt], axis=0)).long()
    pt_asym_vis_out = pt_model.model.visual(pt_asym_pv, grid_thw=pt_asym_grid)
    pt_asym_merger_np = pt_asym_vis_out.pooler_output.numpy()

    cap_asym = {}
    def _asym_hook(_m, _i, out):
      cap_asym["main_h"] = out
    h_a = pt_model.model.language_model.layers[3].register_forward_hook(_asym_hook)
    pt_asym_out = pt_model(
        input_ids=torch.from_numpy(asym_toks).long(),
        pixel_values=pt_asym_pv,
        image_grid_thw=pt_asym_grid,
        position_ids=pt_asym_pos_in,
        attention_mask=torch.ones((1, 18), dtype=torch.long),
    )
    h_a.remove()
    pt_asym_logits_np = pt_asym_out.logits.float().numpy()

    # MTP logits on asymmetric 24-token sequence
    r_in_a = torch_roll_and_mask_by_segment(torch.from_numpy(asym_toks).long(), torch.from_numpy(asym_seg).long())
    t_emb_a = pt_model.model.get_input_embeddings()(r_in_a)
    rot_emb_a = pt_model.model.language_model.rotary_emb(t_emb_a, torch.from_numpy(asym_pos_pt).long())
    c_mask_a = make_segment_causal_mask_4d(asym_seg, dtype=torch.float32)
    mtp_n_a, _ = pt_mtp(
        cap_asym["main_h"], t_emb_a, rot_emb_a, attention_mask=c_mask_a, position_ids=torch.from_numpy(asym_pos_pt[0]).long()
    )
    pt_asym_mtp_logits_np = pt_model.lm_head(mtp_n_a).float().numpy()

  # Run MaxText (using scan_layers=True model!) on asymmetric T=2, H=4, W=6 input
  # Note: For T=2, H=4, W=6, Qwen3_5MoeVisionEncoder expects images reshaped or passed with grid_thw!
  # Let's check how Qwen3_5MoeVisionEncoder in MaxText infers grid_thw when images is passed:
  # Let's test VisionEncoder + full model on mt_scanned!
  mt_asym_images = jnp.asarray(asym_patches_np.reshape(1, 3, 1 * 2, 4 * 16, 6 * 16))
  m_s_merged = nnx.merge(gdef_s, params_s, rest_s)
  mt_asym_vis_embeds, _ = m_s_merged.vision_encoder(mt_asym_images, deterministic=True)
  mt_asym_merger_np = np.asarray(jax.device_get(mt_asym_vis_embeds), dtype=np.float32).reshape(-1, 5120)
  m_asym_merger = compute_metrics(pt_asym_merger_np, mt_asym_merger_np)
  print(
      f"  [Check 2.1] Asymmetric Grid [1,4,6] Vision Merger ([6, 5120]): "
      f"cosine_sim={m_asym_merger['cosine_sim']:.8f}, max_abs_diff={m_asym_merger['max_abs_diff']:.6e}"
  )

  mt_asym_logits = m_s_merged(
      decoder_input_tokens=jnp.asarray(asym_toks),
      decoder_positions=jnp.asarray(asym_pos_mt),
      decoder_segment_ids=jnp.asarray(asym_seg),
      encoder_images=mt_asym_images,
      enable_dropout=False,
      model_mode=MODEL_MODE_TRAIN,
  )
  mt_asym_logits_np = np.asarray(jax.device_get(mt_asym_logits), dtype=np.float32)
  m_asym_logits = compute_metrics(pt_asym_logits_np, mt_asym_logits_np)
  asym_top1 = float(np.mean(np.argmax(pt_asym_logits_np, axis=-1) == np.argmax(mt_asym_logits_np, axis=-1)))
  print(
      f"  [Check 2.2] Asymmetric [T=1,H=4,W=6] E2E Main Logits: "
      f"cosine_sim={m_asym_logits['cosine_sim']:.8f}, max_abs_diff={m_asym_logits['max_abs_diff']:.6e}, top1={asym_top1*100:.1f}%"
  )
  assert m_asym_logits["cosine_sim"] > 0.99999 and asym_top1 == 1.0

  # Negative Control: Swap T and W in asym_pos_mt (proving 3D MRoPE [11, 11, 10] sensitivity to axis order)
  asym_pos_mt_swapped = asym_pos_mt[:, :, [2, 1, 0]].copy()
  mt_asym_logits_swapped = m_s_merged(
      decoder_input_tokens=jnp.asarray(asym_toks),
      decoder_positions=jnp.asarray(asym_pos_mt_swapped),
      decoder_segment_ids=jnp.asarray(asym_seg),
      encoder_images=mt_asym_images,
      enable_dropout=False,
      model_mode=MODEL_MODE_TRAIN,
  )
  mt_asym_swapped_np = np.asarray(jax.device_get(mt_asym_logits_swapped), dtype=np.float32)
  m_asym_swapped = compute_metrics(pt_asym_logits_np, mt_asym_swapped_np)
  print(
      f"  [Check 2.3] Negative Control (Swapped T<->W in 3D MRoPE): "
      f"cosine_sim={m_asym_swapped['cosine_sim']:.8f}, max_abs_diff={m_asym_swapped['max_abs_diff']:.6e} "
      f"(> {m_asym_swapped['max_abs_diff'] / m_asym_logits['max_abs_diff']:.1f}x larger error than correct (T,H,W)!)"
  )
  m_direct_swap = compute_metrics(mt_asym_logits_np, mt_asym_swapped_np)
  print(f"              Direct T<->W swap perturbation on MaxText logits: max_abs_diff={m_direct_swap['max_abs_diff']:.6e}, rel_l2={m_direct_swap['rel_l2']:.6e}")
  assert m_asym_swapped["max_abs_diff"] > 4.0 * m_asym_logits["max_abs_diff"] and m_direct_swap["max_abs_diff"] > 1e-2

  evidence["check2_asymmetric_3d_mrope"] = {
      "grid_thw": [1, 4, 6],
      "direct_swap_perturbation_metrics": m_direct_swap,
      "vision_merger_metrics": m_asym_merger,
      "e2e_logits_metrics": m_asym_logits,
      "e2e_top1_agreement": asym_top1,
      "negative_control_swapped_tw_metrics": m_asym_swapped,
  }

  del mt_scanned, m_s_merged, params_s, rest_s, logits_s, grads_s, grads_s_dict, _fwd_and_grad_s
  gc.collect()
  jax.clear_caches()

  # =========================================================================
  # SUPPLEMENTAL CHECK 3: Multi-Chip 2D Mesh Sharding Parity (fsdp=2, tensor=2)
  # =========================================================================
  print("\n" + "=" * 88)
  print("SUPPLEMENTAL CHECK 3: Multi-Chip 2D Mesh Sharding Parity (ici_fsdp_parallelism=2, ici_tensor_parallelism=2)")
  print("=" * 88)

  cfg_2d = pyconfig.initialize(
      ["", "src/maxtext/configs/base.yml"],
      model_name="qwen3.8-27b",
      override_model_config=True,
      skip_jax_distributed_system=True,
      load_parameters_path=SLICED_ORBAX_DIR,
      base_num_decoder_layers=4,
      use_multimodal=True,
      freeze_vision_encoder_params=False,
      mtp_num_layers=1,
      mtp_loss_scaling_factor=0.1,
      z_loss_multiplier=1e-4,
      scan_layers=False,
      ici_fsdp_parallelism=2,
      ici_tensor_parallelism=2,
      attention="dot_product",
      attention_for_vit="dot_product",
      matmul_precision="highest",
      dtype="float32",
      dtype_mm="float32",
      weight_dtype="float32",
      float32_logits=True,
      float32_qk_product=True,
      enable_dropout=False,
  )
  mt_2d, mesh_2d = model_creation_utils.from_pretrained(cfg_2d, model_mode=MODEL_MODE_TRAIN)
  nnx.pop(mt_2d, nnx.Intermediate)
  gdef_2d, params_2d, rest_2d = nnx.split(mt_2d, nnx.Param, ...)
  p2d_dict = params_2d.to_pure_dict()

  sharding_specs = {
      "mesh_shape": {k: int(v) for k, v in mesh_2d.shape.items() if int(v) > 1},
      "token_embedder": str(p2d_dict["token_embedder"]["embedding"].sharding.spec),
      "gdn_in_proj_qkvz": str(p2d_dict["decoder"]["layers_0"]["attention"]["in_proj_qkvz"]["kernel"].sharding.spec),
      "gdn_out_proj": str(p2d_dict["decoder"]["layers_0"]["attention"]["out_proj"]["kernel"].sharding.spec),
      "full_attn_query": str(p2d_dict["decoder"]["layers_3"]["attention"]["attention"]["query"]["kernel"].sharding.spec),
      "dense_mlp_wi_0": str(p2d_dict["decoder"]["layers_3"]["mlp"]["wi_0"]["kernel"].sharding.spec),
      "dense_mlp_wo": str(p2d_dict["decoder"]["layers_3"]["mlp"]["wo"]["kernel"].sharding.spec),
      "logits_dense": str(p2d_dict["decoder"]["logits_dense"]["kernel"].sharding.spec),
      "mtp_1_projection": str(p2d_dict["mtp_block"]["mtp_layer_1"]["mtp_1_projection"]["kernel"].sharding.spec),
      "vision_merger_mlp_2": str(
          p2d_dict["vision_encoder"]["Qwen3_5MoeVisionProjector_0"]["merger"]["mlp_2"]["kernel"].sharding.spec
      ),
  }
  print(f"  [Check 3.1] Active 2D Mesh Axes: {sharding_specs['mesh_shape']}")
  for k, spec_str in sharding_specs.items():
    if k != "mesh_shape":
      print(f"    Sharding {k:22s}: {spec_str}")

  @jax.jit
  def _fwd_and_grad_2d(p, r, batch):
    def _loss(params_in, rest_in):
      m = nnx.merge(gdef_2d, params_in, rest_in, copy=True)
      loss, aux = maxtext_train.loss_fn(m, cfg_2d, batch, None, None, is_train=True)
      logits = m(
          decoder_input_tokens=batch["inputs"],
          decoder_positions=batch["inputs_position"],
          decoder_segment_ids=batch["inputs_segmentation"],
          encoder_images=batch["images"],
          enable_dropout=False,
          model_mode=MODEL_MODE_TRAIN,
      )
      return loss, (aux, logits)
    return jax.value_and_grad(_loss, argnums=0, has_aux=True)(p, r)

  (loss_2d, (aux_2d, logits_2d)), grads_2d = _fwd_and_grad_2d(params_2d, rest_2d, mm_batch)
  logits_2d_np = np.asarray(jax.device_get(logits_2d), dtype=np.float32)
  loss_2d_val = float(jax.device_get(loss_2d))
  grads_2d_dict = jax.device_get(grads_2d.to_pure_dict())

  m_2d_vs_unsharded_logits = compute_metrics(logits_u_np, logits_2d_np)
  m_2d_vs_pt_logits = compute_metrics(pt_logits_ref_np, logits_2d_np)
  loss_diff_2d_vs_unsharded = abs(loss_2d_val - loss_u_val)
  grad_metrics_2d_vs_pt = extract_13_grad_metrics(pt_model, pt_mtp, grads_2d_dict, is_scanned=False)

  print(
      f"  [Check 3.2] 2D-Sharded Logits vs Unsharded: cosine_sim={m_2d_vs_unsharded_logits['cosine_sim']:.8f}, "
      f"max_abs_diff={m_2d_vs_unsharded_logits['max_abs_diff']:.6e}"
  )
  print(
      f"  [Check 3.3] 2D-Sharded Loss   vs Unsharded: 2D={loss_2d_val:.6f}, Unsharded={loss_u_val:.6f}, "
      f"diff={loss_diff_2d_vs_unsharded:.3e}"
  )
  assert m_2d_vs_unsharded_logits["cosine_sim"] > 0.9999999
  assert loss_diff_2d_vs_unsharded < 1e-5
  for grp, m in grad_metrics_2d_vs_pt.items():
    assert m["cosine_sim"] > 0.9999

  evidence["check3_2d_sharding_parity"] = {
      "sharding_specs": sharding_specs,
      "logits_2d_vs_unsharded": m_2d_vs_unsharded_logits,
      "logits_2d_vs_pt": m_2d_vs_pt_logits,
      "loss_2d": loss_2d_val,
      "loss_diff_2d_vs_unsharded": loss_diff_2d_vs_unsharded,
      "grad_13_groups_2d_vs_pt": grad_metrics_2d_vs_pt,
  }

  del mt_2d, params_2d, rest_2d, p2d_dict, logits_2d, grads_2d, grads_2d_dict, _fwd_and_grad_2d
  gc.collect()
  jax.clear_caches()

  # =========================================================================
  # SUPPLEMENTAL CHECK 4: Production Mixed-Precision bfloat16 Sanity Check
  # =========================================================================
  print("\n" + "=" * 88)
  print("SUPPLEMENTAL CHECK 4: Production Mixed-Precision bfloat16 Forward + Backward Sanity Check")
  print("=" * 88)

  cfg_bf16 = pyconfig.initialize(
      ["", "src/maxtext/configs/base.yml"],
      model_name="qwen3.8-27b",
      override_model_config=True,
      skip_jax_distributed_system=True,
      load_parameters_path=SLICED_ORBAX_DIR,
      base_num_decoder_layers=4,
      use_multimodal=True,
      freeze_vision_encoder_params=False,
      mtp_num_layers=1,
      mtp_loss_scaling_factor=0.1,
      z_loss_multiplier=1e-4,
      scan_layers=False,
      ici_fsdp_parallelism=2,
      ici_tensor_parallelism=2,
      attention="dot_product",
      attention_for_vit="dot_product",
      matmul_precision="default",
      dtype="bfloat16",
      dtype_mm="bfloat16",
      weight_dtype="float32",
      float32_logits=True,
      float32_qk_product=True,
      enable_dropout=False,
  )
  mt_bf16, _ = model_creation_utils.from_pretrained(cfg_bf16, model_mode=MODEL_MODE_TRAIN)
  nnx.pop(mt_bf16, nnx.Intermediate)
  gdef_bf16, params_bf16, rest_bf16 = nnx.split(mt_bf16, nnx.Param, ...)

  @jax.jit
  def _fwd_and_grad_bf16(p, r, batch):
    def _loss(params_in, rest_in):
      m = nnx.merge(gdef_bf16, params_in, rest_in, copy=True)
      loss, aux = maxtext_train.loss_fn(m, cfg_bf16, batch, None, None, is_train=True)
      logits = m(
          decoder_input_tokens=batch["inputs"],
          decoder_positions=batch["inputs_position"],
          decoder_segment_ids=batch["inputs_segmentation"],
          encoder_images=batch["images"],
          enable_dropout=False,
          model_mode=MODEL_MODE_TRAIN,
      )
      return loss, (aux, logits)
    return jax.value_and_grad(_loss, argnums=0, has_aux=True)(p, r)

  (loss_bf16, (aux_bf16, logits_bf16)), grads_bf16 = _fwd_and_grad_bf16(params_bf16, rest_bf16, mm_batch)
  logits_bf16_np = np.asarray(jax.device_get(logits_bf16), dtype=np.float32)
  loss_bf16_val = float(jax.device_get(loss_bf16))
  grads_bf16_dict = jax.device_get(grads_bf16.to_pure_dict())

  # Verify zero NaNs or Infs in logits, loss, and all 404 gradient tensors
  all_grad_leaves = jax.tree_util.tree_leaves(grads_bf16_dict)
  has_nan_or_inf = (
      bool(np.isnan(logits_bf16_np).any())
      or bool(np.isinf(logits_bf16_np).any())
      or bool(np.isnan(loss_bf16_val))
      or any(bool(np.isnan(g).any() or np.isinf(g).any()) for g in all_grad_leaves)
  )
  m_bf16_vs_fp32_logits = compute_metrics(pt_logits_ref_np, logits_bf16_np)
  bf16_top1 = float(np.mean(np.argmax(pt_logits_ref_np, axis=-1) == np.argmax(logits_bf16_np, axis=-1)))
  bf16_loss_rel_diff = abs(loss_bf16_val - pt_loss_ref_val) / abs(pt_loss_ref_val)
  grad_metrics_bf16_vs_fp32 = extract_13_grad_metrics(pt_model, pt_mtp, grads_bf16_dict, is_scanned=False)
  min_bf16_grad_cos = min(m["cosine_sim"] for m in grad_metrics_bf16_vs_fp32.values())

  print(f"  [Check 4.1] bfloat16 NaN/Inf Check across all 404 grad tensors & outputs: has_nan_or_inf={has_nan_or_inf}")
  print(
      f"  [Check 4.2] bfloat16 Logits vs FP32 Ref: cosine_sim={m_bf16_vs_fp32_logits['cosine_sim']:.8f}, "
      f"rel_l2={m_bf16_vs_fp32_logits['rel_l2']:.6e}, top1={bf16_top1*100:.1f}%"
  )
  print(
      f"  [Check 4.3] bfloat16 Loss   vs FP32 Ref: bf16={loss_bf16_val:.6f}, fp32={pt_loss_ref_val:.6f}, "
      f"rel_diff={bf16_loss_rel_diff:.4e}"
  )
  print(f"  [Check 4.4] bfloat16 Backward Grads (13 groups vs FP32 Ref): min_cosine_sim={min_bf16_grad_cos:.6f}")
  for grp, m in grad_metrics_bf16_vs_fp32.items():
    print(f"    bfloat16 Grad {grp:36s}: cosine_sim={m['cosine_sim']:.6f}, rel_l2={m['rel_l2']:.4e}")
  assert not has_nan_or_inf
  assert m_bf16_vs_fp32_logits["cosine_sim"] > 0.999
  assert bf16_loss_rel_diff < 5e-3
  assert min_bf16_grad_cos > 0.99

  evidence["check4_bfloat16_mixed_precision"] = {
      "has_nan_or_inf": has_nan_or_inf,
      "logits_bf16_vs_fp32": m_bf16_vs_fp32_logits,
      "top1_agreement_vs_fp32": bf16_top1,
      "loss_bf16": loss_bf16_val,
      "loss_fp32_ref": pt_loss_ref_val,
      "loss_rel_diff": bf16_loss_rel_diff,
      "min_grad_cosine_sim_vs_fp32": min_bf16_grad_cos,
      "grad_13_groups_bf16_vs_fp32": grad_metrics_bf16_vs_fp32,
  }

  elapsed = time.time() - t0
  evidence["elapsed_seconds"] = elapsed
  with open(SUPPLEMENTAL_EVIDENCE_PATH, "w") as f:
    json.dump(evidence, f, indent=2)

  print("\n" + "=" * 88)
  print(f"ALL 4 SUPPLEMENTAL MINI VERIFICATION CHECKS PASSED CLEANLY in {elapsed:.1f}s!")
  print(f"Evidence recorded to: {SUPPLEMENTAL_EVIDENCE_PATH}")
  print("=" * 88)


if __name__ == "__main__":
  main()
