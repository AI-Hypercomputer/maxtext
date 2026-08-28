"""MaxText trainer (full 35B, fsdp=8): teacher-force prompt+rollout, logprob of each sampled token; own routing and
sampler-replayed routing (per-position mask: positions without sampler routing use the trainer's own top-k)."""
import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
import tpu_inference  # noqa: F401
import os, sys, time, json
os.environ.setdefault("HF_HOME", "/mnt/disks/persist"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.abspath(".")); sys.path.insert(0, os.path.abspath("src"))
from flax import nnx
from flax import linen as nn
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils
from maxtext.layers import moe as moe_mod
from tests.utils.test_helpers import get_test_config_path
CKPT = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"; HF = "Qwen/Qwen3.5-35B-A3B"
OUT = "/mnt/disks/persist/pr4925_repro"; TAG = os.environ.get("TAG", "bf16"); ML = 1024
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
f32 = lambda v: np.asarray(jax.device_get(v)).astype(np.float32)
z = np.load(f"{OUT}/real35b_L3.npz"); prompt = z["tokens"]; B, S = prompt.shape
ro = np.load(f"{OUT}/rollout35b_{TAG}.npz"); gen = ro["gen_ids"]; G = gen.shape[1]; L = S + G; T = B * L
seq = np.concatenate([prompt, gen], 1).astype(np.int32); assert (seq >= 0).all()
LV = L; L = ML  # splash attention needs q_seq_len % 512 == 0 -> pad to ML with segment id 0
seq = np.concatenate([seq, np.full((B, L - LV), 248044, np.int32)], 1); T = B * L
segm = np.concatenate([np.ones((B, LV), np.int32), np.zeros((B, L - LV), np.int32)], 1)
routed = ro["routed"]; log(f"rollout: gen {gen.shape}; routed {routed.shape}")   # [B, rows, layers, k]
common = dict(model_name="qwen3.5-35b-a3b", load_parameters_path=CKPT, tokenizer_path=HF, run_name="pr4925_real",
  base_output_directory="/home/wenxindong_google_com/.claude/jobs/7ea88918/tmp/out", scan_layers=False,
  checkpoint_storage_use_ocdbt=True, checkpoint_storage_use_zarr3=True, convert_checkpoint_if_possible=False,
  dtype="bfloat16", weight_dtype="bfloat16", max_target_length=ML, max_prefill_predict_length=ML,
  per_device_batch_size=1.0, log_config=False, enable_nnx=True, pure_nnx=True, pure_nnx_decoder=True,
  enable_checkpointing=True, async_checkpointing=False, float32_logits=True, float32_gate_logits=True, float32_weight_sum=True)
common.update(json.loads(os.environ.get("EXTRA", "{}")))
cfg = pyconfig.initialize([sys.argv[0], get_test_config_path(), "attention=flash"], use_tokamax_splash=True,
  sa_use_base2_exp=False, sa_fuse_reciprocal=True, sparse_matmul=True, megablox=True, use_tokamax_gmm=True, use_gmm_v2=True,
  wi_tile_fwd_batch_seq=256, wi_tile_fwd_embed_dim=128, wi_tile_fwd_mlp_dim=128, **common)
model, mesh = model_creation_utils.from_pretrained(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN); log("loaded")
NLAY = cfg.num_decoder_layers; K = cfg.num_experts_per_tok
# replay indices per layer: [B, L, K]; -1 where the sampler provided no routing (positions >= routed rows)
ridx = np.full((NLAY, B, L, K), -1, np.int32)
rows = routed.shape[1]
for b in range(B):
  n = min(rows, L); ridx[:, b, :n, :] = routed[b, :n, :, :K].transpose(1, 0, 2)
if os.environ.get("GEN_ONLY_REPLAY") == "1":
  ridx[:, :, :S, :] = -1       # own routing on all prefill rows 0..S-1 (row S-1 predicts the first decode token and is a prefill row)
zero_rows = np.all(routed[:, :, :, :K] == 0, axis=(2, 3))   # engine zero-fills rows it did not capture (prefill under attn_dp+chunked prefill)
for b in range(B):
  n = min(rows, L); ridx[:, b, :n, :][:, zero_rows[b, :n], :] = -1
log(f"zero-filled routed rows treated as no-routing: {zero_rows.sum()} of {zero_rows.size}")
log(f"replay coverage: {rows}/{L} positions per sequence; gen-only replay={os.environ.get('GEN_ONLY_REPLAY')}")
REPLAY = {"queue": None}; _orig = moe_mod.RoutedMoE.get_topk
def _replay_get_topk(self, gate_logits, pre_bias_logits, rngs=None, input_ids=None):
  if REPLAY["queue"] is None: return _orig(self, gate_logits, pre_bias_logits, rngs, input_ids)
  idx = REPLAY["queue"].pop(0); bl = gate_logits.shape[0]
  if bl != idx.shape[0]:
    idx = jax.lax.dynamic_slice_in_dim(idx, jax.lax.axis_index("fsdp") * bl, bl, axis=0)
  _, own = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
  idx = jnp.where(idx >= 0, idx, own)
  sel = jnp.take_along_axis(gate_logits.astype(jnp.float32), idx, axis=-1); w = jax.nn.softmax(sel, axis=-1)
  if self.config.norm_topk_prob: w = w / w.sum(axis=-1, keepdims=True)
  return w.astype(self.dtype), idx
moe_mod.RoutedMoE.get_topk = _replay_get_topk
gd, st = nnx.split(model)
def make_fn(replay):
  @jax.jit
  def fn(st, tok, pos, seg, ridx):
    m = nnx.merge(gd, st); REPLAY["queue"] = list(ridx) if replay else None
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      out = m(tok, pos, seg, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
    REPLAY["queue"] = None
    logits = out[0] if isinstance(out, tuple) else out
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    nxt = jnp.concatenate([tok[:, 1:], tok[:, :1]], 1)
    return jnp.take_along_axis(lp, nxt[..., None], -1)[..., 0], jnp.argmax(lp, -1).astype(jnp.int32)
  return fn
res = {}
with mesh, nn.logical_axis_rules(cfg.logical_axis_rules):
  sh2 = NamedSharding(mesh, P(("data", "fsdp"), None)); sh3 = NamedSharding(mesh, P(("data", "fsdp"), None, None))
  tok = jax.device_put(seq, sh2); pos = jax.device_put(np.broadcast_to(np.arange(L, dtype=np.int32), (B, L)), sh2); seg = jax.device_put(segm, sh2)
  rl = [jax.device_put(ridx[i], sh3) for i in range(NLAY)]
  for mode in ("own", "replay"):
    lp, top1 = make_fn(mode == "replay")(st, tok, pos, seg, rl); jax.block_until_ready(lp)
    res[f"{mode}_logp"] = f32(lp); res[f"{mode}_top1"] = np.asarray(jax.device_get(top1)).astype(np.int32)
    log(f"{mode}: done; mean logp of sampled tokens = {res[f'{mode}_logp'][:, S-1:LV-1].mean():.3f}")
np.savez(f"{OUT}/maxtext35b_score_{TAG}" + os.environ.get("SUFFIX", "") + ".npz", **res); log("done")
