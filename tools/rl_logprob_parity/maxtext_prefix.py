"""MaxText trainer, Qwen3.5-35B-A3B, real ckpt + real tokens: layers 0..7 with (a) its own MoE routing and (b) the
sampler's routing replayed (per-layer top-k expert indices captured from the torchax prefix run), logit-lens logprobs
after 4 and 8 layers, and layer 3 on h_in with replayed routing (single-layer remeasure)."""
import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
import tpu_inference  # noqa: F401
import os, sys, time
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
OUT = "/mnt/disks/persist/pr4925_repro"; S = 512; B = 8; NL = 8; L3 = 3
TX = os.environ.get("TX_NPZ", f"{OUT}/torchax_prefix_dp4tp2_ep1.npz")
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
f32 = lambda v: np.asarray(jax.device_get(v)).astype(np.float32)
z = np.load(f"{OUT}/real35b_L3.npz"); h_np, tokens = z["h_in"], z["tokens"]; T = B * S
tx = np.load(TX); experts_tx = tx["experts"]; experts3_tx = tx["experts3_on_maxtext_h"]
log(f"torchax experts: {experts_tx.shape} (layers x T x k); layer3-on-h_in experts: {experts3_tx.shape}")

common = dict(model_name="qwen3.5-35b-a3b", load_parameters_path=CKPT, tokenizer_path=HF, run_name="pr4925_real",
  base_output_directory="/home/wenxindong_google_com/.claude/jobs/7ea88918/tmp/out", scan_layers=False,
  checkpoint_storage_use_ocdbt=True, checkpoint_storage_use_zarr3=True, convert_checkpoint_if_possible=False,
  dtype="bfloat16", weight_dtype="bfloat16", max_target_length=S, max_prefill_predict_length=S,
  per_device_batch_size=1.0, log_config=False, enable_nnx=True, pure_nnx=True, pure_nnx_decoder=True,
  enable_checkpointing=True, async_checkpointing=False, float32_logits=True, float32_gate_logits=True, float32_weight_sum=True)
cfg = pyconfig.initialize([sys.argv[0], get_test_config_path(), "attention=flash"], use_tokamax_splash=True,
  sa_use_base2_exp=False, sa_fuse_reciprocal=True, sparse_matmul=True, megablox=True, use_tokamax_gmm=True, use_gmm_v2=True,
  wi_tile_fwd_batch_seq=256, wi_tile_fwd_embed_dim=128, wi_tile_fwd_mlp_dim=128, **common)
model, mesh = model_creation_utils.from_pretrained(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN); log("loaded")

# ---- router replay: class-level patch of RoutedMoE.get_topk consuming a per-trace queue of replayed indices ----
REPLAY = {"queue": None}
_orig_get_topk = moe_mod.RoutedMoE.get_topk
def _replay_get_topk(self, gate_logits, pre_bias_logits, rngs=None, input_ids=None):
  if REPLAY["queue"] is None:
    return _orig_get_topk(self, gate_logits, pre_bias_logits, rngs, input_ids)
  idx = REPLAY["queue"].pop(0)                                   # [B,S,k] int32 (global), sampler's expert ids
  bl = gate_logits.shape[0]                                       # get_topk runs inside the MoE shard_map: local batch
  if bl != idx.shape[0]:                                          # slice this shard's rows (batch is sharded over fsdp here)
    shard = jax.lax.axis_index("fsdp")
    idx = jax.lax.dynamic_slice_in_dim(idx, shard * bl, bl, axis=0)
  sel = jnp.take_along_axis(gate_logits.astype(jnp.float32), idx, axis=-1)
  w = jax.nn.softmax(sel, axis=-1)                               # same math as MaxText's own path (softmax over the top-k logits)
  if self.config.norm_topk_prob:
    w = w / w.sum(axis=-1, keepdims=True)
  return w.astype(self.dtype), idx
moe_mod.RoutedMoE.get_topk = _replay_get_topk

gd, st = nnx.split(model)
def make_fn(n_layers, replay):
  @jax.jit
  def fn(st, tokens_b, pos, seg, replay_idx):
    m = nnx.merge(gd, st); dec = m.decoder
    REPLAY["queue"] = list(replay_idx) if replay else None
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      y = dec._apply_embedding(m.token_embedder, tokens_b, pos, True, MODEL_MODE_TRAIN); outs = {}
      for i in range(n_layers):
        y = getattr(dec, f"layers_{i}")(y, seg, pos, True, MODEL_MODE_TRAIN)[0]; outs[i] = y
      lens = {}
      for k in (4, 8):
        if k <= n_layers:
          logits = dec.apply_output_head(m.token_embedder, outs[k - 1], True, MODEL_MODE_TRAIN)
          lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1).reshape(T, -1)
          ids = tokens_b.reshape(-1); nxt = jnp.concatenate([ids[1:], ids[:1]])
          lens[k] = (lp[jnp.arange(T), nxt], jnp.argmax(lp, -1).astype(jnp.int32), lp[: 2 * S])
    REPLAY["queue"] = None
    return {k: outs[k].astype(jnp.float32) for k in (2, 3, 7) if k < n_layers}, lens
  return fn
def make_layer3_fn(replay):
  @jax.jit
  def fn(st, h, pos, seg, replay_idx):
    m = nnx.merge(gd, st); dec = m.decoder
    REPLAY["queue"] = [replay_idx] if replay else None
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      y = getattr(dec, f"layers_{L3}")(h, seg, pos, True, MODEL_MODE_TRAIN)[0]
    REPLAY["queue"] = None
    return y.astype(jnp.float32)
  return fn

res = {}
with mesh, nn.logical_axis_rules(cfg.logical_axis_rules):
  sh3 = NamedSharding(mesh, P(("data", "fsdp"), None, None)); sh2 = NamedSharding(mesh, P(("data", "fsdp"), None))
  tok = jax.device_put(tokens.astype(np.int32), sh2)
  pos = jax.device_put(np.broadcast_to(np.arange(S, dtype=np.int32), (B, S)), sh2)
  seg = jax.device_put(np.ones((B, S), np.int32), sh2)
  h = jax.device_put(jnp.asarray(h_np).astype(jnp.bfloat16), sh3)
  ridx = [jax.device_put(experts_tx[i].reshape(B, S, -1).astype(np.int32), NamedSharding(mesh, P(("data", "fsdp"), None, None))) for i in range(NL)]
  ridx3 = jax.device_put(experts3_tx.reshape(B, S, -1).astype(np.int32), NamedSharding(mesh, P(("data", "fsdp"), None, None)))
  for mode, replay in (("own", False), ("replay", True)):
    outs, lens = make_fn(NL, replay)(st, tok, pos, seg, ridx); jax.block_until_ready(outs[7])
    res[f"{mode}_h_after3"] = f32(outs[2]); res[f"{mode}_h_after4"] = f32(outs[3]); res[f"{mode}_h_after8"] = f32(outs[7])
    for k in (4, 8):
      a, b, c = lens[k]; res[f"{mode}_lens{k}_logp_actual"] = f32(a); res[f"{mode}_lens{k}_top1"] = np.asarray(jax.device_get(b)).astype(np.int32); res[f"{mode}_lens{k}_logprobs_2seq"] = f32(c)
    log(f"{mode}: prefix done; lens4 mean logp={res[f'{mode}_lens4_logp_actual'].mean():.3f} top1-acc={np.mean(res[f'{mode}_lens4_top1']==np.roll(tokens.reshape(-1),-1)):.3f}; "
        f"lens8 mean logp={res[f'{mode}_lens8_logp_actual'].mean():.3f} top1-acc={np.mean(res[f'{mode}_lens8_top1']==np.roll(tokens.reshape(-1),-1)):.3f}")
    y3 = make_layer3_fn(replay)(st, h, pos, seg, ridx3); jax.block_until_ready(y3); res[f"{mode}_y3"] = f32(y3)
    log(f"{mode}: layer3 on h_in done")
np.savez(f"{OUT}/maxtext_prefix" + os.environ.get("SUFFIX", "") + ".npz", **res); log("done")
