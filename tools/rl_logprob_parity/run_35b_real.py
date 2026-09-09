"""Qwen3.5-35B-A3B layer 3 (first full-attention layer) with REAL checkpoint weights and REAL tokens:
training path vs vLLM-serving path, both through Qwen3_5DecoderLayer.__call__ under jit.

  train : from_pretrained(base.yml, attention=flash/Tokamax splash, Tokamax GMM v2), fsdp mesh, MODEL_MODE_TRAIN
  infer : from_pretrained(inference/vllm.yml, attention=vllm_rpa, model_call_mode=inference,
          prefuse_moe_weights=True, TP=INFER_TP with kv heads padded to TP like the adapter), MODEL_MODE_AUTOREGRESSIVE,
          tokens as [T,1,D], real tpu_inference AttentionMetadata + RPA kv cache
  input : real text -> HF tokenizer -> embedding + layers 0..2 on the training model (jit) -> hidden state into layer 3
"""
import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
import tpu_inference  # noqa: F401
import os, sys, time, glob
os.environ.setdefault("HF_HOME", "/mnt/disks/persist"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.abspath(".")); sys.path.insert(0, os.path.abspath("src"))
from flax import nnx
from flax import linen as nn
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from transformers import AutoTokenizer

CKPT = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"
HF = "Qwen/Qwen3.5-35B-A3B"
OUT = "/mnt/disks/persist/pr4925_repro"; SEQ_LEN = 512; B = 8; LAYER_IDX = 3
INFER_TP = int(os.environ.get("INFER_TP", "8"))
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
f32 = lambda v: np.asarray(jax.device_get(v)).astype(np.float32)
def hbm(tag):
  log(f"HBM {tag}: " + " ".join(f"{d.memory_stats()['bytes_in_use']/2**30:.1f}" for d in jax.local_devices()) + " GiB")

# ---- real tokens ----
tok = AutoTokenizer.from_pretrained(HF)
text = "\n\n".join(open(f).read() for f in sorted(glob.glob("docs/**/*.md", recursive=True)))
ids = tok(text)["input_ids"]
assert len(ids) >= B * SEQ_LEN, len(ids)
tokens_np = np.array(ids[: B * SEQ_LEN], dtype=np.int32).reshape(B, SEQ_LEN)
log(f"tokens {tokens_np.shape} from {len(ids)} doc tokens; first: {tok.decode(tokens_np[0, :12])!r}")

common = dict(model_name="qwen3.5-35b-a3b", load_parameters_path=CKPT, tokenizer_path=HF, run_name="pr4925_real",
  base_output_directory="/home/wenxindong_google_com/.claude/jobs/7ea88918/tmp/out", scan_layers=False,
  checkpoint_storage_use_ocdbt=True, checkpoint_storage_use_zarr3=True, convert_checkpoint_if_possible=False,
  dtype="bfloat16", weight_dtype="bfloat16", max_target_length=SEQ_LEN, max_prefill_predict_length=SEQ_LEN,
  per_device_batch_size=1.0, log_config=False, enable_nnx=True, pure_nnx=True, pure_nnx_decoder=True,
  enable_checkpointing=True, async_checkpointing=False, float32_logits=True, float32_gate_logits=True, float32_weight_sum=True)
cfg_train = pyconfig.initialize([sys.argv[0], get_test_config_path(), "attention=flash"], use_tokamax_splash=True,
  sa_use_base2_exp=False, sa_fuse_reciprocal=True, sparse_matmul=True, megablox=True, use_tokamax_gmm=True, use_gmm_v2=True,
  wi_tile_fwd_batch_seq=256, wi_tile_fwd_embed_dim=128, wi_tile_fwd_mlp_dim=128, **common)
infer_over = {}
if INFER_TP > cfg_train.num_kv_heads and INFER_TP % cfg_train.num_kv_heads == 0:
  infer_over["base_num_kv_heads"] = INFER_TP  # adapter pads kv heads to TP
# adapter pads moe_intermediate_size so each TP shard's (2*F/TP) is a multiple of 2*num_lanes
_num_lanes = 128; _F = cfg_train.moe_mlp_dim
if (_F // INFER_TP) % (2 * _num_lanes) != 0:
  _p = 1 << (_F - 1).bit_length()
  while (_p // INFER_TP) < (2 * _num_lanes):
    _p = 1 << _p.bit_length()
  infer_over["padded_base_moe_mlp_dim"] = _p
  print(f"padding moe_mlp_dim {_F} -> {_p} for TP={INFER_TP} (adapter logic)")
cfg_infer = pyconfig.initialize([sys.argv[0], get_test_config_path("inference/vllm.yml"), "attention=vllm_rpa",
  "prefuse_moe_weights=True", "model_call_mode=inference", "allow_split_physical_axes=True",
  f"ici_tensor_parallelism={INFER_TP}"], **common, **infer_over)
log(f"cfg: emb={cfg_train.emb_dim} q={cfg_train.num_query_heads} kv={cfg_train.num_kv_heads}->{cfg_infer.num_kv_heads} "
    f"E={cfg_train.num_experts} k={cfg_train.num_experts_per_tok} layers={cfg_train.num_decoder_layers} infer_tp={INFER_TP}")

# ---- load both models from the checkpoint ----
train_model, train_mesh = model_creation_utils.from_pretrained(cfg_train, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN)
log(f"train model loaded; mesh {dict(zip(train_mesh.axis_names, train_mesh.devices.shape))}"); hbm("train load")
infer_mesh = Mesh(np.array(jax.devices()[:INFER_TP]).reshape(tuple(INFER_TP if a == "model" else 1 for a in cfg_infer.mesh_axes)), cfg_infer.mesh_axes)
with infer_mesh, nn.logical_axis_rules(cfg_infer.logical_axis_rules):
  infer_model = model_creation_utils.from_pretrained(cfg_infer, mesh=infer_mesh, model_mode=MODEL_MODE_AUTOREGRESSIVE)
log("infer model loaded"); hbm("infer load")
train_layer = getattr(train_model.decoder, f"layers_{LAYER_IDX}"); infer_layer = getattr(infer_model.decoder, f"layers_{LAYER_IDX}")
log("layer types:", type(train_layer.attention).__name__, type(infer_layer.attention).__name__)
wi = infer_layer.mlp.routed_experts.wi; log(f"infer prefused wi {wi.value.shape} {wi.value.sharding.spec}")

# ---- train side: embedding + layers 0..2 -> h_in ; layer 3 -> y3 ; full model -> logits (ckpt sanity) ----
gd, st = nnx.split(train_model)
@jax.jit
def train_fn(st, tokens, pos, seg):
  m = nnx.merge(gd, st); dec = m.decoder
  with nn.logical_axis_rules(cfg_train.logical_axis_rules):
    y = dec._apply_embedding(m.token_embedder, tokens, pos, True, MODEL_MODE_TRAIN)
    for i in range(LAYER_IDX):
      y = getattr(dec, f"layers_{i}")(y, seg, pos, True, MODEL_MODE_TRAIN)[0]
    h_in = y
    y3 = getattr(dec, f"layers_{LAYER_IDX}")(h_in, seg, pos, True, MODEL_MODE_TRAIN)[0]
    out = m(tokens, pos, seg, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
    logits = out[0] if isinstance(out, tuple) else out
  return h_in, y3, logits
with train_mesh, nn.logical_axis_rules(cfg_train.logical_axis_rules):
  sh3 = NamedSharding(train_mesh, P(("data", "fsdp"), None));
  tokens = jax.device_put(tokens_np, sh3)
  pos = jax.device_put(np.broadcast_to(np.arange(SEQ_LEN, dtype=np.int32), (B, SEQ_LEN)), sh3)
  seg = jax.device_put(np.ones((B, SEQ_LEN), np.int32), sh3)
  h_in, y3, logits = train_fn(st, tokens, pos, seg); jax.block_until_ready(y3)
pred = np.asarray(jnp.argmax(logits[:, :-1], -1)); acc = np.mean(pred == tokens_np[:, 1:])
log(f"train prefix+layer{LAYER_IDX}+full model done; next-token top-1 acc on real text = {acc:.3f} (ckpt sanity)")
h_np = f32(h_in); log(f"h_in rms={np.sqrt(np.mean(h_np**2)):.3f} max={np.abs(h_np).max():.1f}; y3 rms={np.sqrt(np.mean(f32(y3)**2)):.3f}")

# ---- infer side: layer 3, adapter-style ----
D = cfg_train.emb_dim; T = B * SEQ_LEN
block = 128; pages = B * ((SEQ_LEN + block - 1) // block)
kv_shape = get_kv_cache_shape(pages, block, cfg_infer.num_kv_heads, cfg_infer.head_dim, jnp.bfloat16)
rs = NamedSharding(infer_mesh, P())
gd_i, st_i = nnx.split(infer_layer)
@jax.jit
def infer_fn(st, x2, pos2, kv, bt, sl, qsl, rd):
  m = nnx.merge(gd_i, st)
  md = AttentionMetadata(input_positions=pos2.reshape(-1), block_tables=bt, seq_lens=sl, query_start_loc=qsl, request_distribution=rd)
  with nn.logical_axis_rules(cfg_infer.logical_axis_rules):
    y, kv2 = m(x2, None, pos2, True, MODEL_MODE_AUTOREGRESSIVE, kv_cache=kv, attention_metadata=md)
  return y, kv2
with infer_mesh, nn.logical_axis_rules(cfg_infer.logical_axis_rules):
  x2 = jax.device_put(jnp.asarray(h_in).astype(jnp.bfloat16).reshape(T, 1, D), rs)
  pos2 = jax.device_put(np.broadcast_to(np.arange(SEQ_LEN, dtype=np.int32), (B, SEQ_LEN)).reshape(T, 1), rs)
  kv = jax.device_put(jnp.zeros(kv_shape, jnp.bfloat16), rs)
  bt = jax.device_put(jnp.arange(pages, dtype=jnp.int32), rs); sl = jax.device_put(jnp.array([SEQ_LEN] * B, jnp.int32), rs)
  qsl = jax.device_put(jnp.arange(0, (B + 1) * SEQ_LEN, SEQ_LEN, dtype=jnp.int32), rs); rd = jax.device_put(jnp.array([0, 0, B], jnp.int32), rs)
  y_inf, kv2 = infer_fn(st_i, x2, pos2, kv, bt, sl, qsl, rd); jax.block_until_ready(y_inf)
log(f"infer layer{LAYER_IDX} done {y_inf.shape} kv written={bool(jnp.any(kv2 != 0))}"); hbm("after infer")

# ---- compare ----
def stats(name, a, b):
  a = f32(a).reshape(-1, D); b = f32(b).reshape(-1, D)
  cos = np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))
  pt = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
  print(f"  {name:<36} L_inf={np.max(np.abs(a-b)):.4e} MAE={np.mean(np.abs(a-b)):.4e} relL2={np.linalg.norm(a-b)/np.linalg.norm(a):.4e} "
        f"cos={cos:.6f} per-token cos mean={pt.mean():.6f} min={pt.min():.6f}")
yt, yi = f32(y3), f32(y_inf).reshape(B, SEQ_LEN, D)
print("== Qwen3.5-35B-A3B layer 3, real weights, real tokens, bf16 ==")
stats("train vs infer (layer output)", yt, yi)
stats("(train - h_in) vs (infer - h_in)", yt - h_np, yi - h_np)
bf = lambda a: f32(jnp.asarray(a).astype(jnp.bfloat16))
stats("bf16 rounding floor: bf16(train) vs train", bf(yt), yt)
np.savez(f"{OUT}/real35b_L{LAYER_IDX}.npz", h_in=h_np, train=yt, infer=yi, tokens=tokens_np)
log("done")
