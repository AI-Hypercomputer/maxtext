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
log('train model loaded')

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
lp = np.asarray(jax.nn.log_softmax(logits.astype(jnp.float32), -1)); nxt = np.roll(tokens_np, -1, axis=1)
logp_actual = np.take_along_axis(lp, nxt[..., None], -1)[..., 0]; top1 = lp.argmax(-1)
np.savez(f'{OUT}/maxtext35b_full_logprobs.npz', logp=logp_actual, top1=top1)
log(f'saved full-model logprobs; mean logp={logp_actual[:, :-1].mean():.3f}')
h_np = f32(h_in); log(f"h_in rms={np.sqrt(np.mean(h_np**2)):.3f} max={np.abs(h_np).max():.1f}; y3 rms={np.sqrt(np.mean(f32(y3)**2)):.3f}")


log('done')
