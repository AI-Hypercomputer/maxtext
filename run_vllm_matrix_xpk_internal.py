"""Drives the vLLM decode validation matrix across models and scan modes.
Optimized for XPK execution.
"""

import os
import subprocess
import csv
import json
import sys

# --- ABSOLUTE TOP: Register OmegaConf resolver for HF_TOKEN ---
# This fixes the "Unsupported interpolation type HF_TOKEN" error in MaxText
try:
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver("HF_TOKEN"):
        print("[INFO] Registering OmegaConf resolver for HF_TOKEN...")
        OmegaConf.register_new_resolver("HF_TOKEN", lambda: os.environ.get("HF_TOKEN", ""))
except ImportError:
    print("[WARNING] omegaconf not found. Resolver registration skipped.")
# -------------------------------------------------------------

# Try to import HF_IDS, fallback to empty dict if maxtext is not available locally
try:
    from maxtext.utils.globals import HF_IDS
except ImportError:
    HF_IDS = {}

# Explicit (model, scan mode) pairs rather than a full cross product: not every model has been
# converted in both layouts, and asking for one that was not just produces a confusing failure.
# Ordered by risk, not alphabetically, and deliberately so. A case that crashes the Pathways
# workers hard enough takes the whole jobset with it -- gemma4-e2b crash-looped the workers until
# the jobset hit its restart limit, which killed the run at case 5 of 19 and lost every result
# after it. The cases known to run cleanly go first so their results survive; the ones that are
# expected to fail, or have been seen to take the pod down, go last.
CASES = [
    ("gemma2-2b", "scanned"),
    ("gemma2-2b", "unscanned"),
    ("gemma3-4b", "scanned"),
    ("gemma3-4b", "unscanned"),
    ("qwen3-0.6b", "scanned"),
    ("qwen3-0.6b", "unscanned"),
    ("qwen2.5-1.5b", "scanned"),
    ("qwen2.5-1.5b", "unscanned"),
    ("mistral-7b", "scanned"),
    ("mistral-7b", "unscanned"),
    # Crash-looped the Pathways workers on the previous run; kept, but last among the runnable
    # cases so it cannot take the others down with it.
    ("gemma4-e2b", "unscanned"),
    # The three below are expected to fail on this v6e-32, for reasons that are not configuration:
    #   deepseek4-tiny  - a unit-test shape with no HuggingFace config, which vLLM needs for the
    #                     model geometry; it can only borrow V4-Flash's 284B dimensions.
    #   deepseek3-671b  - ~1342GiB of bf16 weights against the slice's 1024GiB of HBM.
    #   gemma4-31b      - no converted checkpoint exists in any layout (the bucket has 26b).
    ("deepseek4-tiny", "scanned"),
    ("deepseek4-tiny", "unscanned"),
    ("deepseek4-284b", "scanned"),
    ("deepseek4-284b", "unscanned"),
    ("deepseek3-671b", "scanned"),
    ("deepseek3-671b", "unscanned"),
    ("gemma4-31b", "scanned"),
    ("gemma4-31b", "unscanned"),
]

# The Pathways workers do not survive many vLLM engines in one pod: two separate runs crash-looped
# the workers on the fifth case and lost the whole jobset, once on gemma4-e2b and once on
# qwen3-0.6b, which is far too small to be at fault on its own. Whatever each engine leaves behind
# accumulates. So the matrix is submitted in small batches, one workload each, selected with
# MATRIX_CASE_SLICE="start:end" (0-based, end exclusive). Unset runs everything.
_slice = os.environ.get("MATRIX_CASE_SLICE", "")
if _slice:
  _start, _end = (int(x) for x in _slice.split(":"))
  CASES = CASES[_start:_end]
  print(f"[INFO] Running cases {_start}:{_end} of the matrix -> {CASES}")

GCS_BASE = "gs://mesa-maxtext/validation_runs/post_train_layout_0811_2"
LOCAL_LOGS = "/tmp/local_logs_vllm"
CSV_REPORT = "/tmp/vllm_validation_summary.csv"

# HF_IDS only resolves when maxtext is importable, which it isn't when this script is run
# outside the container. Pin the repos this matrix needs so the tokenizer resolves either way.
# deepseek4-tiny is a unit-test shape with no HuggingFace repo of its own; it borrows the V4-Flash
# tokenizer, whose vocab_size (129280) matches the tiny config exactly.
HF_ID_FALLBACKS = {
    "deepseek4-284b": "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek4-tiny": "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek3-671b": "deepseek-ai/DeepSeek-V3",
    # Matches the tokenizer.mistral-v3 asset MaxText ships for this model.
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
}

# (tensor, expert) parallelism for the DeepSeek MoE models. These are separate mesh axes and
# multiply, and vLLM derives data parallelism as num_devices // (tp * ep), so the product must
# divide the slice. Sized for the v6e-32 this matrix runs on.
DEEPSEEK_PARALLELISM = {
    "deepseek4-284b": (2, 16),  # 32 devices, DP=1; EP divides 256 experts, TP <= 64 query heads
    "deepseek4-tiny": (2, 4),  # 16 experts, 4 query heads, DP=4
    "deepseek3-671b": (2, 16),  # 256 experts, DP=1
}

# Pre-converted base checkpoints to decode from, keyed by (model, scan mode). Anything not
# listed here falls back to the GCS_BASE layout written by the pre-train matrix.
# The DeepSeek paths are the golden checkpoints the daily v4 e2e test decodes from, see
# tests/end_to_end/tpu/deepseek/v4-284b/2_test_deepseek.sh.
#
# Models mapped to "" have no checkpoint anywhere. We drop load_parameters_path for those, which
# makes vLLM fall back to load_format="dummy" (random weights). The generated text is garbage, but
# the run still exercises model build, sharding and the scanned/unscanned code paths end to end.
HF_CONVERSIONS = "gs://mesa-maxtext/huggingface_transformers"

BASE_CHECKPOINTS = {
    # No conversion exists for these, so they decode from random weights.
    ("deepseek4-tiny", "scanned"): "",
    ("deepseek4-tiny", "unscanned"): "",
    ("gemma4-31b", "scanned"): "",
    ("gemma4-31b", "unscanned"): "",
    ("llama3.1-8b", "scanned"): f"{HF_CONVERSIONS}/llama3.1-8b-Instruct/to_maxtext/scanned/0/items",
    # The checkpoint the gemma2 e2e test uses, rebuilt daily by CI. Kept because it is the
    # canonical source, not because it fixed anything: it was run as an A/B against the
    # mesa-maxtext conversion to see whether that conversion was the reason the scanned layout
    # decodes to garbage. It is not. Both decode to garbage, and both start with the same tokens
    # ("car non di ...") despite being different dtypes and 2.5x apart in size, which says the
    # loaded weights barely reach the computation. The fault is in the scanned weight mapping at
    # run time, not in either checkpoint.
    ("gemma2-2b", "scanned"): "gs://maxtext-gemma/unified/gemma2/2b/scanned/2026-08-12-08-08/0/items",
    # DeepSeek keeps its golden checkpoints in a bucket of its own, see
    # tests/end_to_end/tpu/deepseek/.
    ("deepseek4-284b", "scanned"): "gs://maxtext-deepseek/deepseek4-284b/2026-07-24/scanned/0/items",
    ("deepseek4-284b", "unscanned"): "gs://maxtext-deepseek/deepseek4-284b/2026-07-24/unscanned/0/items",
    ("deepseek3-671b", "scanned"): "gs://maxtext-deepseek/deepseek3-671b/2025-03-31/scanned/0/items",
    ("deepseek3-671b", "unscanned"): "gs://maxtext-deepseek/deepseek3-671b/2025-03-31/unscanned/0/items",
    # Everything else reads the standard HuggingFace conversion layout.
    **{
        (model, mode): f"{HF_CONVERSIONS}/{model}/to_maxtext/{mode}/0/items"
        for model, mode in (
            ("gemma2-2b", "unscanned"),
            ("gemma3-4b", "scanned"),
            ("gemma3-4b", "unscanned"),
            ("gemma4-e2b", "unscanned"),
            ("qwen3-0.6b", "scanned"),
            ("qwen3-0.6b", "unscanned"),
            ("qwen2.5-1.5b", "scanned"),
            ("qwen2.5-1.5b", "unscanned"),
            ("mistral-7b", "scanned"),
            ("mistral-7b", "unscanned"),
        )
    },
}

# vLLM HBM headroom per model, against the 1024GiB this v6e-32 has. The small models need almost
# none; the DeepSeek MoEs need nearly all of it (284B is ~568GiB in bf16, 671B ~1342GiB and so
# does not fit at all -- it is in the matrix because it was asked for, and will report the
# shortfall rather than silently doing something else).
HBM_UTILIZATION = {
    "deepseek4-284b": 0.9,
    "deepseek3-671b": 0.9,
}
# The dense models run on a single chip, so this budget is per chip (32GiB) and is shared with
# XLA's compilation temporaries. Both extremes fail: 0.35 caps at 11.2GiB and gemma3-4b needs
# 14.45GiB, while 0.9 caps at 28.8GiB and leaves too little for the temporaries ("HLO temporaries
# (33.70G) exceeds available HBM"). 0.5 clears the largest dense model here and still leaves 16GiB.
#
# Sharding them instead would be the natural fix, but Pathways rejects it on this topology:
# ici_tensor_parallelism=4 asks for a 4,1,1 subslice while each VM's chips are 2,2,1, and the
# mesh is built linearly ("Not a valid subslice size because bounds are not along host
# boundaries"). So gemma4-31b, at ~62GiB, cannot fit here in any configuration.
DEFAULT_HBM_UTILIZATION = 0.5


# Models whose HuggingFace repo carries no chat template at all -- neither a chat_template key in
# tokenizer_config.json nor a chat_template.jinja beside it. These are base rather than
# instruction-tuned repos, and asking for one dies in tokenizer.apply_chat_template with "Cannot
# use chat template functions because tokenizer.chat_template is not set". They decode the raw
# prompt instead, which is the right thing for a base model.
NO_CHAT_TEMPLATE = {
    "gemma2-2b",  # google/gemma-2-2b
    "mistral-7b",  # mistralai/Mistral-7B-v0.3
    "deepseek4-284b",  # deepseek-ai/DeepSeek-V4-Flash
    "deepseek4-tiny",  # borrows the V4-Flash tokenizer
}


def get_hf_id(model_name):
  """Returns the HuggingFace repo id backing a MaxText model name."""
  return HF_IDS.get(model_name) or HF_ID_FALLBACKS.get(model_name, model_name)


def execute_command(cmd, log_path):
  """Executes a subprocess command, writes to log and prints to stdout."""
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  env = os.environ.copy()

  cmd_str = " ".join(cmd)
  print(f"\n[EXECUTING]: {cmd_str}")
  print(f"[LOG PATH]: {log_path}")

  with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"Command: {cmd_str}\n\n")
    f.flush()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
        process.wait(timeout=3600)

  if process.returncode != 0:
    print(f"[ERROR] Job failed with return code {process.returncode}. Check logs at: {log_path}")
  return process.returncode == 0


def run_matrix():
  """Runs the vLLM decode validation matrix for all models and scan modes."""
  os.makedirs(LOCAL_LOGS, exist_ok=True)
  with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Scan Mode", "Phase", "Run Name", "Status"])

  all_passed = True

  for model, scan_mode in CASES:
    print(f"\n{'='*80}\nStarting vLLM decode for Model: {model} | Scan Mode: {scan_mode}\n{'='*80}")
    scan_bool = "True" if scan_mode == "scanned" else "False"

    run_name = "ckpt_pretrain_base"
    load_path = BASE_CHECKPOINTS.get(
        (model, scan_mode), f"{GCS_BASE}/{scan_mode}/pre_train/{model}/{run_name}/checkpoints/9/items"
    )
    hf_overrides = {"architectures": ["MaxTextForCausalLM"]}
    if "deepseek" in model:
      # An empty dict, not None. vLLM's DeepseekV4FP8Config.override_quantization_method only
      # claims the model when the config "is a dict whose quant_method is fp8", so {} fails that
      # test and leaves model_config.quantization as None, which tpu_inference maps to
      # UnquantizedConfig. None instead leaves the attribute present-but-None, which passes the
      # hasattr checks upstream and then dies on subscript.
      hf_overrides["quantization_config"] = {}

    cmd = [
        "python3",
        "-m",
        "maxtext.inference.vllm_decode",
        f"model_name={model}",
        f"vllm_hf_overrides={json.dumps(hf_overrides)}",
        f"hbm_utilization_vllm={HBM_UTILIZATION.get(model, DEFAULT_HBM_UTILIZATION)}",
        "prompt=Suggest some famous landmarks in London.",
        f"use_chat_template={model not in NO_CHAT_TEMPLATE}",
        # base.yml defaults weight_dtype to float32, which doubles every model in memory for no
        # benefit: these checkpoints are bfloat16 on disk (gemma2-2b reads 3.8GiB for 2B params)
        # and get upcast on load. mistral-7b then wants 26.98GiB against a 16GiB cap. Keeping
        # them bfloat16 matches the checkpoint and halves the footprint.
        "weight_dtype=bfloat16",
        "dtype=bfloat16",
        "enable_single_controller=True",
        f"scan_layers={scan_bool}",
    ]
    if load_path:
      cmd.append(f"load_parameters_path={load_path}")
    else:
      print(f"[INFO] No checkpoint for {model}/{scan_mode}; decoding with random (dummy) weights.")

    # Note: hf_access_token is NOT passed here explicitly.
    # The global OmegaConf resolver will handle the ${HF_TOKEN} interpolation 
    # by reading it from the environment variable inside the process.

    # vLLM builds its engine around a HuggingFace repo: decode_with_vllm passes tokenizer_path as
    # vLLM's `model`, and that is where it reads the config from. A local tokenizer asset has no
    # config beside it, so every case names the repo rather than an on-disk tokenizer file.
    hf_model_name = get_hf_id(model)
    cmd.extend([f"tokenizer_path={hf_model_name}", "tokenizer_type=huggingface"])
    if model != "deepseek4-tiny":
      # The tiny shape has no HuggingFace config of its own; pointing vLLM at V4-Flash's would
      # hand it the 284B dimensions.
      cmd.append(f"vllm_hf_config_path={hf_model_name}")

    if "llama3" in model:
      cmd.append("ici_tensor_parallelism=8")
      cmd.append("max_prefill_predict_length=1024")
      cmd.append("max_target_length=1024")
      cmd.append("attention=dot_product")
    elif "deepseek" in model:
      tp, ep = DEEPSEEK_PARALLELISM[model]
      cmd.append(f"ici_tensor_parallelism={tp}")
      cmd.append(f"ici_expert_parallelism={ep}")
      cmd.append("max_prefill_predict_length=1024")
      cmd.append("max_target_length=1024")
      # Required, not optional: types.py rejects the deepseek4 decoder block with any other
      # attention ("DeepSeek4 decoder block currently only supports dot_product attention").
      cmd.append("attention=dot_product")
      # DeepSeek is an MLA model, and vLLM refuses to build one unless attention is data
      # parallel: "MLA models require both the NEW_MODEL_DESIGN=1 environment variable to be
      # set and DP attention set via ... enable_dp_attention". vllm_decode.py already exports
      # NEW_MODEL_DESIGN=1, so this flag is the missing half.
      cmd.append("enable_dp_attention=True")
      # The DeepSeek HF repos advertise block-fp8 weights, which tpu_inference rejects outright
      # ("deepseek_v4_fp8 quantization method not supported"). MaxText builds its own bfloat16
      # weights and never reads the HF tensors, so the right answer is to tell tpu_inference not
      # to quantize at all -- qwix_utils.get_default_qwix_quantization_config returns early on
      # skip_quantization. Overriding hf_overrides.quantization_config to null does not work:
      # that code reads it behind a hasattr check and then subscripts it, so a present-but-None
      # value fails with "'NoneType' object is not subscriptable".
      cmd.append('vllm_additional_config={"skip_quantization": true}')
    elif "gemma3" in model:
      # NOTE: inert on this path. use_standalone_converter is only read by train_rl.py; setting it
      # here changes nothing, verified by running qwen2.5-1.5b scanned with and without it and
      # getting byte-identical output. Kept only because it matches the original script.
      cmd.append("use_standalone_converter=True")

    log_path = f"{LOCAL_LOGS}/{scan_mode}/{model}_vllm_decode.log"
    success = execute_command(cmd, log_path)
    if not success:
        all_passed = False

    status = "PASS" if success else "FAIL"
    print(f"[{status}] {model} | {scan_mode} | vllm_decode")
    with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow([model, scan_mode, "vllm_decode", run_name, status])

  print("\n" + "="*80)
  print("VLLM VALIDATION SUMMARY")
  print("="*80)
  with open(CSV_REPORT, "r") as f:
      print(f.read())
  print("="*80)

  if not all_passed:
      sys.exit(1)

if __name__ == "__main__":
  run_matrix()
