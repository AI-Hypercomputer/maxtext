"""
Module for running a pure Tunix DPO reference training.
"""

import os
import jax
import optax
import datasets
from tunix.models.automodel import AutoModel, ModelSource
from tunix.sft.dpo.dpo_trainer import DPOTrainer, DPOTrainingConfig, DataInput
from tunix.generate.tokenizer_adapter import Tokenizer
from tunix.sft.metrics_logger import MetricsLoggerOptions

# 1. Hardware & Mesh Setup (v4-8 TPU)
# We use a 1x8 mesh for data and FSDP sharding.
print("Initializing JAX distributed...")
try:
  jax.distributed.initialize()
except RuntimeError as e:
  print(f"JAX distributed already initialized or failed: {e}")

mesh = jax.make_mesh((1, 1, jax.device_count()), ("data", "tp", "fsdp"))
print(f"Mesh created: {mesh}")

# 2. Model Loading (Pure Tunix / NNX)
model_id = "Qwen/Qwen2.5-1.5B"
print(f"Loading {model_id} from Hugging Face via AutoModel...")

with mesh:
  # Tunix AutoModel downloads and converts the model to NNX automatically.
  # We specify model_download_path to avoid issues with None return from download helper.
  model_download_path = "./quals/models"
  model, _ = AutoModel.from_pretrained(
      model_id, mesh=mesh, model_source=ModelSource.HUGGINGFACE, model_download_path=model_download_path
  )
  ref_model, _ = AutoModel.from_pretrained(
      model_id, mesh=mesh, model_source=ModelSource.HUGGINGFACE, model_download_path=model_download_path
  )

# 3. Dataset Loading (UltraFeedback Binarized)
print("Loading UltraFeedback Binarized dataset...")
# Using streaming=True for training to avoid high memory usage
train_ds = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True)
# Using non-streaming for eval to ensure it's not exhausted and is reliable
eval_ds = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", streaming=False)


def dataset_iterator(huggingface_ds, batch_size=8, limit=None):
  """Iterates over the HF dataset and yields Tunix DataInput batches."""
  batch = {"prompts": [], "chosen_responses": [], "rejected_responses": []}

  count = 0
  # Create an explicit iterator to handle both Dataset and StreamingDataset
  it = iter(huggingface_ds)

  while True:
    try:
      item = next(it)
      batch["prompts"].append(item["prompt"])
      batch["chosen_responses"].append(item["chosen"][-1]["content"])
      batch["rejected_responses"].append(item["rejected"][-1]["content"])

      if len(batch["prompts"]) == batch_size:
        yield DataInput(**batch)
        batch = {"prompts": [], "chosen_responses": [], "rejected_responses": []}
        count += 1
        if limit and count >= limit:
          break
    except StopIteration:
      break

  # Yield any remaining examples in the last partial batch
  if batch["prompts"]:
    yield DataInput(**batch)


# 4. Training Configuration
print("Configuring DPO Trainer...")
learning_rate = 1e-6
# Simple AdamW optimizer
tx = optax.adamw(learning_rate=learning_rate)

# Define local and GCS paths
LOCAL_LOG_DIR = "./quals/logs/tunix_ref"
GCS_BASE_PATH = "gs://igorts_europe/ttl=30d/dpo_quals/tunix_ref_ckpt"

# Ensure local log dir exists
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

config = DPOTrainingConfig(
    algorithm="dpo",
    beta=0.1,
    max_steps=200,
    eval_every_n_steps=50,
    max_prompt_length=512,
    max_response_length=512,
    checkpoint_root_directory=GCS_BASE_PATH,
    metrics_logging_options=MetricsLoggerOptions(log_dir=LOCAL_LOG_DIR, flush_every_n_steps=10),
)

# 5. Initialization and Training
tokenizer = Tokenizer(
    tokenizer_type="huggingface", tokenizer_path="Qwen/Qwen2.5-1.5B-Instruct", hf_access_token=os.environ.get("HF_TOKEN")
)

trainer = DPOTrainer(model=model, ref_model=ref_model, optimizer=tx, training_config=config, tokenizer=tokenizer)

print("🚀 Starting Pure-Tunix DPO Reference Run...")
print("This establishes the 'Gold Standard' baseline for alignment metrics.")

with mesh:
  # Pass the eval iterator directly (it was loaded non-streaming)
  trainer.train(dataset_iterator(train_ds), dataset_iterator(eval_ds, limit=5))

print("✅ Reference run complete.")
print(f"Check metrics in TensorBoard: tensorboard --logdir {GCS_BASE_PATH}")
print("Focus on 'rewards/accuracy' and 'rewards/margin'.")
