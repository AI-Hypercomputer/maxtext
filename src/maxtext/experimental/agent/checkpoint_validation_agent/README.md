# MaxText Checkpoint Validation Agent

This tool automates the validation of MaxText language models on TPU clusters. It uses Apache Airflow to provision the TPU hardware (via XPK) and a Python agent to execute the validation command.

## Quick Start: How to Run a Validation Test

To test a model, you do not need to edit any code. You simply trigger the Airflow DAG with a JSON configuration payload.

### Step 1: Create your JSON Payload
You must provide a single JSON file that tells Airflow where to run the test, and tells the MaxText engine how to configure the model. 

Copy this template and edit/update the values for your specific run:

```json
{
  "run_name": "my-qwen-test-run",
  "xpk_cluster_name": "tpu-v5p-cluster-central-a", 
  "checkpoint_gcs_path": "gs://my-bucket/models/qwen3-8b/0/items",
  "maxtext_model_name": "qwen3-8b",
  "maxtext_overrides": {
    "tokenizer_path": "Qwen/Qwen3-8B",
    "scan_layers": false,
    "max_target_length": 4096,
    "per_device_batch_size": 16.0
  }
}