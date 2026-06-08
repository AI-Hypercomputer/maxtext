# Phase 1: Architecture Investigation for HF and MaxText

## Objective
Conduct a detailed analysis of model specifics for both Hugging Face and MaxText utilizing `graph_tracing.py` to establish a project roadmap.

## Result
Generation of two JSON files documenting the checkpoint architectures of the models. These files will map out the dimensions, names, and hierarchical structure of parameters, ensuring a thorough understanding before any mapping begins.


## Command to run scripts
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu /home/yixuannwang_google_com/maxtext_venv/bin/python \
src/maxtext/experimental/agent/ckpt_conversion_agent/utils/graph_tracing.py \
--model_name gemma3-4b --scan_layers`