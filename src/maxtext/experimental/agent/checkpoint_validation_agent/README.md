# Checkpoint Validation Agent

This agent validates MaxText checkpoints to ensure they are compatible with the inference engine.

## How to run
Set the checkpoint directory:
`export MAXTEXT_CHECKPOINT_DIR="/path/to/your/maxtext/folder"`

Run the validation:
`python3 -m src.maxtext.experimental.agent.checkpoint_validation_agent.main <model_key>`

Example:
`python3 -m src.maxtext.experimental.agent.checkpoint_validation_agent.main qwen3-4b-unscanned`

## Adding a new model
1. Open `model_registry.py`.
2. Add the model metadata to the `MODEL_REGISTRY` dictionary.