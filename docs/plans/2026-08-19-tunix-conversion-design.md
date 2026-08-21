# Tunix Checkpoint Conversion on Load

## Overview

Post-training currently outputs checkpoints in a "Tunix" layout (featuring `model_params` and `optimizer_state`), which is quite different from what MaxText's pre-training load weight structure expects (Linen layout with `items/params`, `items/opt_state`, etc.).

This design implements a mechanism to automatically detect and convert Tunix-formatted checkpoints on the fly during the pre-training load process, ensuring a smooth transition back to pre-training without requiring manual checkpoint conversion.

## Architecture & Auto-Detection

The conversion logic will be integrated into the pre-training loader, specifically within `src/maxtext/common/checkpointing.py` in the `load_state_if_possible` and `load_params_from_path` functions.

We will use an auto-detection mechanism: when loading a checkpoint, the system will inspect the directory structure (or metadata) of the given path. If it contains a `model_params` key/directory, the system will flag the checkpoint as being in the Tunix layout.

## Conversion Logic Flow

When a Tunix checkpoint is detected during the loading phase:

1. **Mapping:** The `restore_args` will be dynamically modified to instruct Orbax to read weights from `model_params` (and `optimizer_state` if doing a full restore) instead of the standard `items/params` collection.
2. **Reshaping:** Once Orbax restores the PyTree into memory, the Tunix tree structure will be reshaped to match the standard `TrainState` layout. This includes:
   - Restoring the Linen `params` collection level (wrapping weights inside a `params` dict).
   - Stripping the `inject_hyperparams` shell that `optax` wraps a scheduled optimizer in (ensuring `mu` and `nu` are at the expected depth).
   - Dropping the `adapter` layer that a LoRA wrapper typically adds.
3. **Integration:** After reshaping, the standard MaxText pre-training code will receive the exact `TrainState` structure it expects, remaining completely agnostic to the fact that the original on-disk checkpoint was in the Tunix layout.
