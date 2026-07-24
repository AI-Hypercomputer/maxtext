# The Unified MaxText Checkpoint Architecture: Bridging Flax Linen and NNX

This document provides a technical comparison of the checkpoint architectures, serialization structures, and interoperability designs in MaxText, covering Flax Linen, Flax NNX, and the unified packed auxiliary layout.

## 1. Flax Linen Checkpoint Structure

In traditional Flax Linen, models are stateless blueprints, and training variables are tracked inside a standard `TrainState` container (as seen in [flax/training/train_state.py](https://github.com/google/flax/blob/main/flax/training/train_state.py)).

### Standard Mode (No Emergency)

Standard Flax Linen checkpointing organizes variables under independent, modular subdirectories inside the training step directory.

**On-Disk Directory Structure:**

- `step_1000/`
  - `items/` — Core model weights and optimizer parameters.
    - `params/` — Learnable model parameters (nested under "params" collection).
      - `params/` — Nested collection layer representing Linen weights.
    - `opt_state/` — Optimizer states (list of states with `None` placeholders).
    - `step` — Training step counter (serialized as `int32`).
  - `iter/` — Dataset progress state (optional, serialized when `dataset_type == "grain"` as seen in `save_checkpoint` in [`src/maxtext/common/checkpointing.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/checkpointing.py) and [`src/maxtext/common/grain_utility.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/grain_utility.py)).

**Serialized PyTree Schema:**
The `items` directory contains a single serialized PyTree of arrays. Note that the weights are nested under the double `"params" -> "params"` collection layer to conform to Linen's collection structure:

```json
{
    "params": {
        "params": {
            "decoder": {
                "layers_0": {
                    "self_attention": {
                        "query_proj": {
                            "kernel": jax.Array(...)
                        },
                        "key_proj": {
                            "kernel": jax.Array(...)
                        }
                    }
                }
            }
        }
    },
    "opt_state": [
        {
            "count": jax.Array(...),
            "mu": {"params": {...}},
            "nu": {"params": {...}}
        },
        None  # Placeholders for EmptyState elements in the Optax chain
    ],
    "step": jax.Array(1000, dtype=jnp.int32)
}
```

### Emergency & Multi-Tier Checkpointing Mode

Emergency managers (`EmergencyCheckpointManager` and `EmergencyReplicatorCheckpointManager` as seen in [`src/maxtext/common/emergency_checkpointing.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/emergency_checkpointing.py)) power both legacy emergency checkpointing (`enable_emergency_checkpoint`) and modern Multi-Tier Checkpointing (`enable_multi_tier_checkpointing`; see the [Emergency Checkpointing Guide](emergency_checkpointing.md) and [Multi-Tier Checkpointing Guide](multi_tier_checkpointing.md)).

Both emergency checkpointing and Multi-Tier Checkpointing share the exact same structural layout on disk. In both modes, Orbax operates under v0 experimental fast-path writers designed for rapid synchronous ramdisk writes during preemption. Because these managers only support outputting a single, consolidated PyTree payload per write, they do not create parallel composite directories (such as separate `iter/` subdirectories).

**On-Disk Directory Structure:**

- `step_1000/`
  - Contains a single consolidated folder holding the unified PyTree state payload.

**Serialized PyTree Schema:**
The schema is a single, flattened `state` payload containing parameters, steps, and optimizers:

```json
{
    "params": {
        "params": {...}
    },
    "opt_state": [...],
    "step": jax.Array(...)
}
```

## 2. Native NNX Checkpoint Structure

Flax NNX shifts from a functional, stateless paradigm to an object-oriented, stateful paradigm. In MaxText, the native training state is managed inside the stateful container **`TrainStateNNX`** (as seen in [src/maxtext/common/train_state_nnx.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/train_state_nnx.py)), which wraps both the model (`nnx.Module`) and its optimizer (`nnx.Optimizer`) as stateful, mutable sub-objects.

The model module contains parameters (`nnx.Param`), batch statistics (`nnx.BatchStat`), attention caches (`nnx.Cache`), and random number generator states (`nnx.RngState`) directly as mutable attributes (as seen in [flax/nnx/variablelib.py](https://github.com/google/flax/blob/main/flax/nnx/variablelib.py) and [flax/nnx/rnglib.py](https://github.com/google/flax/blob/main/flax/nnx/rnglib.py)).

### Default Serialization Layout

A native, unmodified `nnx.State` (as seen in [flax/nnx/statelib.py](https://github.com/google/flax/blob/main/flax/nnx/statelib.py)) serializes everything (including learnable parameters, active random generators, and dynamic caches) together as a single flat PyTree representation of attributes.

**On-Disk Directory Structure:**

- `step_1000/`
  - Contains a flat directory holding the full NNX State tree.

**Serialized PyTree Schema:**

```json
{
    "model": {
        "decoder": {
            "layers_0": {
                "self_attention": {
                    "query_proj": {
                        "kernel": jax.Array(...)
                    },
                    "key_proj": {
                        "kernel": jax.Array(...)
                    }
                }
            }
        },
        "dropout": {
            "rngs": {
                "default": {
                    "key": jax.Array(...)
                }
            }
        }
    },
    "optimizer": {
        "opt_state": {
            "0": {
                "count": jax.Array(...),
                "mu": {...},
                "nu": {...}
            }
        },
        "step": jax.Array(1000, dtype=jnp.uint32)
    }
}
```

- **Optimizer States:** Represent Optax chains as integer-keyed dictionaries (skipping empty states) rather than lists with `None` placeholders (as seen in [flax/nnx/training/optimizer.py](https://github.com/google/flax/blob/main/flax/nnx/training/optimizer.py)).
- **Step Counters:** Track iterations as 32-bit unsigned integers (`uint32`) instead of `int32`.
- **In-flight variables:** RNG keys, dropout counters, and activation caches are packed directly inside the model tree, polluting the clean weight parameters.

## 3. Bidirectional Interoperability and Conversion Mechanics

As MaxText transitions to Flax NNX, maintaining strict bi-directional interoperability between Flax Linen and NNX checkpoints is critical. This ensures **zero-downtime training resumption**, a **shared downstream ecosystem** (serving, decoding, quantization), and **unified weight conversion tooling** without code duplication.

To achieve this seamless interoperability, MaxText implements specialized mapping logic to bidirectionally convert parameters, optimizer tracking arrays, and training step counters. The bidirectional conversion functions implementing this parity layer are as seen in [src/maxtext/common/train_state_nnx.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/train_state_nnx.py).

### A. Enforcing Strict Parameter Structural Parity & Missing Weights Policy

The core foundation of this cross-framework compatibility is the **Strict Parameter Structural Parity Assumption**:

- **Core Assumption:** The learnable parameters (weights and biases) of the NNX model and the corresponding Linen model must share the exact same underlying dictionary structure, names, and tensor shapes once stripped of their respective metadata and auxiliary tracking variables. This alignment ensures that the same architecture-specific parameter mappings and layout conversion hooks can be reused across both models.
- **Strict Weight Validation (No Silent Failures):** MaxText enforces a strict missing weights policy during restoration. If a learnable parameter expected by the model architecture is genuinely missing from the checkpoint (or has a shape mismatch), the checkpointer explicitly raises a `ValueError` detailing the exact paths. This prevents the model from silently retaining untrained initialization values and suffering silent accuracy degradation upon resume.
- **Permissive Auxiliary Fallbacks:** In contrast to weights, if transient or framework-specific variables (like RNG states or dropout counters) are absent from a checkpoint—which occurs when an NNX run resumes from a pure Linen checkpoint—the framework handles this gracefully. During state alignment (as seen in [`replace_by_pure_dict`](https://github.com/google/flax/blob/main/flax/nnx/statelib.py)), any variables missing from the checkpoint remain as unmaterialized `ShapeDtypeStruct`s. The initialization caller then populates these safely with fresh initial values, allowing training to continue seamlessly.

### B. Extracting and Restoring NNX Model Parameters

To obtain clean, Linen-compatible parameter weights during saving, or to rebuild the stateful objects during loading, the framework converts between live PyTrees and unboxed arrays:

- **In-Memory Variable Splitting (Saving):** The framework separates model parameters from auxiliary trackers natively (as seen in `split_for_checkpoint` in [`src/maxtext/common/train_state_nnx.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/train_state_nnx.py)):
  ```python
  params, rng_state, batch_stats, caches, intermediates, rest = nnx.split_state(
      state, nnx.Param, nnx.RngState, nnx.BatchStat, nnx.Cache, nnx.Intermediate, ...
  )
  ```
  This ensures that the output is explicitly partitioned into `linen_state` (strictly learnable weights and optimizers), `aux` (RNG states, batch stats, and custom persistent variables), and `ephemeral` (caches and activations that are not checkpointed). Calling `to_pure_dict()` (as seen in [flax/nnx/statelib.py](https://github.com/google/flax/blob/main/flax/nnx/statelib.py)) unboxes these variables to extract their raw array values, discarding the `nnx.Param` class wrappers to create a clean nested dictionary on disk.
- **Native Permissive Fallback (`nnx.replace_by_pure_dict`) and Partial State Recovery (Loading):** MaxText relies entirely on native Flax NNX utilities to handle missing variables gracefully during restoration:
  - **The Loading Step:** When Orbax restores parameters from disk into memory, the result is a partial pure dictionary containing only the arrays that were saved on disk.
  - **The Rebinding Step:** The `_linen_items_to_nnx` utility (as seen in [`src/maxtext/common/checkpointing.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/checkpointing.py)) invokes `nnx.replace_by_pure_dict` (as seen in [flax/nnx/statelib.py](https://github.com/google/flax/blob/main/flax/nnx/statelib.py)) (e.g. `nnx.replace_by_pure_dict(linen_state, {"model": weights["model"]})` and `nnx.replace_by_pure_dict(aux_state, nnx_aux)`) to merge the restored dictionary into your model's abstract blueprint.
  - **Unified Restoration Entry Point (`load_state_if_possible`):** Both Linen and NNX restoration workflows are unified within `load_state_if_possible` (as seen in [`src/maxtext/common/checkpointing.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/checkpointing.py)). Rather than maintaining separate restore flows, NNX models project their target blueprint into Linen structure via `train_state_nnx.to_checkpoint_dict()`, reuse the standard composite Orbax load routines (including Grain iterator restoration), run structural parity verification (`_raise_on_weight_mismatch`), and map back via `_linen_items_to_nnx()`.
  - **Graceful Degradation:** Because `nnx.replace_by_pure_dict` is non-exhaustive, any keys absent from the restored dictionary (such as missing `nnx.RngState` or `nnx.BatchStat` trackers when resuming from a pure Linen checkpoint) are simply skipped. They remain as unmaterialized `ShapeDtypeStruct` placeholders on your in-memory model graph.
  - **Final Initialization:** The initialization caller (typically `maxtext_utils.init_initial_state`) then safely identifies these lingering placeholders and populates them natively with fresh, initial random streams or tracking zeros, allowing training to continue seamlessly without any custom imputation scripts.

### C. Bidirectional Optimizer State Conversion

Chained optimizer states (Optax momentum and velocity buffers) use different structural representations in each framework that must be mapped symmetrically:

- **NNX to Linen (as seen in `_opt_state_to_linen` in [`src/maxtext/common/train_state_nnx.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/train_state_nnx.py)):** NNX serializes chained optimizer steps as an integer-keyed dictionary (skipping empty steps). The conversion maps this dictionary into Linen's position-indexed list layout, inserting `None` placeholders for skipped steps. Inside each active step, the momentum parameters (`mu` and `nu`) are nested under an inner `"params"` key to match Linen's parameter collection schema.
- **Linen to NNX (as seen in `_opt_state_from_linen` in [`src/maxtext/common/train_state_nnx.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/train_state_nnx.py)):** Reverses the mapping by translating position-indexed lists back into an integer-keyed dictionary (filtering out `None` steps) and stripping out the nested `"params"` collection wrapper from the `mu` and `nu` momentum buffers to return standard flat arrays to NNX.

### D. Bidirectional Step Counter Conversion

The training iteration counter tracks execution step continuity across training sessions:

- **NNX to Linen:** The step counter is cast (as seen in `_cast_step` in [`src/maxtext/common/train_state_nnx.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/train_state_nnx.py)) from a 32-bit unsigned integer (`uint32`) to a standard 32-bit signed integer (`int32`) expected by Linen's `TrainState` counter.
- **Linen to NNX:** Reverses the cast by promoting the restored 32-bit signed step integer back to a 32-bit unsigned integer (`uint32`) to conform to the native NNX optimizer step tracking constraints.

### E. Scanned vs. Unscanned Layer Structure Parity

The `scan_layers` configuration fundamentally dictates whether layer weights are stacked together or kept separate in the serialized checkpoint on disk:

- **Scanned Layers (`scan_layers=True`):**
  Weights for all layers are stacked together into a single, combined tensor along a scanning axis (to optimize compilation via `jax.lax.scan`). In the checkpoint, they are represented under a single collective key (e.g., `layers`), and the arrays have an extra dimension corresponding to the number of layers.

  - *Example Path:* `decoder/layers/self_attention/query_proj/kernel` (Shape includes the `num_layers` dimension).

- **Unscanned Layers (`scan_layers=False`):**
  Weights for each layer are stored entirely separately. In the checkpoint, each layer gets its own distinct dictionary key ending with an underscore index (e.g., `layers_0`, `layers_1`), holding unstacked, individual tensors.

  - *Example Path:* `decoder/layers_0/self_attention/query_proj/kernel` (Standard unstacked shape).

  To maintain exact structural parity with Linen, Flax NNX avoids using `nnx.List` containers for these unscanned layers (which would create incompatible dot-index paths like `decoder.layers.0`) and instead registers them as directly named attributes (e.g., `decoder.layers_0`).

## 4. MaxText Unified Checkpointing: The Linen-Interoperable NNX Layout with Packed `nnx_aux`

To achieve perfect cross-framework compatibility while completely unifying standard and emergency checkpointer paths, MaxText uses the **Packed Auxiliary Structure** as the single, universal layout for both standard and emergency managers.

In this unified design, any dynamic, NNX-only auxiliary variables (RNG states, batch stats, custom routing tables) are **always packed directly inside the `"items"` dictionary** before being written to disk. This eliminates the need for separate composite directories on disk for standard runs, maintaining a single, consistent structure across all modes (as seen in [src/maxtext/common/checkpointing.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/checkpointing.py)).

### Unified On-Disk Directory Structure

Because the auxiliary state is embedded inside the `"items"` PyTree, the `"nnx_aux"` directory is written **under (inside)** the `"items"` folder on disk.

- `step_1000/`

  - `items/` — The items checkpointable (containing packed aux).
    - `params/` — Base model parameters (stripped of RNG keys).
      - `params/` — Nested collection layer representing Linen weights.
    - `opt_state/` — Optimizer states (list with `None` placeholders).
    - `step` — Step counter file (cast to `int32`).
    - `nnx_aux/` — Auxiliary state saved directly inside `items/`.
      - `dropout/`
        - `count` — Maintained RNG continuity.
      - `batch_stats/` — Batch normalization variables (if present).
  - `iter/` — Dataset iterator progress state (optional, written as a composite item when `config.dataset_type == "grain"` as seen in `save_checkpoint` in [`src/maxtext/common/checkpointing.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/checkpointing.py)).

- **For Standard Checkpointers (`ocp.training.Checkpointer`):** Orbax creates the parallel `"iter"` directory for dataset progress and writes `"items"` as a single composite checkpointable.

- **For Emergency & Multi-Tier Checkpointers (`enable_emergency_checkpoint` / `enable_multi_tier_checkpointing`):** Both operations are handled as seen in [`src/maxtext/common/emergency_checkpointing.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/emergency_checkpointing.py). Emergency and multi-tier checkpointing share the **exact same single-PyTree on-disk layout** (`enable_multi_tier_checkpointing` being the modern preferred flag over legacy `enable_emergency_checkpoint`). The manager saves only the unified `"items"` payload containing `"nnx_aux"` nested directly inside it, bypassing single-PyTree composite limitations.

### Unified Serialized PyTree Schema

The complete PyTree representation of the `"items"` payload saved to disk is:

```json
{
    "params": {
        "params": {
            "decoder": {
                "layers_0": {
                    "self_attention": {
                        "query_proj": {
                            "kernel": jax.Array(...)
                        },
                        "key_proj": {
                            "kernel": jax.Array(...)
                        }
                    }
                }
            }
        }
    },
    "step": jax.Array(1000, dtype=jnp.int32),  # Cast step counter
    "opt_state": [  # Optimizer states (list with None placeholders)
        {
            "count": jax.Array(...),
            "mu": {"params": {...}},
            "nu": {"params": {...}}
        },
        None
    ],
    "nnx_aux": {  # Packed auxiliary state (RNGs/BatchStat)
        "dropout": {
            "count": jax.Array(42)  # Maintained RNG continuity
        }
    }
}
```

### Cross-Framework Restoration Behavior

This unified packed solution achieves perfect interoperability during restoration across both frameworks:

- **Flax Linen Runs:** Linen trainers initialize a standard `TrainState` containing only `"step"`, `"params"`, and `"opt_state"` keys. Because `"nnx_aux"` is absent from the Linen template, Orbax target-guided loading simply ignores the `step_1000/items/nnx_aux/` folder on disk during load, loading parameters and optimizer states normally.
- **Flax NNX Runs:** The NNX checkpointer expects `"nnx_aux"` to be present in its `linen_abstract` target template. It restores the folder from `step_1000/items/nnx_aux/`, pops the branch, and merges the RNG stream back into the model state, guaranteeing RNG and dropout continuity across resumes.

## 5. External Format Conversion: Hugging Face & MaxText Parity

Beyond internal compatibility between Linen and NNX, MaxText provides robust tooling to convert models bidirectionally between the open-source Hugging Face ecosystem and the native MaxText formats (see the operational [Checkpoint Conversion Guide](convert_checkpoint.md) for full instructions). This conversion relies heavily on the strict structural parity rules defined earlier.

### A. Converting Hugging Face to MaxText (`to_maxtext.py`)

The `to_maxtext.py` script (as seen in [src/maxtext/checkpoint_conversion/to_maxtext.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/to_maxtext.py)) converts Hugging Face Safetensors checkpoints into MaxText-compatible Orbax checkpoints. Because of the Linen-NNX interoperability design, this script only needs to generate a **single, standardized Linen-layout output**.

- **Lazy Loading & Mapping:** It reads Hugging Face index files and lazily loads individual tensors into memory, applying architecture-specific layout hooks (such as `PARAM_MAPPING`).
- **Handling Scanned Layers:** If `scan_layers=True` is configured, the script dynamically stacks the un-nested Hugging Face layers along the designated `param_scan_axis` before writing them to Orbax.
- **Universal Output:** The resulting output is written to the standard `"items"` directory. Whether you intend to run a standard Linen inference engine or a stateful NNX training loop, both frameworks can seamlessly load this identical converted checkpoint.

### B. Converting MaxText to Hugging Face (`to_huggingface.py`)

Conversely, `to_huggingface.py` (as seen in [src/maxtext/checkpoint_conversion/to_huggingface.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/to_huggingface.py)) exports a MaxText checkpoint back to the open-source Safetensors format.

- **Dynamic Structure Detection:** The script dynamically inspects the checkpoint structure to support multiple formats simultaneously. It automatically handles unrolling standard Linen-layout scanned tensors back into independent layer slices.
- **Stripping Post-Training Wrappers:** If the script detects specialized NNX-SFT (`{"value": ...}` wrappers) or NNX-RL (top-level `"base"` wrappers) formats, it strips those layers on the fly and re-aligns them to standard MaxText conventions before exporting to Hugging Face.

### C. Dynamic On-the-Fly Safetensors Loading (`safetensors_dynamic`)

In addition to offline batch conversion via `to_maxtext.py`, MaxText can stream and load raw `.safetensors` files directly into training and evaluation jobs:

- **Zero-Preconversion Loading:** Setting `source_checkpoint_layout="safetensors_dynamic"` and pointing `load_parameters_path` to a directory of `.safetensors` files (or a remote Hugging Face repository ID such as `hf://meta-llama/Meta-Llama-3-8B`) causes `load_state_if_possible` to call `load_safetensors_dynamic_state()` (as seen in [`src/maxtext/checkpoint_conversion/utils/load_dynamic.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/utils/load_dynamic.py)).
- **Distributed Sharded Download & Parity Mapping:** The dynamic loader parses `.safetensors` index metadata, applies model-specific parameter name mapping (`PARAM_MAPPING`), and shards the tensor downloads directly onto TPU/CPU device meshes. This eliminates the storage and runtime overhead of generating an intermediate Orbax directory before training.
- **Native Orbax Safetensors Support:** Alternatively, setting `source_checkpoint_layout="safetensors"` uses Orbax v1's native `CheckpointLayout.SAFETENSORS` backend layout to write and restore raw `.safetensors` checkpoint files natively.

## 6. Post-Training Checkpoint Formats: NNX-SFT vs. NNX-RL

MaxText’s post-training pipelines—such as Supervised Fine-Tuning (as seen in [`train_sft.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/sft/train_sft.py)), Reinforcement Learning (as seen in [`train_rl.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/rl/train_rl.py)), and DPO (as seen in [`train_dpo.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/dpo/train_dpo.py))—build directly on top of pre-trained model weights.

At high level, checkpoint loading during post-training proceeds through two phases:

1. **Pre-Training Parameter Loading via `from_pretrained`:** Before fine-tuning begins, the training script invokes **`model_creation_utils.from_pretrained()`** (as seen in [src/maxtext/utils/model_creation_utils.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/model_creation_utils.py)) to load existing parameters from the directory specified by `load_parameters_path` (or converts a remote Hugging Face checkpoint on the fly if needed). At a high level, `from_pretrained` performs structure detection (supporting legacy Linen or native NNX formats), strips away transient runtime state (such as RNG keys and attention caches), aligns weights to the mesh sharding specs, and populates the base model.
2. **Tunix Handoff & Specialized Formats:** Once loaded, these parameters are passed to downstream orchestration frameworks like **Tunix**, which wraps the core model into adapters to calculate loss gradients, track actor-critic states, or run RL generation loops.

As Tunix periodically saves checkpoint steps during fine-tuning or RL training, the wrapper structures are preserved on disk, giving rise to two distinct post-training checkpoint formats:

### A. NNX-SFT (Standard SFT Format)

The **NNX-SFT** format represents the state of a standalone MaxText NNX `Transformer` model (such as an Instruct model trained as seen in [`train_sft.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/sft/train_sft.py)). Because there are no outer wrappers around the core model, the parameters (like `decoder` and `token_embedder`) are stored directly at the root of the checkpoint. At each leaf node, weights are wrapped in a `{"value": ...}` dictionary:

```json
{
    "decoder": {
        "layers_0": {
            "self_attention": {
                "query_proj": {
                    "kernel": {
                        "value": jax.Array(...)
                    }
                }
            }
        }
    },
    "token_embedder": {
        "embedding": {
            "value": jax.Array(...)
        }
    }
}
```

### B. NNX-RL (Reinforcement Learning / Policy Format)

The **NNX-RL** format is generated when a model is trained using Reinforcement Learning (as seen in [`train_rl.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/rl/train_rl.py)). During RL, the model is wrapped within a `TunixMaxTextAdapter` class (as seen in [src/maxtext/integration/tunix/tunix_adapter.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/integration/tunix/tunix_adapter.py)). Because the adapter assigns the base model to an attribute named `self.base`, the serialized state wraps the entire model state tree under a top-level `"base"` key:

```json
{
    "base": {
        "decoder": {
            "layers_0": {
                "self_attention": {
                    "query_proj": {
                        "kernel": {
                            "value": jax.Array(...)
                        }
                    }
                }
            }
        },
        "token_embedder": {
            "embedding": {
                "value": jax.Array(...)
            }
        }
    }
}
```

### Downstream Usage and Extraction

To reuse these specialized post-training formats (NNX-SFT and NNX-RL) for downstream tasks—such as running inference or converting the weights back to Hugging Face format—MaxText employs dynamic detection and unwrapping mechanisms.

Because the downstream tools expect a standard, uniform parameter structure, MaxText automatically strips away the `{"value": ...}` wrappers and the `"base"` nesting on the fly during loading:

- **During Inference and Decoding:** The parameter loading function dynamically inspects the checkpoint's metadata to determine its structure (as seen in [`decode.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/inference/decode.py)). It extracts the target dictionary (from `"base"` if RL, or the root if SFT) and maps over every leaf node to extract the raw arrays from their `{"value": ...}` dictionaries, restoring the arrays into the standard, clean parameter PyTree that the base MaxText model expects.
- **During Weight Export and Conversion:** The conversion utility (as seen in [`to_huggingface.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/to_huggingface.py) and [`to_maxtext.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/to_maxtext.py)) dynamically inspects the checkpoint dictionary, extracting parameters from inside the `"base"` wrapper if detected. After stripping the wrappers, the extraction function flattens the parameters and aligns them to the standard MaxText-Linen naming conventions (such as `params-decoder-decoder_norm-scale`), allowing the script to reuse the exact same parameter mapping logic for pre-training, SFT, and RL checkpoints.
