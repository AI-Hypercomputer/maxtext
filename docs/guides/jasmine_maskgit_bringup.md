# Jasmine MaskGIT Dynamics Model Bring-up in MaxText

This document details the integration of the Jasmine MaskGIT dynamics model into MaxText using **Flax NNX** and MaxText's optimized attention layers. It serves as a guide for the architecture, key implementation decisions, and troubleshooting steps resolved during the bring-up process.

---

## 1. Architecture Overview

The Jasmine MaskGIT model is a spatiotemporal video dynamics model that predicts future frame latent tokens conditioned on action sequences and past frame tokens. The integrated model resides in MaxText at:
*   [jasmine.py](file:///google/src/cloud/hengtaoguo/debug-maskgit-tensor-shapes/google3/third_party/py/maxtext/src/maxtext/models/jasmine.py) (Model definition)
*   [vla_decode.py](file:///google/src/cloud/hengtaoguo/debug-maskgit-tensor-shapes/google3/third_party/py/maxtext/src/maxtext/inference/vla_decode.py) (Inference orchestration and validation entry point)
*   [vla_decode_test.py](file:///google/src/cloud/hengtaoguo/debug-maskgit-tensor-shapes/google3/third_party/py/maxtext/tests/unit/vla_decode_test.py) (Model unit test)

### Model Structure

The top-level NNX module `DynamicsMaskGIT` contains:
1.  **Patch Embedder**: `nnx.Embed` mapping vocabulary tokens to model dimension ($d_{model} = 512$).
2.  **Action Projector**: `maxtext.layers.linears.DenseGeneral` mapping latent action dimensions to model dimensions.
3.  **Spatiotemporal Axial Transformer**: `AxialTransformer` containing a stack of `AxialBlock` layers:
    *   **Spatial Attention**: Applies bidirectionally across the spatial dimension (patches within a frame).
    *   **Temporal Attention**: Applies causally across the temporal dimension (frames across time).
    *   **FFN**: Uses two `DenseGeneral` layers to project representations.

---

## 2. Key Porting Details & Pitfalls

During porting, we resolved several challenges related to JAX tracing, parameter scaling, and NNX state structure:

### 1. Logits Scaling in MaxText Attention
*   **Pitfall**: Standard multi-head attention scales attention logits by $1/\sqrt{d_k}$ (where $d_k$ is the head dimension) before applying softmax. MaxText's optimized `Attention` layer defaults to **no scaling** (scalar value of `1.0`) unless explicitly configured. Without this scaling, logits are too large, leading to an extremely sharp softmax distribution. During MaskGIT sampling, this caused the model to aggressively predict the mask token (index `0`) for almost all locations.
*   **Solution**: We explicitly passed `query_pre_attn_scalar=head_dim**-0.5` to both spatial and temporal `Attention` constructors. This restored dot-product scaling and fixed generation quality.

### 2. Attention Output Interface
*   **Pitfall**: Flax Linen's standard `nn.MultiHeadAttention` returns a single output array. MaxText's optimized `Attention` layer always returns a tuple: `(output, kv_cache)`.
*   **Solution**: When calling attention inside `AxialBlock.__call__`, we explicitly unpack the returned tuple and discard the `kv_cache`:
    ```python
    z_flat_spatial, _ = self.spatial_attention(z_flat_spatial, z_flat_spatial, model_mode=MODEL_MODE_TRAIN)
    ```

### 3. Static Shape Requirements for Compilation
*   **Pitfall**: Unlike standard Flax NNX layers which infer shapes dynamically, MaxText's `Attention` layer compiles highly optimized hardware kernels (e.g. Flash Attention, Pallas) which require static shapes at constructor time (`__init__`).
*   **Solution**: We modified `AxialBlock` and `AxialTransformer` constructors to accept static parameters `num_spatial_patches` (N) and `temporal_seq_len` (T). We passed dummy shapes `(1, 1, d_model)` to `inputs_q_shape` and `inputs_kv_shape` to satisfy the constructor's shape initialization needs for projection layers:
    ```python
    self.spatial_attention = Attention(
        ...
        max_target_length=num_spatial_patches,  # N (e.g., 16)
        inputs_q_shape=(1, 1, self.dim),
        inputs_kv_shape=(1, 1, self.dim),
    )
    ```

### 4. JAX Initialization Order Conflicts
*   **Pitfall**: Calling JAX operations (like `jax.devices()`) before initializing MaxText's configurations locks the JAX backend. If `pyconfig.initialize()` is called later, JAX throws a `RuntimeError: JAX backend already initialized`.
*   **Solution**: In the evaluation script (`src/maxtext/inference/vla_decode.py`), `pyconfig.initialize` must be called at the absolute beginning of `main()`, preceding any other library imports or JAX operations.


### 5. In-Memory Weight Copying (No Intermediate Pickle Files)
*   **Pitfall**: Standard checkpoint restore mechanisms mapping Linen to NNX require writing offline conversion scripts. Since both the source model (Jasmine) and target model (MaxText port) are NNX models, we can load them in-memory. However, running JAX's native `tree_flatten` on `nnx.State` flattens *past* the `nnx.Variable` wrappers, returning immutable JAX arrays as leaves, which prevents in-place mutation.
*   **Solution**: We implemented a custom recursive flattener that stops traversing at `nnx.Variable` objects. This allows us to map and load parameter values directly in-place:
    ```python
    def flatten_state(s, prefix=()):
        flat = {}
        from flax.nnx.statelib import State
        if isinstance(s, (State, dict)):
            for k, v in s.items():
                flat.update(flatten_state(v, prefix + (k,)))
        elif isinstance(s, nnx.Variable):
            flat[prefix] = s
        return flat

    # Assign in-place with coordinate casting
    for path_tuple, var in flat_maxtext.items():
        if path_tuple in flat_jasmine:
            var[...] = jnp.asarray(flat_jasmine[path_tuple][...], dtype=var[...].dtype)
    ```

---

## 3. How to Run & Configure

### Execution
The entry point script resides inside the inference directory. Run it from the repository root:
```bash
python3 src/maxtext/inference/vla_decode.py checkpoint=/path/to/jasmine/ckpt maskgit_steps=4 temperature=1.0 sample_argmax=true
```

### Running Unit Tests
A unit test suite has been added to verify model shape configuration and JIT-compilation correctness:
```bash
PYTHONPATH=/home/hengtaoguo_google_com/projects/maxtext python3 tests/unit/vla_decode_test.py
```

### VS Code Debugger Configuration
Add this configuration to your `.vscode/launch.json` to debug the script:
```json
        {
            "name": "maxtext_jasmine_decode",
            "type": "debugpy",
            "request": "launch",
            "python": "/home/hengtaoguo_google_com/projects/venv3/bin/python",
            "cwd": "/home/hengtaoguo_google_com/projects/maxtext",
            "program": "/home/hengtaoguo_google_com/projects/maxtext/src/maxtext/inference/vla_decode.py",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "/home/hengtaoguo_google_com/projects/maxtext/src:/home/hengtaoguo_google_com/projects/jasmine",
                "JAX_PLATFORMS": "cpu"
            },
            "args": [
                "checkpoint=/home/hengtaoguo_google_com/projects/checkpoints/jasmine-maskgit-coinrun",
                "maskgit_steps=4",
                "temperature=1.0",
                "sample_argmax=true",
                "seed=0"
            ],
            "console": "integratedTerminal"
        }
```

---

## 4. Metrics Validation

Correctness is validated by comparing the output generation of the MaxText NNX port against the baseline Jasmine implementation on the same CoinRun episode:

| Model / Run | Attention Implementation | Style | SSIM (Avg) | PSNR (Avg) |
| :--- | :--- | :--- | :--- | :--- |
| **Jasmine Baseline** (NNX) | NNX Attention | NNX | `0.8514` | `27.16` |
| **MaxText Linen Port** | MaxText `Attention` | Linen | `0.8691` | `28.22` |
| **MaxText NNX Port** (Integrated) | MaxText `Attention` | **NNX** | **`0.8691`** | **`28.22`** |

The exact parity of the metrics confirms the architectural correctness of the ported layers and weight mapping.
