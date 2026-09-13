# JAX Layer Implementation & Verification

This guide covers Phase 2 of the MaxText model bring-up workflow: implementing layers in JAX/Flax and performing layer-wise unit testing against HuggingFace (PyTorch) implementations.

## 1. Layer Implementation Strategy

- **Maximize Architectural Reuse**: Before writing JAX code from scratch, examine what layers already exist in MaxText that can be reused or configured to form the target layer. For example, inspect `src/maxtext/layers/attentions.py` to see if attention block mechanisms (MHA, GQA, RoPE) can be re-used or adapted.
- Review existing layers in `src/maxtext/layers/` (e.g. `decoders.py`, `attentions.py`, `moe.py`).
- Inherit or extend standard MaxText structures (like scanned layers blocks) whenever possible.
- If a new model architecture introduces new layers, implement them in JAX. Maintain Flax styling where layer parameters are managed in a PyTree.
- **Autoregressive Generation Support**: Ensure custom decoder and attention layers support autoregressive generation (`decode=True` in Flax/linen), including KV cache management, position index handling, and causal attention masks during single-step token generation.

> [!IMPORTANT]
> **Direct TPU VM Development**:
> Edit, run, and verify code directly within the MaxText checkout on the TPU VM where JAX and TPU drivers are configured.

### 1.1 Multimodal Preprocessing Implementation (If applicable)

If the model is multimodal (e.g., VLM or ALM) and requires specific preprocessing:

1.  **Create Model-Specific Processor**: Create `src/maxtext/multimodal/processor_{model_name}.py`.
    -   Implement functions for resizing, normalization, and padding (e.g., padding video/image to static max grids if configured).
    -   Implement token expansion (e.g., expanding `<|image_pad|>` to the correct number of tokens based on grid/merge sizes).
2.  **Register in Central Router**: Register these functions in `src/maxtext/multimodal/processor.py` by adding your model to the conditional blocks in:
    -   `preprocess_mm_data`
    -   `preprocess_image_for_training`
    -   `get_image_offsets`
    -   `reformat_prompt`
    -   `reformat_response`
    -   `prepare_text_for_image_fusion`
    -   `get_dummy_image_shape_for_init`
    -   `get_bidirectional_mask_vision`
    -   `get_bidirectional_mask_audio` (if audio is supported)

## 2. Write Layer-Wise Unit Tests
Write layer-wise tests in a new file under `tests/unit/{model_name}_layers_test.py`.
For each class of layers, define a test that compares the JAX output directly to PyTorch.

> [!TIP]
> **Testing Reference**:
> Refer to existing layer tests such as `tests/unit/gemma3_layers_test.py` or `tests/unit/gemma4_layers_test.py` as codebase references. They provide excellent best-practice examples on comparing JAX layers against reference PyTorch equivalents.

### Unit Test Best Practices:

1. **Use Mini-Layer Dimensions**: Use small parameters to make tests fast and runnable on CPU:
   - `batch_size = 2`, `seq_len = 4`, `hidden_dim = 128`, `num_heads = 4`.
2. **Generate Fixed Random Inputs**: Share the same random input tensors between PyTorch and JAX:
   - Create inputs using `numpy` or standard seed, then cast to PyTorch `torch.Tensor` and JAX `jnp.ndarray`.
3. **Explicitly Copy PyTorch Weights to Flax PyTree**:
   - Retrieve weights from the PyTorch model layer.
   - Convert them using `.detach().numpy()`.
   - Remember to **TRANSPOSE** linear weights! PyTorch shape is `(out_features, in_features)`, JAX/Flax shape is `(in_features, out_features)`.
   - Map them exactly to your Flax model parameter dictionary.

### Concrete Unit Test Template:

```python
import jax
import jax.numpy as jnp
import numpy as np
import torch
from maxtext.src.maxtext.layers import decoders as maxtext_decoders

def to_jax(pt_tensor):
  return jnp.array(pt_tensor.detach().numpy())

def test_layer_correctness():
  # 1. Setup dimensions
  batch, seq, hidden_dim = 2, 4, 128
  
  # 2. Initialize PyTorch Reference Layer
  pt_layer = PyTorchReferenceLayer(hidden_dim)
  
  # 3. Initialize MaxText (Flax/JAX) Layer
  jax_layer = maxtext_decoders.NewJaxLayer(hidden_dim)
  
  # 4. Generate fixed random inputs
  np_input = np.random.randn(batch, seq, hidden_dim).astype(np.float32)
  pt_input = torch.tensor(np_input)
  jax_input = jnp.array(np_input)
  
  # 5. Run PyTorch forward pass
  pt_output = pt_layer(pt_input)
  
  # 6. Initialize JAX variables and replace weights
  init_rng = jax.random.PRNGKey(0)
  variables = jax_layer.init(init_rng, jax_input)
  params = variables["params"]
  
  # WEIGHT COPYING & TRANSPOSITION GOTCHA:
  # PyTorch weights (out_features, in_features) must be transposed to JAX (in_features, out_features)
  params["query"]["kernel"] = to_jax(pt_layer.q_proj.weight).transpose()
  params["out_proj"]["kernel"] = to_jax(pt_layer.out_proj.weight).transpose()
  
  # 7. Run JAX forward pass
  jax_output = jax_layer.apply({"params": params}, jax_input)
  
  # 8. Compare outputs using allclose
  np.testing.assert_allclose(
      to_jax(pt_output),
      jax_output,
      rtol=1e-3,
      atol=0.05,
      err_msg="JAX output does not match PyTorch reference!"
  )
```

### 2.1 Multimodal Preprocessing Unit Tests (If applicable)

For multimodal models, write a unit test to verify that the MaxText preprocessing pipeline matches the HuggingFace reference implementation.

-   **Reference**: `tests/unit/qwen3_omni_layers_test.py` (e.g. `TestQwen3OmniPreprocessing`).
-   **Key Verifications**:
    1.  **Prompt Formatting**: Verify `reformat_prompt` output matches HF.
    2.  **Tokenization & Padding**: Verify input IDs (including inserted pad tokens) match HF.
    3.  **Feature Values**: Verify preprocessed pixel values (images/videos) and audio features are numerically close to HF outputs (use `np.allclose` with appropriate tolerance, e.g., `1e-2` or `5e-2` for video).

#### Preprocessing Test Template:

```python
class TestModelNamePreprocessing(unittest.TestCase):
  def setUp(self):
    self.config = pyconfig.initialize(
        ["", base_config_path],
        model_name="{model_name}",
        tokenizer_type="huggingface",
        tokenizer_path="{hf_model_id}",
        image_path=test_image_path,
    )

  def test_preprocess_matches_hf(self):
    # 1. Run MaxText Preprocessor
    mt_outputs = mm_processor.preprocess_mm_data(self.config)
    mt_prompt = mm_processor.reformat_prompt(...)
    # ... tokenize and prepare text ...

    # 2. Run HF Preprocessor
    hf_processor = AutoProcessor.from_pretrained("{hf_model_id}")
    hf_outputs = hf_processor(text=hf_prompt, images=image, return_tensors="pt")

    # 3. Assertions
    self.assertEqual(mt_prompt, hf_prompt)
    np.testing.assert_array_equal(mt_input_ids, hf_input_ids)
    np.testing.assert_allclose(mt_pixel_values, hf_pixel_values, rtol=1e-2, atol=1e-2)
```

## 3. Define the Model Config (.yml)
Create a config yml file for the new model in `src/maxtext/configs/models/{model_name}.yml`.

- Define core model structures: `vocab_size`, `base_emb_dim`, `base_num_decoder_layers`, `base_num_query_heads`, `base_num_kv_heads`, `head_dim`.
- Specify target decoder block custom name (e.g. `decoder_block: "{model_name}"`).
- Set custom hyperparams if needed (e.g., `inhomogeneous_layer_cycle_interval`).

## 4. Run a Quick E2E Decode Check with Mini-Config on TPU VM
Once JAX layers are implemented and the config yml is defined, run a fast end-to-end structural check using `decode.py` directly on the TPU VM.

- **Golden Command Format**: Format python module commands using `python3 -m maxtext.inference.decode`.
- **Use a Mini-Config**: Set `base_num_decoder_layers=1` inside your run command to bypass full model size HBM limits.
- **Use a Random Checkpoint**: Rather than using a fully converted checkpoint, run with random weights.
- This is extremely fast, verifies JAX code trace, shapes, and autoregressive generation support without HBM OOM issues.

Example command:

```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.inference.decode   src/maxtext/configs/base.yml   model_name={model_name}   base_num_decoder_layers=1   force_random_weights=true   per_device_batch_size=1   max_target_length=64
```

## 5. Handover Document Update
Update `{model_name}_handover.md` in the MaxText root directory:
- Log layer implementation details (e.g., reuse of attention blocks, custom layer definitions).
- Record layer unit test results and tolerances (`rtol`, `atol`).
- Document any shape or numerical bugs found and how they were resolved.
- Set Phase 2 status to `DONE` and Phase 3 to `IN PROGRESS`.
