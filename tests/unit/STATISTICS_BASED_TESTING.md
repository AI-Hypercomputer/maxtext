# Statistics-Based Testing for Qwen3 Omni Layers

This document explains how to use the statistics-based testing feature that allows running tests without PyTorch dependencies.

## Overview

The Qwen3 Omni layer tests compare MaxText (JAX) implementations against PyTorch reference implementations. However, this requires PyTorch to be installed, which may not be available in all CI environments.

The statistics-based testing approach allows tests to run without PyTorch by comparing JAX outputs against precomputed statistical signatures (mean, std, max, min, median, and sample values) instead of full PyTorch forward passes.

## Environment Variables

### `USE_TORCH_FORWARD` (default: `true`)
Controls whether to run PyTorch forward passes or use statistics-based comparison.

- `true`: Run full PyTorch forward pass and compare outputs directly (requires PyTorch)
- `false`: Skip PyTorch and compare JAX outputs against hardcoded statistics (no PyTorch needed)

### `PRINT_STATS` (default: `false`)
When enabled, prints statistics in a format that can be copied into test code.

- `true`: Print statistics for hardcoding
- `false`: Normal test execution

## Usage Examples

### Running tests with PyTorch (default behavior)
```bash
# Run all tests with PyTorch
pytest tests/unit/qwen3_omni_layers_test.py

# Or explicitly enable
USE_TORCH_FORWARD=true pytest tests/unit/qwen3_omni_layers_test.py
```

### Running tests without PyTorch (CI mode)
```bash
# Run tests using precomputed statistics (no PyTorch needed)
USE_TORCH_FORWARD=false pytest tests/unit/qwen3_omni_layers_test.py
```

### Generating statistics for a new test

1. **Run test with statistics printing enabled:**
```bash
PRINT_STATS=true USE_TORCH_FORWARD=true pytest tests/unit/qwen3_omni_layers_test.py::TestQwen3OmniMoeVisionAttention::test_attention_output_matches_torch -v
```

2. **Copy the printed statistics:**
The test will print output like:
```python
  # Statistics for vision_attention_output:
  EXPECTED_STATS = {
    "shape": (16, 1024),
    "mean": -0.00019758939743041992,
    "std": 0.023577280715107918,
    "max": 0.11169242858886719,
    "min": -0.10485196113586426,
    "median": -0.000415802001953125,
    "first_5": [-0.017169952392578125, -0.020233154296875, ...],
    "last_5": [0.008697509765625, -0.002193450927734375, ...],
  }
```

3. **Add to your test:**
```python
def test_my_new_component(self):
  """Test my new component."""
  # Copy the EXPECTED_STATS here
  EXPECTED_STATS = {
    "shape": (16, 1024),
    # ... paste the rest
  }
  
  if USE_TORCH_FORWARD:
    # Full test with PyTorch
    torch_output = torch_model(...)
    jax_output = jax_model(...)
    
    if os.environ.get("PRINT_STATS", "false").lower() in ("true", "1", "yes"):
      print_statistics_for_hardcoding(jax_output, "my_component_output")
    
    assert_all_close_jax_torch(jax_output, torch_output, ...)
  else:
    # Lightweight test without PyTorch
    jax_output = jax_model(...)
    compare_with_statistics(jax_output, EXPECTED_STATS, name="my_component_output")
```

4. **Verify statistics-based testing works:**
```bash
USE_TORCH_FORWARD=false pytest tests/unit/qwen3_omni_layers_test.py::TestMyNewComponent::test_my_new_component -v
```

## Using the Helper Script

For generating statistics from saved arrays:

```python
# In your test or debugging code
import numpy as np
from maxtext.tests.utils.generate_test_statistics import generate_and_print_stats

# Save your output
jax_output = model(inputs)
np.save('/tmp/my_output.npy', np.array(jax_output))

# Generate statistics
generate_and_print_stats(jax_output, "my_output")
```

Or from command line:
```bash
python maxtext/tests/utils/generate_test_statistics.py /tmp/my_output.npy my_output
```

## Workflow for Updating Statistics

When you change the model implementation or weights, you may need to update the expected statistics:

1. **Run tests with both flags:**
```bash
PRINT_STATS=true USE_TORCH_FORWARD=true pytest tests/unit/qwen3_omni_layers_test.py -v
```

2. **Review the printed statistics** to ensure they look reasonable

3. **Update the EXPECTED_STATS** in the affected test methods

4. **Verify both modes work:**
```bash
# Test with PyTorch
USE_TORCH_FORWARD=true pytest tests/unit/qwen3_omni_layers_test.py -v

# Test without PyTorch
USE_TORCH_FORWARD=false pytest tests/unit/qwen3_omni_layers_test.py -v
```

## How It Works

### Full PyTorch Testing (USE_TORCH_FORWARD=true)
1. Creates PyTorch reference model
2. Copies weights from PyTorch to JAX
3. Runs forward pass on both models with same inputs
4. Compares full outputs element-wise

### Statistics-Based Testing (USE_TORCH_FORWARD=false)
1. Creates JAX model only (no PyTorch import needed)
2. Runs forward pass on JAX model
3. Extracts statistics (mean, std, max, min, median, sample values)
4. Compares statistics against expected values within tolerance

### Trade-offs

**Full PyTorch Testing:**
- ✅ Most thorough validation
- ✅ Catches numerical differences anywhere in output
- ❌ Requires PyTorch installation
- ❌ Slower execution
- ❌ Needs weight copying logic

**Statistics-Based Testing:**
- ✅ No PyTorch dependency
- ✅ Faster execution
- ✅ Can run in restricted CI environments
- ❌ Less thorough (could miss localized issues)
- ❌ Needs statistics to be pre-generated and kept up-to-date

## Best Practices

1. **Always generate statistics with PyTorch enabled first** - ensures the expected values come from a known-good comparison

2. **Run both modes in CI** - use PyTorch testing when available, fall back to statistics-based testing when not

3. **Update statistics when weights or model change** - outdated statistics will cause false failures

4. **Use reasonable tolerances** - statistics may vary slightly due to floating-point arithmetic; typical values: `rtol=1e-2, atol=1e-2`

5. **Test critical paths with PyTorch** - use statistics-based testing for CI convenience, but periodically validate with full PyTorch comparison

## Troubleshooting

### Test fails with "statistics don't match"
- Check if model or weights have changed
- Regenerate statistics with `PRINT_STATS=true USE_TORCH_FORWARD=true`
- Consider if tolerances need adjustment

### Test passes with PyTorch but fails without
- Ensure statistics were generated from the same test run
- Check that random seeds are set correctly
- Verify the same input data is used

### Statistics seem wrong
- Verify `USE_TORCH_FORWARD=true` when generating
- Check that weights are loaded correctly
- Ensure random seeds match between generation and usage

## Example Tests

The following tests have been updated to support both modes:
- `TestQwen3OmniMoeVisionAttention::test_attention_output_matches_torch`
- `TestQwen3OmniMoeVisionEncoderEndToEnd::test_vision_encoder_single_image`

Refer to these tests as templates for adding statistics-based testing to new tests.
