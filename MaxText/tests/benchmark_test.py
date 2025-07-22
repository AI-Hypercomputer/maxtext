from collections.abc import Sequence

from absl import app
import jax
from jax import numpy as jnp
import numpy as np
from MaxText.kernels import megablox as qwix_mblx
from MaxText.kernels import aqt_megablox as aqt_mblx
import datetime


def simple_timeit(f, *args, tries=10, task=None) -> float:
  """Simple utility to time a function for multiple runs."""
  # if trace_dir:
  #     return timeit_from_trace(f, *args, tries=tries, task=task, trace_dir=trace_dir)

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  for _ in range(tries):
    jax.devices()  # Force synchronization across devices
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())
  return outcomes_ms


def test_gmm_kernel(group_sizes, k, n, tiling, quant_dtype=None, qwix=True):
  """Smoke-test + correctness check for the grouped matrix-multiply kernel.

  For each group i, gmm should compute
      lhs[start_i:end_i, :]  @  rhs[i]
  and stitch the results back together along rows.

  Args:
    group_sizes: A 1d array with the size of each group.
    k: The number of columns of lhs and rows of rhs.
    n: The number of columns of rhs.
    tiling: 2-tuple that represents the tiling strategy.

  Returns:
    The output of the gmm kernel.
  """

  def f(group_sizes, k, n, tiling):
    group_sizes = jnp.array(group_sizes, dtype=jnp.int32)
    m = int(group_sizes.sum())

    key = jax.random.key(0)
    _, k1, k2 = jax.random.split(key, 3)

    lhs = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (group_sizes.size, k, n), dtype=jnp.bfloat16)

    # ---- run the Pallas kernel ----------------------------------
    if qwix:
      out = qwix_mblx.gmm(
          lhs,
          rhs,
          group_sizes,
          tiling=tiling,  # small tiles so the shapes above work
          interpret=True,  # avoids device-specific compilation in CI
          lhs_quantize_dtype=quant_dtype,
          rhs_quantize_dtype=quant_dtype,
      )
    else:
      out = aqt_mblx.gmm(
        lhs,
        rhs,
        group_sizes,
        tiling=tiling,  # small tiles so the shapes above work
        interpret=True,  # avoids device-specific compilation in CI
        lhs_quantize_dtype=quant_dtype,
        rhs_quantize_dtype=quant_dtype,
      )
    return out
  output = f(group_sizes, k, n, tiling)
  jax.block_until_ready(output)
  
  times = simple_timeit(f, group_sizes, k, n, tiling)
  
  jax.profiler.start_trace('gs://runner-maxtext-logs/mohitkhatwani_benchmark_test/' + str(quant_dtype) + '_qwix' + str(qwix) + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  output = f(group_sizes, k, n, tiling)
  jax.block_until_ready(output)
  jax.profiler.stop_trace()
  
  metrics = {
      "p50": np.percentile(times, 50),
      "p90": np.percentile(times, 90),
      "p95": np.percentile(times, 95),
      "p99": np.percentile(times, 99),
      "avg": np.mean(times),
  }
  return metrics


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  group_sizes = [3, 5]
  k = 6
  n = 4
  tiling = (1, 1, 1)
  
  aqt_metrics = test_gmm_kernel(group_sizes, k, n, tiling, quant_dtype=jnp.int8, qwix=False)
  qwix_metrics = test_gmm_kernel(group_sizes, k, n, tiling, quant_dtype=jnp.int8, qwix=True)
  # aqt_bf16_metrics = test_gmm_kernel(group_sizes, k, n, tiling, quant_dtype=None, qwix=False)
  bf16_metrics = test_gmm_kernel(group_sizes, k, n, tiling, quant_dtype=None, qwix=True)
  print(f"aqt_metrics {aqt_metrics}")
  print(f"qwix_metrics {qwix_metrics}")
  print(f"bf16_metrics {bf16_metrics}")


if __name__ == "__main__":
  app.run(main)
