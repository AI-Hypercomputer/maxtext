"""Common flags that are shared among metrics writers."""

from absl import flags

RUN_ID = flags.DEFINE_string("run_id", None, "The ID of the benchmark run.")
IS_TEST = flags.DEFINE_bool(
    "is_test",
    False,
    "True to write the metrics to the test project.",
)
ADDITIONAL_METRICS = flags.DEFINE_string(
    "additional_metrics",
    None,
    "The ad-hoc metrics which are only needed by specific benchmarks.",
)
