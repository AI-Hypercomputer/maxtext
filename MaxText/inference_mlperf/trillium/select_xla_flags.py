xla_flags_1 = {
    "xla_jf_auto_cross_replica_sharding": "False",
    "xla_tpu_enable_windowed_einsum_for_reduce_scatter": "False",
    "xla_tpu_enable_windowed_einsum_for_all_gather": "False",
    "xla_tpu_prefer_latch_optimized_rhs_layouts": "True",
}


def dump_flags(flags_set):
  flags_str = [f"--{k}={v}" for k, v in flags_set.items()]
  return " ".join(flags_str)


print(dump_flags(xla_flags_1))
