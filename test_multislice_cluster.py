import jax
from cloud_tpu_cluster import SingleSliceGceTpuCluster



c = SingleSliceGceTpuCluster()

print(f"{c=}")
print(f"{c.is_env_present()=}")
print(f"{c.get_coordinator_address()=}")
print(f"{c.get_process_count()=}") # requires all workers
print(f"{c.get_process_id()=}") # ???