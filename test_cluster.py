import jax
from jax._src.clusters.cloud_tpu_cluster import TpuCluster



c = TpuCluster()

print(f"{c=}")
print(f"{c.is_env_present()=}")
print(f"{c.get_coordinator_address()=}")
print(f"{c.get_process_count()=}")
print(f"{c.get_process_id()=}")