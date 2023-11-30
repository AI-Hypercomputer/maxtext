import jax
#from jax._src.clusters.cloud_tpu_cluster import SingleSliceGceTpuCluster
from cloud_tpu_cluster import SingleSliceGceTpuCluster
from cloud_tpu_cluster import MultisliceGceTpuCluster
from cloud_tpu_cluster import GkeTpuCluster
import time

time.sleep(10)

c = GkeTpuCluster()

print(f"{c=}", flush=True)
print(f"{c.is_env_present()=}", flush=True)
print(f"{c.get_coordinator_address()=}", flush=True)
print(f"{c.get_process_count()=}", flush=True)
print(f"{c.get_process_id()=}", flush=True)


print("Calling jdi...", flush=True)
jax.distributed.initialize(
    coordinator_address = c.get_coordinator_address(),
    num_processes = c.get_process_count(),
    process_id = c.get_process_id()
)
print("JDI successful!!!", flush=True)