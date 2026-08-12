import sys
import jax
from maxtext import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train import train_compile

argv = list(sys.argv)
if not any("compile_topology=" in arg for arg in argv):
  argv.append("compile_topology=v6e-256")
if not any("compile_topology_num_slices=" in arg for arg in argv):
  argv.append("compile_topology_num_slices=1")
if not any("per_device_batch_size=" in arg for arg in argv):
  argv.append("per_device_batch_size=1")
if not any("skip_jax_distributed_system" in arg for arg in argv):
  argv.append("skip_jax_distributed_system=True")
config = pyconfig.initialize(argv)
topology_mesh = train_compile.get_topology_mesh(config)
shaped_train_args, shaped_train_kwargs, state_mesh_shardings, logical_annotations, model = train_compile.get_shaped_inputs(topology_mesh, config)
params_shardings, state_mesh_shardings = maxtext_utils.sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
input_state_mesh_shardings = maxtext_utils.sharding.build_zero1_input_state_mesh_shardings(config, state_mesh_shardings, params_shardings)

abstract_state = shaped_train_args[0]

abs_leaves = jax.tree_util.tree_leaves_with_path(abstract_state)
sh_leaves = jax.tree_util.tree_leaves_with_path(input_state_mesh_shardings)

print(f"Total abstract_state leaves: {len(abs_leaves)}")
print(f"Total input_state_mesh_shardings leaves: {len(sh_leaves)}")

mismatches = 0
for i, ((p1, l1), (p2, l2)) in enumerate(zip(abs_leaves, sh_leaves)):
    p1_str = "/".join(str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in p1)
    p2_str = "/".join(str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in p2)
    if p1_str != p2_str:
        mismatches += 1
        print(f"ORDER MISMATCH at index {i}: abstract={p1_str} | sharding={p2_str}")

print(f"TOTAL FLATTEN ORDER MISMATCHES: {mismatches}")
