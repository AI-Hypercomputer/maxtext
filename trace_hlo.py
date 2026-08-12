import sys
import jax
from maxtext import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train import train_compile
from maxtext.trainers.pre_train import train

config = pyconfig.initialize(sys.argv)
topology_mesh = train_compile.get_topology_mesh(config)
shaped_train_args, shaped_train_kwargs, state_mesh_shardings, logical_annotations, model = train_compile.get_shaped_inputs(topology_mesh, config)

params_shardings, state_mesh_shardings = maxtext_utils.sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
input_state_mesh_shardings = maxtext_utils.sharding.build_zero1_input_state_mesh_shardings(config, state_mesh_shardings, params_shardings)
data_sharding = maxtext_utils.sharding.get_input_data_sharding(config, topology_mesh)

(
    func_to_compile,
    in_shard,
    out_shard,
    static_argnums,
    donate_argnums,
) = maxtext_utils.get_functional_train_with_signature(
    train.train_step,
    data_sharding,
    input_state_mesh_shardings,
    model,
    config,
    params_shardings,
)

jitted = jax.jit(
    func_to_compile,
    in_shardings=in_shard,
    out_shardings=out_shard,
    static_argnums=static_argnums,
    donate_argnums=donate_argnums,
)

lowered = jitted.lower(*shaped_train_args, **shaped_train_kwargs)
hlo_module = lowered.as_text()

# Search for CopyD2H / host offload instructions in HLO
lines = hlo_module.splitlines()
print(f"Total HLO lines: {len(lines)}", flush=True)

copy_host_lines = [l for l in lines if "copy" in l.lower() or "pinned_host" in l.lower() or "host" in l.lower()]
print(f"Found {len(copy_host_lines)} HLO lines referencing host / copy:", flush=True)
for l in copy_host_lines[:30]:
  print("  ", l, flush=True)
