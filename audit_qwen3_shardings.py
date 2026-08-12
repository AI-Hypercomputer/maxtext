import os
import sys
import jax
from flax import nnx
from maxtext import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train import train_compile

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
    train_compile.train.train_step,
    data_sharding,
    input_state_mesh_shardings,
    model,
    config,
    params_shardings,
)

in_state_shardings = in_shard[0]
out_state_shardings = out_shard[0]

in_leaves = list(jax.tree_util.tree_leaves_with_path(in_state_shardings, is_leaf=lambda x: isinstance(x, (jax.sharding.Sharding, nnx.Variable))))
out_leaves = list(jax.tree_util.tree_leaves_with_path(out_state_shardings, is_leaf=lambda x: isinstance(x, (jax.sharding.Sharding, nnx.Variable))))

print(f"Total in_state leaves: {len(in_leaves)}")
print(f"Total out_state leaves: {len(out_leaves)}")

scanned_params = []
non_scanned_params = []
optimizer_leaves = []
rng_leaves = []

errors = []

for i, ((p1, l1), (p2, l2)) in enumerate(zip(in_leaves, out_leaves)):
  p1_str = "/".join(str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in p1)
  p2_str = "/".join(str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in p2)
  
  s1 = l1.value if hasattr(l1, "value") else l1
  s2 = l2.value if hasattr(l2, "value") else l2
  
  mk1 = getattr(s1, "memory_kind", None)
  mk2 = getattr(s2, "memory_kind", None)
  
  spec1 = getattr(s1, "spec", None)
  spec2 = getattr(s2, "spec", None)
  
  # Categorize leaf
  is_opt = any(k in p1_str for k in ("optimizer", "opt_state", "mu", "nu", "step", "count"))
  is_rng = any(k in p1_str for k in ("rngs", "dropout"))
  is_scanned = maxtext_utils.sharding.is_scanned_block_param_path(p1)
  
  if is_opt:
    optimizer_leaves.append((p1_str, mk1, mk2, spec1, spec2))
    if mk1 != "device" or mk2 != "device":
      errors.append(f"OPTIMIZER STATE ERROR at {p1_str}: in_mk={mk1}, out_mk={mk2} (expected device)")
  elif is_rng:
    rng_leaves.append((p1_str, mk1, mk2, spec1, spec2))
    if mk1 != "device" or mk2 != "device":
      errors.append(f"RNG LEAF ERROR at {p1_str}: in_mk={mk1}, out_mk={mk2} (expected device)")
  elif is_scanned:
    scanned_params.append((p1_str, mk1, mk2, spec1, spec2))
    if mk1 != "pinned_host" or mk2 != "pinned_host":
      errors.append(f"SCANNED PARAM ERROR at {p1_str}: in_mk={mk1}, out_mk={mk2} (expected pinned_host)")
  else:
    non_scanned_params.append((p1_str, mk1, mk2, spec1, spec2))
    if mk1 != "device" or mk2 != "device":
      errors.append(f"NON-SCANNED PARAM ERROR at {p1_str}: in_mk={mk1}, out_mk={mk2} (expected device)")
      
  # Sharding spec match check
  if spec1 != spec2:
    errors.append(f"SPEC MISMATCH at {p1_str}: in_spec={spec1}, out_spec={spec2}")

print("\n================ AUDIT SUMMARY ================")
print(f"1. Scanned Block Parameters (pinned_host): {len(scanned_params)} leaves")
for p, mk1, mk2, s1, s2 in scanned_params[:5]:
  print(f"   - {p}: in_mk={mk1}, out_mk={mk2}, spec={s1}")
print("   ...")

print(f"\n2. Non-Scanned Parameters (device): {len(non_scanned_params)} leaves")
for p, mk1, mk2, s1, s2 in non_scanned_params:
  print(f"   - {p}: in_mk={mk1}, out_mk={mk2}, spec={s1}")

print(f"\n3. Optimizer States (device): {len(optimizer_leaves)} leaves")
for p, mk1, mk2, s1, s2 in optimizer_leaves[:5]:
  print(f"   - {p}: in_mk={mk1}, out_mk={mk2}, spec={s1}")
print("   ...")

print(f"\n4. RNG / Dropout States (device): {len(rng_leaves)} leaves")
for p, mk1, mk2, s1, s2 in rng_leaves:
  print(f"   - {p}: in_mk={mk1}, out_mk={mk2}, spec={s1}")

print("\n================ VERIFICATION RESULT ================")
if not errors:
  print(f"ALL {len(in_leaves)} LEAVES VERIFIED PERFECTLY! ZERO ERRORS.")
else:
  print(f"FOUND {len(errors)} ERRORS:")
  for e in errors:
    print(f"  [ERROR] {e}")

print("=====================================================")
