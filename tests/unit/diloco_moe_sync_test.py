
import unittest
import chex
from flax.experimental import nnx
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from maxtext.configs.pyconfig import initialize_pydantic
from maxtext.trainers.diloco import diloco
from tests.utils.test_helpers import get_test_config_path


class MoEModel(nnx.Module):
  """Simple MoE model for testing router synchronization."""

  def __init__(self, *, rngs: nnx.Rngs):
    self.dense = nnx.Linear(
        2,
        2,
        kernel_init=nnx.initializers.constant(jnp.array([[1.0, 0.0], [0.0, 1.0]])),
        rngs=rngs,
    )
    self.gate = nnx.Linear(
        2,
        2,
        kernel_init=nnx.initializers.constant(jnp.array([[0.5, 0.5], [0.5, 0.5]])),
        rngs=rngs,
    )

  def __call__(self, x):
    return self.dense(x) + self.gate(x)


class DiLoCoMoESyncUnitTest(unittest.TestCase):

  def test_get_and_apply_flat_router_params(self):
    """Test router parameter extraction and application PyTree helpers."""
    params = {
        "dense": {"kernel": jnp.ones((4, 4))},
        "moe": {
            "gate": {"kernel": jnp.zeros((4, 8))},
            "expert": {"kernel": jnp.ones((8, 4, 4))},
        },
    }

    router_flat = diloco.get_flat_router_params(params)
    target_key = "[\x27moe\x27][\x27gate\x27][\x27kernel\x27]"
    self.assertIn(target_key, router_flat)
    self.assertNotIn("[\x27dense\x27][\x27kernel\x27]", router_flat)
    self.assertNotIn("[\x27moe\x27][\x27expert\x27][\x27kernel\x27]", router_flat)

    # Update router params
    new_router_flat = {target_key: jnp.ones((4, 8)) * 5.0}
    updated_params = diloco.apply_flat_router_params(params, new_router_flat)

    # Check that gate kernel was updated while other params remain unchanged
    self.assertEqual(float(jnp.sum(updated_params["moe"]["gate"]["kernel"])), 4 * 8 * 5.0)
    self.assertEqual(float(jnp.sum(updated_params["dense"]["kernel"])), 16.0)
    self.assertEqual(float(jnp.sum(updated_params["moe"]["expert"]["kernel"])), 128.0)

  def test_fragment_exclusion(self):
    """Test FragmentedTreeManipulator exclude_router parameter."""
    params = {
        "decoder": {
            "layers": {
                "dense": jnp.ones((4, 8, 8)),
                "gate": jnp.zeros((4, 8, 2)),
            }
        }
    }
    config = initialize_pydantic([
        "",
        get_test_config_path(),
        "num_decoder_layers=4",
        "num_diloco_fragments=2",
        "skip_jax_distributed_system=True",
    ])
    manipulator = diloco.FragmentedTreeManipulator.create(params, config)
    
    frag1_with_router = manipulator.get_flat_fragment(params, fragment_idx=1, exclude_router=False)
    frag1_no_router = manipulator.get_flat_fragment(params, fragment_idx=1, exclude_router=True)

    gate_key = "[\x27decoder\x27][\x27layers\x27][\x27gate\x27]"
    self.assertIn(gate_key, frag1_with_router)
    self.assertNotIn(gate_key, frag1_no_router)

  def test_moe_router_frequent_syncing_period_1(self):
    num_replicas = 2
    devices = jax.devices()
    if len(devices) < num_replicas:
      self.skipTest(f"Test requires {num_replicas} devices, but only {len(devices)} are available.")

    mesh_devices = np.array(devices[:num_replicas]).reshape(1, num_replicas)
    mesh = jax.sharding.Mesh(mesh_devices, axis_names=("data", "diloco"))

    config = initialize_pydantic([
        "",
        get_test_config_path(),
        "skip_jax_distributed_system=True",
        f"dcn_diloco_parallelism={num_replicas}",
        "ici_diloco_parallelism=1",
        "diloco_outer_momentum=0.0",
        "diloco_outer_lr=1.0",
        "diloco_sync_period=4",
        "moe_router_syncing_period=1",
    ])

    with jax.set_mesh(mesh):
      tx = optax.sgd(learning_rate=0.1)
      rngs = nnx.Rngs(params=jax.random.key(seed=42))
      model = MoEModel(rngs=rngs)
      graphdef, params = nnx.split(model)

      def nnx_apply_fn(params, inputs):
        model_replica = nnx.merge(graphdef, params)
        return model_replica(inputs)

      vmapped_apply = jax.vmap(nnx_apply_fn, in_axes=(None, 0))

      def _test_train_step(state: train_state.TrainState, batch, prng_key: diloco.PRNGKey):
        del prng_key
        def loss_fn(params, batch):
          inputs, labels = batch
          logits = vmapped_apply(params, inputs)
          return jnp.mean(jnp.square(logits - labels))

        loss, grad = jax.value_and_grad(loss_fn)(state.params, batch)
        return state.apply_gradients(grads=grad), loss

      initial_state = train_state.TrainState.create(
          apply_fn=vmapped_apply,
          params=params,
          tx=tx,
      )

      diloco_state, _ = diloco.build_diloco_state(config, lambda: initial_state)
      diloco_train_step = diloco.build_diloco_train_step(config, _test_train_step)

      inputs = jnp.array([
          [[1.0, 0.0], [1.0, 0.0]],  # Replica 0
          [[0.0, 1.0], [0.0, 1.0]],  # Replica 1
      ])
      labels = jnp.array([
          [[2.0, 2.0], [2.0, 2.0]],
          [[3.0, 3.0], [3.0, 3.0]],
      ])

      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "diloco"))
      inputs = jax.device_put(inputs, sharding)
      labels = jax.device_put(labels, sharding)

      # Step 1 (not full model sync step, but moe_router_syncing_period=1 so router IS synced)
      diloco_state, _ = diloco_train_step(diloco_state, (inputs, labels), jax.random.key(seed=42))

      inner_params = diloco_state.inner_state.params
      r0_gate = inner_params["gate"]["kernel"][0]
      r1_gate = inner_params["gate"]["kernel"][1]
      chex.assert_trees_all_close(r0_gate, r1_gate, atol=1e-5)

      r0_dense = inner_params["dense"]["kernel"][0]
      r1_dense = inner_params["dense"]["kernel"][1]
      self.assertFalse(jnp.allclose(r0_dense, r1_dense))


if __name__ == "__main__":
  unittest.main()

