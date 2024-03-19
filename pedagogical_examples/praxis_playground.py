import jax
from praxis import layers
from praxis.layers import pipeline
from praxis import pax_fiddle
from praxis import base_layer
from praxis import py_utils
from jax import numpy as jnp
import numpy as np
from jax.experimental.pjit import pjit

instantiate = base_layer.instantiate
NestedMap = py_utils.NestedMap
WeightHParams = base_layer.WeightHParams
WeightInit = base_layer.WeightInit
SplitDimsMapping = base_layer.SplitDimsMapping



n_devs = jax.device_count()
if n_devs % 4 != 0:
    stages = n_devs
else:
    stages = 4
mesh_shape = [stages, n_devs // stages]
device_mesh = np.array(jax.local_devices()).reshape(mesh_shape)
stage_axis = 'stage'
mdl_axis = 'mdl'
mesh_axis_names = [stage_axis, mdl_axis]
w_in_mh_sharding = [None, mdl_axis]
w_out_hm_sharding = [mdl_axis, None]
bsm_sharding = [None, None, mdl_axis]
bsh_sharding = [None, None, mdl_axis]

model_dim = 3
hidden_dim = 5
seq_len = 7
microbatch_size = 9
microbatches = 8

print("Running praxis_playground with \n")
print(f"{stages=}")
print(f"{model_dim=}")
print(f"{hidden_dim=}")
print(f"{seq_len=}")
print(f"{microbatch_size=}")
print(f"{microbatches=}")

class SingleStageLayer(base_layer.BaseLayer):
  """Stage-parallel dense-relu-dense.

  Attributes:
    model_dim: Model dimension size.
    hidden_dim: Hidden dimension size.
    w_in_mh_sharding: w_in_mh_sharding.
    w_out_hm_sharding: w_out_hm_sharding.
    bsm_sharding: bsm_sharding.
    bsh_sharding: bsh_sharding.
  """
  model_dim: int = 0
  hidden_dim: int = 0
  w_in_mh_sharding: SplitDimsMapping = None
  w_out_hm_sharding: SplitDimsMapping = None
  bsm_sharding: SplitDimsMapping = None
  bsh_sharding: SplitDimsMapping = None

  def setup(self):
    assert self.name
    assert self.model_dim > 0
    assert self.hidden_dim > 0

    self.create_variable(
        'w_in',
        WeightHParams(
            shape=[self.model_dim, self.hidden_dim],
            init=WeightInit.Gaussian(1.0),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=self.w_in_mh_sharding))
    self.create_variable(
        'w_out',
        WeightHParams(
            shape=[self.hidden_dim, self.model_dim],
            init=WeightInit.Gaussian(1.0),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=self.w_out_hm_sharding))
    # A counter keeping track of how many times fprop is invoked.
    self.create_variable(
        'counter',
        WeightHParams(
            shape=[],
            dtype=jnp.int32,
            init=WeightInit.Constant(0),
            tensor_split_dims_mapping=()),
        trainable=False)

  def __call__(self, inputs):
    theta = self.theta
    w_in = base_layer.maybe_shard(theta.w_in, self.w_in_mh_sharding,
                                  self.mesh_axis_names)
    w_out = base_layer.maybe_shard(theta.w_out, self.w_out_hm_sharding,
                                   self.mesh_axis_names)
    inputs = base_layer.maybe_shard(inputs, self.bsm_sharding,
                                    self.mesh_axis_names)
    h = jnp.einsum(
        'bsm,mh->bsh', inputs, w_in, precision=jax.lax.Precision.HIGHEST)
    h = jax.nn.relu(h)
    h = base_layer.maybe_shard(h, self.bsh_sharding, self.mesh_axis_names)
    outp = jnp.einsum(
        'bsh,hm->bsm', h, w_out, precision=jax.lax.Precision.HIGHEST)
    # This constant is summed across stages, but averaged across microbatches.
    self.add_aux_loss('one', 1.0, 0.5)
    # This summary is averaged across microbatches, and kept as per-stage.
    self.add_summary('one', 1.0)
    self.update_var('counter', self.get_var('counter') + 1)
    return base_layer.maybe_shard(outp, self.bsm_sharding, self.mesh_axis_names)



stacked_transformer = layers.StackedTransformer(num_layers=4, num_heads=8)

pipelined_transformer = layers.PipelinedTransformer(stream_io=True)

layerwise_pipeline = layers.LayerwiseShardablePipelined(stream_io=True)

inner_params = pax_fiddle.Config(
    SingleStageLayer,
    model_dim=model_dim,
    hidden_dim=hidden_dim,
    w_in_mh_sharding=w_in_mh_sharding, #w_in_mh_sharding,
    w_out_hm_sharding=w_out_hm_sharding, # w_out_hm_sharding,
    bsm_sharding=bsm_sharding, #bsm_sharding,
    bsh_sharding=bsh_sharding, #bsh_sharding)
)


pipelined_layer_p = pax_fiddle.Config(
    pipeline.LayerwiseShardablePipelined, #pipeline.CircularLayerwiseShardablePipelined,
    name='pipeline',
    num_stages=4,
    mesh_axis_names=None,
    single_stage_body=inner_params,
    stream_io=True,
    polluting_bubbles_with_nan=True
)

pipelined_layer = instantiate(pipelined_layer_p)

test_inputs = np.ones((microbatches, microbatch_size, seq_len, model_dim))
print(f"test_inputs is [microbatches, microbatch_size, seq_len, model_dim] = [{microbatches}, {microbatch_size}, {seq_len}, {model_dim}]")






with jax.sharding.Mesh(device_mesh, mesh_axis_names):

      def init(key, inp):
        return pipelined_layer.init(key, inp)

      pjit_init = pjit(init, in_shardings=None, out_shardings=None)

      prng_key = jax.random.PRNGKey(seed=123)
      weight_hparams = pipelined_layer.abstract_init_with_metadata(test_inputs)
      print('## weight_hparams=', weight_hparams)
      #self.assertEqual(set(weight_hparams), {'params', 'non_trainable'})
      w_in_metadata = weight_hparams['params']['body']['w_in']
      #self.assertEqual(w_in_metadata.shape, [model_dim, hidden_dim])
    #   if not is_circular_schedule or circular_share_weights:
    #     self.assertEqual(w_in_metadata.repeat_prefix, [stages])
    #     self.assertEqual(w_in_metadata.repeat_prefix_split_dims_mapping,
    #                      ('stage',))
    #   else:
    #     self.assertEqual(w_in_metadata.repeat_prefix, [circular_repeat, stages])
    #     self.assertEqual(w_in_metadata.repeat_prefix_split_dims_mapping,
    #                      (None, 'stage'))

      # gpipe:
      #   {'body': {'w_in': (4, 128, 256), 'w_out': (4, 256, 128)}}
      # circular:
      #   {'body': {'w_in': (2, 4, 128, 256), 'w_out': (2, 4, 256, 128)}}
      pipelined_layer_vars = pjit_init(prng_key, test_inputs)
      print('## pipelined_layer_vars=',
            jax.tree.map(lambda x: x.shape, pipelined_layer_vars))

      def loss(v, inp):
        # Need to allow
        result, _ = pipelined_layer.apply(
            v, inp, mutable=[base_layer.NON_TRAINABLE])
        assert result.shape == inp.shape
        return jnp.sum(result**2)

      def get_non_weight_vars(v, inp):
        mutables = [
            base_layer.AUX_LOSS, base_layer.SUMMARIES, base_layer.NON_TRAINABLE
        ]
        _, updated_vars = pipelined_layer.apply(v, inp, mutable=mutables)
        return (updated_vars[base_layer.AUX_LOSS]['body']['one'],
                updated_vars[base_layer.SUMMARIES]['body'],
                updated_vars[base_layer.NON_TRAINABLE]['body']['counter'])

      # Set pjit input/output to be replicated.
      # Rely on sharding annotations within StageParallelLayer.fprop.
      pjit_fprop = pjit(
          jax.grad(loss, allow_int=True), in_shardings=None, out_shardings=None
      )
      result = pjit_fprop(pipelined_layer_vars, test_inputs)

# pipelined_layer_p = pax_fiddle.Config(
#     pipeline.CircularLayerwiseShardablePipelined,
#     name='pipeline',
#     num_stages=4,
#     mesh_axis_names=None,
#     single_stage_body=inner_params,
#     stream_io=True,
#     circular_repeat=1,
#     share_weights=True,
#     polluting_bubbles_with_nan=True
# )