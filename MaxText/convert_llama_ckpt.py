r"""Convert weights from a llama/vicuna model to a pax one.

Usage:

# Get LLaMA pytorch_vars from Meta

# Example cmd:
python3 -m convert_llama_ckpt --base llama_7b --pax pax_7b --model-size 7b

# For large size model (e.g. 70B model), this script requires large memory VM.
# The script load and save weights in a single pass.
# To fit less memory, modify convert() to load/save weights in multiple passes.
# Each pass, load and save partial weights (subset of all weight variables).
"""
# pylint: disable=g-line-too-long
import argparse
import pathlib

import jax
from jax.experimental import pjit
import numpy as np

# from paxml import checkpoints
import checkpointing
from flax.training import train_state
# from paxml import train_states
# from praxis import py_utils
import max_utils
import pyconfig
import optax
from jax.sharding import Mesh
from layers import Transformer
from jax import random
import jax.numpy as jnp
import functools

import max_logging

import torch

MODEL_PARAMS_DICT = {
    '70b': {
        'num_layers': 80,
        'num_heads': 64,
        'num_kv_heads': 8,
        'dims_per_head': 128,
        'vocab': 32000,
        'num_gpus': 1,
        'combined_qkv': False,
    },
    '13b': {
        'num_layers': 40,
        'num_heads': 40,
        'num_kv_heads': 40,
        'dims_per_head': 128,
        'vocab': 32000,
        'num_gpus': 1,
        'combined_qkv': True,
    },
    '7b': {
        'num_layers': 32,
        'num_heads': 32,
        'num_kv_heads': 32,
        'dims_per_head': 128,
        'vocab': 32000,
        'num_gpus': 1,
        # 'combined_qkv': True,
        'combined_qkv': False,
    },
}

#dummy configs for the checkpoint_manager
enable_checkpointing = True
async_checkpointing = True
save_period = 1
step_number_to_save_new_ckpt = 0

def convert(base_model_path, maxtext_model_path, model_size):
  """Convert from Llama to maxtext."""
  model_params = MODEL_PARAMS_DICT[model_size]
  base_num_decoder_layers = model_params['num_layers']
  base_num_heads = model_params['num_heads']
  head_dim = model_params['dims_per_head']
  num_kv_heads = model_params['num_kv_heads'] #Anisha: what is this?
  vocab_size = model_params['vocab']
  combined_qkv = model_params['combined_qkv'] #Anisha: what is this?
  num_gpus = model_params['num_gpus']
  #Anisha: base_mlp_dim
  #Anisha: base_embed_dim
  
  dataset_type = 'c4'

  print(f'Loading the base model from {base_model_path}')
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob('*.pth'))
  pytorch_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    print(f'Loading checkpoint {i+1} of {len(ckpt_paths)} ...')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    pytorch_vars[int(ckpt_path.name.split('.', maxsplit=2)[1])] = checkpoint
  pytorch_vars = [pytorch_vars[i] for i in sorted(list(pytorch_vars.keys()))]

  jax_weights = {
      'decoder': {
        #   'softmax': { #Anisha: what is this?
        #       'logits_ffn': {
        #           'linear': {
        #               'w': np.concatenate([var['output.weight'].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()[:, :vocab_size]
        #               }
        #           }
        #       },
          'decoder': {
             'mlp': {}, 
             'pre_self_attention_layer_norm' : {},
             'post_self_attention_layer_norm' : {}, 
             'self_attention' : {},            
          }, 
          'decoder_norm': {
              'scale': pytorch_vars[0]['norm.weight'].type(torch.float16).numpy()
              },         
        },
       'token_embedder':{
              'embedding': np.concatenate([var['tok_embeddings.weight'].type(torch.float16).numpy() for var in pytorch_vars], axis=1)[:vocab_size,:]
         
       }
      }
    

  self_attention = {
        'query': {
            'kernel' : []
        }, 
        'key': {
            'kernel' : []
        },
        'value': {
            'kernel' : []
        },
        'out': {
            'kernel' : []
        },
    }

  layer_weight = {
        'mlp': {
            'wi': {
                'kernel' : []
                },
            'ffn_layer1': {
                'kernel' : []
                },
            'wo': {
                'kernel' : []
                },
            },
        'pre_self_attention_layer_norm': {
            'scale': []
            },
        'post_self_attention_layer_norm': {
            'scale': []
            }
        } 
  
  for layer_idx in range(base_num_decoder_layers):
    wq = np.concatenate([var['layers.%d.attention.wq.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()
    wk = np.concatenate([var['layers.%d.attention.wk.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()
    wv = np.concatenate([var['layers.%d.attention.wv.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()
    if True:#combined_qkv:
    #   wc = np.stack([wq, wk, wv], axis=0)
    #   wc = np.reshape(wc, [3, base_num_heads * head_dim, base_num_heads, head_dim])
    # else:
      wq = np.reshape(wq, [base_num_heads * head_dim, base_num_heads, head_dim])
      wk = np.reshape(wk, [base_num_heads * head_dim, num_kv_heads, head_dim])
      wv = np.reshape(wv, [base_num_heads * head_dim, num_kv_heads, head_dim])

    w_post = np.concatenate(
        [
            var['layers.%d.attention.wo.weight' % (layer_idx)].type(torch.float16).numpy()
            for var in pytorch_vars
        ],
        axis=1,
    )
    w_post = np.reshape(w_post, [base_num_heads * head_dim, base_num_heads, head_dim])

    # if combined_qkv:
    #   attention_weights = {
    #       'self_attention': {'combined_qkv': {'w': wc}, 'post': {'w': w_post}}
    #   }
    # else:
    #   attention_weights = {
    #       'self_attention': {
    #           'query': {'w': wq},
    #           'key': {'w': wk},
    #           'value': {'w': wv},
    #           'post': {'w': w_post},
    #       },
    #   }

    self_attention['query']['kernel'].append(wq)
    self_attention['key']['kernel'].append(wk)
    self_attention['value']['kernel'].append(wv)
    self_attention['out']['kernel'].append(w_post)

    ffn_layer1_gate = np.concatenate([var['layers.%d.feed_forward.w1.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()
    ffn_layer1 = np.concatenate([var['layers.%d.feed_forward.w3.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()
    ffn_layer2 = np.concatenate([var['layers.%d.feed_forward.w2.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=1).transpose()
    pre_self_attention_layernorm = pytorch_vars[0]['layers.%d.attention_norm.weight' % (layer_idx)].type(torch.float16).numpy()
    post_self_attention_layernorm = pytorch_vars[0]['layers.%d.ffn_norm.weight' % (layer_idx)].type(torch.float16).numpy()


    # layer_weight.update(attention_weights)
    # jax_weights['decoder']['transformer']['x_layers_%d' % layer_idx] = layer_weight

    layer_weight['mlp']['wi']['kernel'].append(ffn_layer1_gate)
    layer_weight['mlp']['ffn_layer1']['kernel'].append(ffn_layer1)
    layer_weight['mlp']['wo']['kernel'].append(ffn_layer2)
    layer_weight['pre_self_attention_layer_norm']['scale'].append(pre_self_attention_layernorm)
    layer_weight['post_self_attention_layer_norm']['scale'].append(post_self_attention_layernorm)
  
  self_attention['query']['kernel'] = np.array(self_attention['query']['kernel'])
  self_attention['key']['kernel'] = np.array(self_attention['key']['kernel'])
  self_attention['value']['kernel'] = np.array(self_attention['value']['kernel'])
  self_attention['out']['kernel'] = np.array(self_attention['out']['kernel'])
  #Swap the 0th and 1st indices for q,k,v and out's dims should be (4, 1, 96, 512), dtype=float32), names=('heads', 'layers', 'kv', 'embed')
  self_attention['query']['kernel'] = jnp.transpose(self_attention['query']['kernel'],axes=(1, 0, 2, 3)) 
  self_attention['key']['kernel'] = jnp.transpose(self_attention['key']['kernel'],axes=(1, 0, 2, 3))
  self_attention['value']['kernel'] = jnp.transpose(self_attention['value']['kernel'],axes=(1, 0, 2, 3))
  self_attention['out']['kernel'] = jnp.transpose(self_attention['out']['kernel'],axes=(1, 0, 2, 3))

  jax_weights['decoder']['decoder']['self_attention'] = self_attention


  layer_weight['mlp']['wi']['kernel'] = np.array(layer_weight['mlp']['wi']['kernel'])
  layer_weight['mlp']['ffn_layer1']['kernel'] = np.array(layer_weight['mlp']['ffn_layer1']['kernel'])
  layer_weight['mlp']['wo']['kernel'] = np.array(layer_weight['mlp']['wo']['kernel'])
  layer_weight['pre_self_attention_layer_norm']['scale'] = np.array(layer_weight['pre_self_attention_layer_norm']['scale'])
  layer_weight['post_self_attention_layer_norm']['scale'] = np.array(layer_weight['post_self_attention_layer_norm']['scale'])
  #swap the layer index
  layer_weight['mlp']['wi']['kernel'] = jnp.transpose(layer_weight['mlp']['wi']['kernel'],axes=(1, 0, 2))
  layer_weight['mlp']['ffn_layer1']['kernel'] = jnp.transpose(layer_weight['mlp']['ffn_layer1']['kernel'],axes=(1, 0, 2))
  layer_weight['mlp']['wo']['kernel'] = jnp.transpose(layer_weight['mlp']['wo']['kernel'],axes=(1, 0, 2))
  layer_weight['pre_self_attention_layer_norm']['scale'] = jnp.transpose(layer_weight['pre_self_attention_layer_norm']['scale'],axes=(1, 0))
  layer_weight['post_self_attention_layer_norm']['scale'] = jnp.transpose(layer_weight['post_self_attention_layer_norm']['scale'],axes=(1, 0))
  
  jax_weights['decoder']['decoder']['mlp'] = layer_weight['mlp']
  jax_weights['decoder']['decoder']['pre_self_attention_layer_norm'] = layer_weight['pre_self_attention_layer_norm']
  jax_weights['decoder']['decoder']['post_self_attention_layer_norm'] = layer_weight['post_self_attention_layer_norm']
  
  base_output_directory="base_output_directory=gs://mazumdera-test-bucket/maxtext/llama2/12062023/1"
  base_num_decoder_layers="base_num_decoder_layers=32"
  base_num_heads = "base_num_heads=32"
  head_nums = "head_dim=128"
  # activation_function="\"relu\""
  # mlp_activations = f"mlp_activations=[{activation_function}]"
  async_checkpointing="async_checkpointing=False" 
  enable_dropout="enable_dropout=False"

  commandline_args = ["dummy", "/home/mazumdera/maxtext/MaxText/configs/base.yml","run_name=1xv4-8", "dcn_data_parallelism=1", "save_period=5","ici_data_parallelism=1","ici_tensor_parallelism=1","steps=20","enable_profiler=true","remat_policy=full","base_emb_dim=512", base_num_heads, head_nums,"vocab_size=50272", base_num_decoder_layers, "per_device_batch_size=0.5","enable_profiler=true", "base_mlp_dim=2048", base_output_directory, "max_predict_length=512", async_checkpointing, enable_dropout]# , mlp_activations]

  pyconfig.initialize(commandline_args)
  config = pyconfig.config
  init_rng, nextrng = random.split(random.PRNGKey(config.init_weights_seed), 2)
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  model = Transformer(config, mesh)
  tx = optax.adam(1e-3)
#   state, state_mesh_annotations = max_utils.setup_initial_state(model, tx, config, init_rng, mesh, checkpoint_manager)


  jax_states = train_state.TrainState(
      step=0,
      params=jax_weights,
      opt_state={},
      tx = tx,
      apply_fn=model.apply
      )
  
  print(f'Verify train state')
  init_train_state_partial = functools.partial(max_utils.init_train_state, model, tx,
                                               config)
  abstract_state = jax.eval_shape(init_train_state_partial, init_rng)
  print(f'abstract_state = {abstract_state}')

  
  print(f'Saving the maxtext model to {maxtext_model_path}')
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      maxtext_model_path,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.save_period,
  )
  if checkpoint_manager is not None:
      if checkpoint_manager.save(step_number_to_save_new_ckpt, jax_states):
          max_logging.log(f"saved a checkpoint at step {step_number_to_save_new_ckpt}")

  
  
#   global_mesh = jax.sharding.Mesh(
#       device_mesh, ['replica', 'data_mdl2', 'mdl'])

#   # Identity pjit is needed to output a GDA model_states.
#   def identity(x):
#     return x

#   pjitted_identity = pjit.pjit(identity,
#                                in_shardings=None,
#                                out_shardings=None)

#   with global_mesh:
#     jax_states_gda = pjitted_identity(jax_states)

#   checkpoints.save_checkpoint(
#       jax_states_gda, maxtext_model_path,
#       checkpoint_type=checkpoints.CheckpointType.GDA)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--base-model-path', type=str, required=True)
  parser.add_argument('--maxtext-model-path', type=str, required=True)
  parser.add_argument('--model-size', type=str, required=True)
  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError
  convert(args.base_model_path, args.maxtext_model_path, args.model_size)