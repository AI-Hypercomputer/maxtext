r"""Convert weights from a llama model to a MaxText one.

Usage:

# Get LLaMA pytorch_vars from Meta

# Example cmd:
##To save a ckpt to finetune (this is the default)
python3 MaxText/convert_llama_ckpt.py --base-model-path <path/to/meta/ckpt> --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size 7b --save-training-ckpt
##To save a ckpt for decoding
python3 MaxText/convert_llama_ckpt.py --base-model-path <path/to/meta/ckpt> --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size 7b --save-decoding-ckpt

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

import checkpointing
from flax.training import train_state
import max_utils
import pyconfig
from jax.sharding import Mesh
from layers.llama2 import Transformer
from jax import random
import max_utils
import maxtext_utils


import max_logging

import torch

import sys

#tmp:
import pickle

MODEL_PARAMS_DICT = {
    '70b': {
        'num_layers': 80,
        'num_heads': 64,
        'num_kv_heads': 8,
        'dims_per_head': 128,
        'vocab': 32000,
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
        'base_emb_dim': 4096,
        'base_mlp_dim': 11008,
        'max_target_length': 11008,
        'max_eval_target_length': 4096,
    },
}

#dummy configs for the checkpoint_manager
enable_checkpointing = True
async_checkpointing = True
save_period = 1
step_number_to_save_new_ckpt = 0

# permute for sliced rotary
def permute(w, n_heads, dim1, dim2):
    # return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
  print(f"w.shape = {w.shape}, n_heads = {n_heads}, dim1 = {dim1}, dim2 = {dim2}")
  w1 = np.reshape(w, (n_heads, dim1 // n_heads // 2, 2, dim2))
  print(f"w1.shape = {w1.shape}")
  w2 = np.transpose(w1, (0, 2 ,1, 3))
  print(f"w2.shape = {w2.shape}")
  return np.reshape(w2, (dim1, dim2))

def convert(base_model_path, maxtext_model_path, model_size, save_training_ckpt):
  
  print(f"save_training_ckpt = {save_training_ckpt}")
  """Convert from Llama to maxtext."""
  model_params = MODEL_PARAMS_DICT[model_size]
  base_num_decoder_layers = model_params['num_layers']
  base_num_heads = model_params['num_heads']
  head_dim = model_params['dims_per_head']
  num_kv_heads = model_params['num_kv_heads']
  vocab_size = model_params['vocab']
  base_emb_dim = model_params['base_emb_dim']
  base_mlp_dim = model_params['base_mlp_dim']
  max_target_length = model_params['max_target_length']
  max_eval_target_length = model_params['max_eval_target_length']
  
  
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
          'decoder': {
             'mlp': {}, 
             'pre_self_attention_layer_norm' : {},
             'post_self_attention_layer_norm' : {}, 
             'self_attention' : {},            
          }, 
          'decoder_norm': {
              'scale': pytorch_vars[0]['norm.weight'].type(torch.float16).numpy()
              },
         'logits_dense':{
              'kernel': np.concatenate([var['output.weight'].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()[:, :vocab_size]         
              }        
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
  
  meta_self_attention = {'query': [], 'key': [], 'value': [], 'out': []}

  for layer_idx in range(base_num_decoder_layers):
    wq = np.concatenate([var['layers.%d.attention.wq.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0)#.transpose()
    wk = np.concatenate([var['layers.%d.attention.wk.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0)#.transpose()
    wv = np.concatenate([var['layers.%d.attention.wv.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()
    t = wq
    #apply permute like HF
    wq = permute(wq, base_num_heads, wq.shape[0], wq.shape[1]).transpose()
    wk = permute(wk, base_num_heads, wk.shape[0], wk.shape[1]).transpose()

    #temp:saving meta's checkpoint
    meta_self_attention['query'].append((wq,t))
    meta_self_attention['key'].append(wk)
    meta_self_attention['value'].append(wv)

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
    #temp:saving meta's checkpoint
    meta_self_attention['out'].append(w_post)

    w_post = np.reshape(w_post, [base_num_heads * head_dim, base_num_heads, head_dim])


    self_attention['query']['kernel'].append(wq)
    self_attention['key']['kernel'].append(wk)
    self_attention['value']['kernel'].append(wv)
    self_attention['out']['kernel'].append(w_post)

    ffn_layer1_gate = np.concatenate([var['layers.%d.feed_forward.w1.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()
    ffn_layer1 = np.concatenate([var['layers.%d.feed_forward.w3.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=0).transpose()
    ffn_layer2 = np.concatenate([var['layers.%d.feed_forward.w2.weight' % (layer_idx)].type(torch.float16).numpy() for var in pytorch_vars], axis=1).transpose()
    pre_self_attention_layernorm = pytorch_vars[0]['layers.%d.attention_norm.weight' % (layer_idx)].type(torch.float16).numpy()
    post_self_attention_layernorm = pytorch_vars[0]['layers.%d.ffn_norm.weight' % (layer_idx)].type(torch.float16).numpy()



    layer_weight['mlp']['wi']['kernel'].append(ffn_layer1_gate)
    layer_weight['mlp']['ffn_layer1']['kernel'].append(ffn_layer1)
    layer_weight['mlp']['wo']['kernel'].append(ffn_layer2)
    layer_weight['pre_self_attention_layer_norm']['scale'].append(pre_self_attention_layernorm)
    layer_weight['post_self_attention_layer_norm']['scale'].append(post_self_attention_layernorm)
  
  #temp:saving meta's checkpoint
    
  with open('/dev/shm/llama2-7b-meta-permute-no-transpose.pickle', 'wb') as handle:
    pickle.dump(meta_self_attention, handle, protocol=pickle.HIGHEST_PROTOCOL)

  self_attention['query']['kernel'] = np.array(self_attention['query']['kernel'])
  self_attention['key']['kernel'] = np.array(self_attention['key']['kernel'])
  self_attention['value']['kernel'] = np.array(self_attention['value']['kernel'])
  self_attention['out']['kernel'] = np.array(self_attention['out']['kernel'])
  self_attention['query']['kernel'] = np.transpose(self_attention['query']['kernel'],axes=(1, 0, 2, 3)) 
  self_attention['key']['kernel'] = np.transpose(self_attention['key']['kernel'],axes=(1, 0, 2, 3))
  self_attention['value']['kernel'] = np.transpose(self_attention['value']['kernel'],axes=(1, 0, 2, 3))
  #layers, base_num_heads * head_dim, base_num_heads, head_dim => base_num_heads, layers,head_dim, base_num_heads * head_dim
  self_attention['out']['kernel'] = np.transpose(self_attention['out']['kernel'],axes=(2, 0, 3, 1))

  jax_weights['decoder']['decoder']['self_attention'] = self_attention


  layer_weight['mlp']['wi']['kernel'] = np.array(layer_weight['mlp']['wi']['kernel'])
  layer_weight['mlp']['ffn_layer1']['kernel'] = np.array(layer_weight['mlp']['ffn_layer1']['kernel'])
  layer_weight['mlp']['wo']['kernel'] = np.array(layer_weight['mlp']['wo']['kernel'])
  layer_weight['pre_self_attention_layer_norm']['scale'] = np.array(layer_weight['pre_self_attention_layer_norm']['scale'])
  layer_weight['post_self_attention_layer_norm']['scale'] = np.array(layer_weight['post_self_attention_layer_norm']['scale'])
  #swap the layer index
  layer_weight['mlp']['wi']['kernel'] = np.transpose(layer_weight['mlp']['wi']['kernel'],axes=(1, 0, 2))
  layer_weight['mlp']['ffn_layer1']['kernel'] = np.transpose(layer_weight['mlp']['ffn_layer1']['kernel'],axes=(1, 0, 2))
  layer_weight['mlp']['wo']['kernel'] = np.transpose(layer_weight['mlp']['wo']['kernel'],axes=(1, 0, 2))
  layer_weight['pre_self_attention_layer_norm']['scale'] = np.transpose(layer_weight['pre_self_attention_layer_norm']['scale'],axes=(1, 0))
  layer_weight['post_self_attention_layer_norm']['scale'] = np.transpose(layer_weight['post_self_attention_layer_norm']['scale'],axes=(1, 0))
  
  jax_weights['decoder']['decoder']['mlp'] = layer_weight['mlp']
  jax_weights['decoder']['decoder']['pre_self_attention_layer_norm'] = layer_weight['pre_self_attention_layer_norm']
  jax_weights['decoder']['decoder']['post_self_attention_layer_norm'] = layer_weight['post_self_attention_layer_norm']
  
  print(f"jax_weights = {jax_weights}")

  base_output_directory="base_output_directory=dummy_base_output_dir"
  base_num_decoder_layers=f"base_num_decoder_layers={base_num_decoder_layers}"
  base_num_heads = f"base_num_heads={base_num_heads}"
  head_dim = f"head_dim={head_dim}"
  async_checkpointing="async_checkpointing=False" 
  enable_dropout="enable_dropout=False"

  vocab_size=f"vocab_size={vocab_size}"  
  base_emb_dim=f"base_emb_dim={base_emb_dim}" 
  base_mlp_dim=f"base_mlp_dim={base_mlp_dim}" 
  max_target_length=f"max_target_length={max_target_length}" 
  max_eval_target_length=f"max_eval_target_length={max_eval_target_length}" 
  
  commandline_args = ["", 
                      "MaxText/configs/base.yml","save_period=5",
                      "steps=20", base_emb_dim, base_mlp_dim, base_num_heads, head_dim,
                      vocab_size, base_num_decoder_layers, max_target_length, max_eval_target_length,  
                      async_checkpointing, 
                      base_output_directory, enable_dropout]

  pyconfig.initialize(commandline_args)
  config = pyconfig.config
  init_rng, nextrng = random.split(random.PRNGKey(config.init_weights_seed), 2)
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  model = Transformer(config, mesh)

  
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      maxtext_model_path,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.save_period,
  )


  with open('/dev/shm/llama2-7b.pickle', 'wb') as handle:
    pickle.dump(jax_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

  if save_training_ckpt:
    print("Creating a training state")
    learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
    tx = maxtext_utils.get_optimizer(config, learning_rate_schedule)
    state_new, _ = max_utils.setup_training_state(
      model, tx, config, init_rng, mesh, checkpoint_manager
      )
  else:
    print("Creating a decoding state")
    state_new, _ = max_utils.setup_decode_state(
      model, config, init_rng, mesh, checkpoint_manager
      )

  print(f"default trainstate={state_new}")

  for key in state_new.params.keys():
    state_new.params[key] = jax_weights[key]


  print(f"trainstate after replacing params with jax_weights={state_new}")

  if checkpoint_manager is not None:
      if checkpoint_manager.save(0, state_new):
        max_logging.log(f"saved a checkpoint at step {0}")
      # Upon preemption, exit when and only when all ongoing saves are complete.
      if checkpoint_manager.reached_preemption(0):
          checkpoint_manager.wait_until_finished()
          sys.exit()

  





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--base-model-path', type=str, required=True)
  parser.add_argument('--maxtext-model-path', type=str, required=True)
  parser.add_argument('--model-size', type=str, required=True)
  parser.add_argument('--save-training-ckpt', dest='save_training_ckpt', action='store_true')
  parser.add_argument('--save-decoding-ckpt', dest='save_training_ckpt', action='store_false')
  parser.set_defaults(save_training_ckpt=True)

  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError
  convert(args.base_model_path, args.maxtext_model_path, args.model_size, args.save_training_ckpt)