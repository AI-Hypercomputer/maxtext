"""
 Copyright 2023 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

r"""Convert weights from a Llama or Mistral model to a MaxText one.

Usage:

Get LLaMA pytorch_vars from Meta

Example cmd:
To save a ckpt
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path <path/to/meta/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size llama2-7b

For large size model (e.g. 70B model), this script requires large memory VM.
The script load and save weights in a single pass.
To fit less memory, modify convert() to load/save weights in multiple passes.
Each pass, load and save partial weights (subset of all weight variables).
"""
# pylint: disable=g-line-too-long
import argparse
import pathlib

import numpy as np

import checkpointing
import jax
import jax.numpy as jnp
from flax.training import train_state
import max_logging
import torch
import sys
import gc
import copy

jax.config.update('jax_platform_name', 'cpu')

def permute_to_match_maxtext_rope(arr):
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim-1)

MODEL_PARAMS_DICT = {
    'llama2-70b': {
        'num_layers': 80,
        'num_heads': 64,
        'num_kv_heads': 8,
        'dims_per_head': 128,
        'vocab': 32000,
    },
    'llama2-13b': {
        'num_layers': 40,
        'num_heads': 40,
        'num_kv_heads': 40,
        'dims_per_head': 128,
        'vocab': 32000,
        'num_gpus': 1,
        'fused_qkv': True,
    },
    'llama2-7b': {
        'num_layers': 32,
        'num_heads': 32,
        'num_kv_heads': 32,
        'dims_per_head': 128,
        'vocab': 32000,
        'base_emb_dim': 4096,
        'base_mlp_dim': 11008,
    },
    'mistral-7b': {
        'num_layers': 32,
        'num_heads': 32,
        'num_kv_heads': 8,
        'dims_per_head': 128,
        'vocab': 32000,
        'base_emb_dim': 4096,
        'base_mlp_dim': 14336,
    },
}


def convert(base_model_path, maxtext_model_path, model_size):
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and save at maxtext_model_path

  Attributes:
  base_model_path: checkpoint path
  maxtext_model_path: Path to save the MaxText checkpoint to
  model_size: llama2-7b to 70b or mistral-7b
  """
  """Convert model to maxtext."""
  model_params = MODEL_PARAMS_DICT[model_size]
  base_num_decoder_layers = model_params['num_layers']
  base_num_query_heads = model_params['num_heads']
  head_dim = model_params['dims_per_head']
  base_num_kv_heads = model_params['num_kv_heads']
  vocab_size = model_params['vocab']


  print(f'Loading the base model from {base_model_path}')
  # Skip any hidden files for checkpoints
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob('[!.]*.pth'))
  pytorch_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    print(f'Loading checkpoint {i+1} of {len(ckpt_paths)} ...', flush = True)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    pytorch_vars[int(ckpt_path.name.split('.', maxsplit=2)[1])] = checkpoint
  pytorch_vars = [pytorch_vars[i] for i in sorted(list(pytorch_vars.keys()))]

  jax_weights = {
      'decoder': {
          'layers': {
             'mlp': {}, 
             'pre_self_attention_layer_norm' : {},
             'post_self_attention_layer_norm' : {}, 
             'self_attention' : {},            
          },
          'decoder_norm': {
              'scale': copy.deepcopy(pytorch_vars[0]['norm.weight'].type(torch.float16)).numpy()
              },
         'logits_dense':{
              'kernel': np.concatenate([copy.deepcopy(var['output.weight'].type(torch.float16).numpy()) 
                                        for var in pytorch_vars], axis=0).transpose()[:, :vocab_size]
              }
        },
       'token_embedder':{
              'embedding': np.concatenate([copy.deepcopy(var['tok_embeddings.weight'].type(torch.float16).numpy()) 
                                           for var in pytorch_vars], axis=1)[:vocab_size,:]

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
            'wi_0': {
                'kernel' : []
                },
            'wi_1': {
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
    wq = np.concatenate([var[f'layers.{layer_idx}.attention.wq.weight'].type(torch.float16).numpy()
                         for var in pytorch_vars], axis=0).transpose()
    wk = np.concatenate([var[f'layers.{layer_idx}.attention.wk.weight'].type(torch.float16).numpy()
                         for var in pytorch_vars], axis=0).transpose()
    wv = np.concatenate([var[f'layers.{layer_idx}.attention.wv.weight'].type(torch.float16).numpy()
                         for var in pytorch_vars], axis=0).transpose()

    wq = np.reshape(wq, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])
    wk = np.reshape(wk, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
    wv = np.reshape(wv, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
    wq = permute_to_match_maxtext_rope(wq)
    wk = permute_to_match_maxtext_rope(wk)

    w_post = np.concatenate(
        [
            var[f'layers.{layer_idx}.attention.wo.weight'].type(torch.float16).numpy()
            for var in pytorch_vars
        ],
        axis=1,
    )

    w_post = np.reshape(w_post, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])


    self_attention['query']['kernel'].append(wq)
    self_attention['key']['kernel'].append(wk)
    self_attention['value']['kernel'].append(wv)
    self_attention['out']['kernel'].append(w_post)

    wi_0 = np.concatenate([var[f'layers.{layer_idx}.feed_forward.w1.weight'].type(torch.float16).numpy()
                           for var in pytorch_vars], axis=0).transpose()
    wi_1 = np.concatenate([var[f'layers.{layer_idx}.feed_forward.w3.weight'].type(torch.float16).numpy()
                           for var in pytorch_vars], axis=0).transpose()
    wo = np.concatenate([var[f'layers.{layer_idx}.feed_forward.w2.weight'].type(torch.float16).numpy()
                                 for var in pytorch_vars], axis=1).transpose()
    pre_self_attention_layernorm = pytorch_vars[0][f'layers.{layer_idx}.attention_norm.weight'].type(torch.float16).numpy()
    post_self_attention_layernorm = pytorch_vars[0][f'layers.{layer_idx}.ffn_norm.weight'].type(torch.float16).numpy()



    layer_weight['mlp']['wi_0']['kernel'].append(wi_0)
    layer_weight['mlp']['wi_1']['kernel'].append(wi_1)
    layer_weight['mlp']['wo']['kernel'].append(wo)
    layer_weight['pre_self_attention_layer_norm']['scale'].append(pre_self_attention_layernorm)
    layer_weight['post_self_attention_layer_norm']['scale'].append(post_self_attention_layernorm)
    print(f" layer = {layer_idx} complete", flush=True)
  


  def npify_and_delete(x):
    new_x = np.array(x)
    del x
    return new_x
  

  self_attention['query']['kernel'] = npify_and_delete(self_attention['query']['kernel'])
  print("self_attention['query']['kernel'] complete", flush=True)
  self_attention['key']['kernel'] = npify_and_delete(self_attention['key']['kernel'])
  print("self_attention['key']['kernel'] complete", flush=True)
  self_attention['value']['kernel'] = npify_and_delete(self_attention['value']['kernel'])
  print(" self_attention['value']['kernel'] complete", flush=True)
  self_attention['out']['kernel'] = npify_and_delete(self_attention['out']['kernel'])
  print("self_attention['out']['kernel'] complete", flush=True)
  self_attention['query']['kernel'] = np.transpose(self_attention['query']['kernel'],axes=(1, 0, 2, 3))
  print("self_attention['query']['kernel'] transpose complete", flush=True)
  self_attention['key']['kernel'] = np.transpose(self_attention['key']['kernel'],axes=(1, 0, 2, 3))
  print("self_attention['key']['kernel'] transpose  complete", flush=True)
  self_attention['value']['kernel'] = np.transpose(self_attention['value']['kernel'],axes=(1, 0, 2, 3))
  print("self_attention['value']['kernel'] transpose complete", flush=True)
  #layers, base_num_query_heads * head_dim, base_num_query_heads, head_dim =>
  #base_num_query_heads, layers,head_dim, base_num_query_heads * head_dim
  self_attention['out']['kernel'] = np.transpose(self_attention['out']['kernel'],axes=(2, 0, 3, 1))
  print("complete self_attention['out']['kernel'] transpose", flush=True)
  #scale the query weights
  self_attention['query']['kernel'] = self_attention['query']['kernel']/np.sqrt(head_dim)
  print("complete self_attention['query']['kernel'] scaling", flush=True)

  jax_weights['decoder']['layers']['self_attention'] = self_attention
  print("complete self attention assignment ", flush=True)

  layer_weight['mlp']['wi_0']['kernel'] = npify_and_delete(layer_weight['mlp']['wi_0']['kernel'])
  layer_weight['mlp']['wi_1']['kernel'] = npify_and_delete(layer_weight['mlp']['wi_1']['kernel'])
  layer_weight['mlp']['wo']['kernel'] = npify_and_delete(layer_weight['mlp']['wo']['kernel'])
  layer_weight['pre_self_attention_layer_norm']['scale'] = npify_and_delete(layer_weight['pre_self_attention_layer_norm']['scale'])
  layer_weight['post_self_attention_layer_norm']['scale'] = npify_and_delete(layer_weight['post_self_attention_layer_norm']['scale'])

  del pytorch_vars[:]
  del pytorch_vars
  print("pytorch_vars deleted")
  gc.collect()

  print("npify_and_delete complete", flush=True)
  #swap the layer index
  layer_weight['mlp']['wi_0']['kernel'] = np.transpose(layer_weight['mlp']['wi_0']['kernel'],axes=(1, 0, 2))
  layer_weight['mlp']['wi_1']['kernel'] = np.transpose(layer_weight['mlp']['wi_1']['kernel'],axes=(1, 0, 2))
  layer_weight['mlp']['wo']['kernel'] = np.transpose(layer_weight['mlp']['wo']['kernel'],axes=(1, 0, 2))
  layer_weight['pre_self_attention_layer_norm']['scale'] = np.transpose(
                                    layer_weight['pre_self_attention_layer_norm']['scale'],
                                    axes=(1, 0))
  layer_weight['post_self_attention_layer_norm']['scale'] = np.transpose(
                                    layer_weight['post_self_attention_layer_norm']['scale'],
                                    axes=(1, 0))
  print("complete layer weight", flush=True)
  jax_weights['decoder']['layers']['mlp'] = layer_weight['mlp']
  jax_weights['decoder']['layers']['pre_self_attention_layer_norm'] = layer_weight['pre_self_attention_layer_norm']
  jax_weights['decoder']['layers']['post_self_attention_layer_norm'] = layer_weight['post_self_attention_layer_norm']
  
  print("about to start jnp conversion", flush=True)
  #convert all weights to jax.numpy
  def jnpify_and_delete(x):
    out_x = jnp.array(x)
    del x
    return out_x
  
  #jax_weights = jax.tree_map(jnp.array, jax_weights)
  jax_weights = jax.tree_map(jnpify_and_delete, jax_weights)
  print("jax_weights jnp conversion complete", flush=True)
  #dummy configs for the checkpoint_manager
  step_number_to_save_new_ckpt = 0
  enable_checkpointing=True
  async_checkpointing=False
  save_interval_steps=1


  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      maxtext_model_path,
      enable_checkpointing,
      async_checkpointing,
      save_interval_steps
  )

  state_new = train_state.TrainState(
    step=0,
    apply_fn=None,
    params=jax_weights,
    tx=None, # type: ignore
    opt_state={}
  )

  if checkpoint_manager is not None:
    if checkpoint_manager.save(step_number_to_save_new_ckpt, state_new):
      max_logging.log(f"saved a checkpoint at step {step_number_to_save_new_ckpt}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--base-model-path', type=str, required=True)
  parser.add_argument('--maxtext-model-path', type=str, required=True)
  parser.add_argument('--model-size', type=str, required=True)

  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError
  convert(args.base_model_path, args.maxtext_model_path, args.model_size)
