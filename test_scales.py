from collections import OrderedDict

import math
import os
import sys
import warnings
import yaml

def get_scales(scale):
  '''Choose appropriate scales for individual dimensions based on global scale
  We choose to rotate between doubling:
    embed_dim
    num_head and mlp_dim
    num_layers
  Any one of these steps is not a perfect doubling, although going through a cycle
  of three is a near perfect 8x scaling except for the linear -> softmax -> output step'''


  log_2_scale = math.floor((math.log2(scale)))
  if 2**log_2_scale != scale:
    scale_warning = ("Scale is rounded down to the nearest power of 2. If you want finer grained control "
                    "of the model sizes explicitly set embed_dim, num_head, mlp_dim, num_layers or head_dim")
    warnings.warn(scale_warning, category=Warning)
  base_scale, rem = divmod(log_2_scale, 3)
  base_scale += 1 
  emb_scale = base_scale + int(rem > 0)
  num_head_scale = base_scale + int(rem > 1)
  mlp_dim_scale = base_scale + int(rem > 1)
  layer_scale = base_scale
  print("scales:",emb_scale, num_head_scale, mlp_dim_scale, layer_scale, flush=True)
  return emb_scale, num_head_scale, mlp_dim_scale, layer_scale

scale = 64
get_scales(scale)
