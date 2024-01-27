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

"""Operations used by Grain"""

import dataclasses
from typing import Dict
import grain.python as grain
import numpy as np
import tensorflow as tf
Features = Dict[str, tf.Tensor]

@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized example"""
  def map(self, features):
    def _parse(example):
      parsed = tf.io.parse_example(
        example, {
        'text': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
        })
      return parsed
    return _parse(features)


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""
  def map(self, features):
    return {
      'inputs':features['text'].numpy().decode(), 
      'targets': features['text'].numpy().decode()
    }


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""
  def map(self, data):
    return{
        'inputs':data[0]['inputs'],
        'targets':data[0]['targets'],
        'inputs_segmentation':data[1]['inputs'],
        'targets_segmentation':data[1]['targets'],
        'inputs_position':data[2]['inputs'],
        'targets_position':data[2]['targets'],
    }


@dataclasses.dataclass
class PadToMaxLength(grain.MapTransform):
  """Pads each input to the specified length"""  
  def __init__(self, max_length):
    self.max_length = max_length
  def map(self, data):
    """map to each element"""
    def _pad(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
      return np.pad(x, pad_amount)
    data['inputs_segmentation'] = np.ones(data['inputs'].shape, dtype = np.int32)
    data['inputs_position'] = np.arange(data['inputs'].shape[0], dtype = np.int32)
    data['targets_segmentation'] = np.ones(data['targets'].shape, dtype = np.int32)
    data['targets_position'] = np.arange(data['targets'].shape[0], dtype = np.int32)
    for key, _ in data.items():
      data[key] = _pad(data[key], self.max_length)
    return data


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [slice(None),] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(
      x,
      pad_widths,
      mode='constant',
      constant_values=x.dtype.type(0)
      )
  return padded[tuple(slices)]

def shift_and_refine(x, axis=1):
  """Shift inputs, set segmentation to 0 when target element is 0.
  Replace EOS by 0 for packed inputs."""
  x['inputs'] = shift_right(x['inputs'], axis=axis)
  targets_nonzero = x['targets'] != 0
  x['inputs_segmentation'] *= targets_nonzero
  x['targets_segmentation'] *= targets_nonzero
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  x['inputs'] *= x['inputs_segmentation'] == shift_right(x['inputs_segmentation'], axis=axis)

  return x

@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Shift inputs and refine annotations."""
  def __init__(self, axis = 1):
    self.axis = axis
  def map(self, data):
    return shift_and_refine(data, axis=self.axis)
