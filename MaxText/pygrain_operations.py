from collections.abc import Mapping, Sequence
import dataclasses
from typing import Dict
import grain.python as pygrain
import numpy as np
import tensorflow as tf
Features = Dict[str, tf.Tensor]

@dataclasses.dataclass
class ParseFeatures(pygrain.MapTransform):
    def map(self, features):
        def _parse(example):
            parsed = tf.io.parse_example(
                example, {
                'text': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
            })
            return parsed
        return _parse(features)


@dataclasses.dataclass
class NormalizeFeatures(pygrain.MapTransform):
    def map(self, features):
        return {
            'inputs':features['text'].numpy().decode(), 
            'targets': features['text'].numpy().decode()
            }


@dataclasses.dataclass
class ReformatPacking(pygrain.MapTransform):
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
class PadToMaxLength(pygrain.MapTransform):
    def __init__(self, max_length):
        self.max_length = max_length
    def map(self, data):
        def pad(x, max_length):
            pad_amount = max(max_length - x.shape[0], 0)
            pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
            return np.pad(x, pad_amount)
        data['inputs_segmentation'] = np.ones(data['inputs'].shape, dtype = np.int32)
        data['inputs_position'] = np.arange(data['inputs'].shape[0], dtype = np.int32)
        data['targets_segmentation'] = np.ones(data['targets'].shape, dtype = np.int32)
        data['targets_position'] = np.arange(data['targets'].shape[0], dtype = np.int32)
        for key, _ in data.items():
            data[key] = pad(data[key], self.max_length)
        return data                

# class PadToMaxLength():
#     """Pads each input to the specified length
#     """
#     def __init__(self, feature_lengths):
#         self.feature_lengths = feature_lengths

#     def __call__(self, data):
#         def pad(x, max_length):
#             pad_amount = max(max_length - x.shape[0], 0)
#             pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
#             return np.pad(x, pad_amount)
#         data['inputs_segmentation'] = np.ones(data['inputs'].shape, dtype = np.int32)
#         data['inputs_position'] = np.arange(data['inputs'].shape[0], dtype = np.int32)
#         data['targets_segmentation'] = np.ones(data['targets'].shape, dtype = np.int32)
#         data['targets_position'] = np.arange(data['targets'].shape[0], dtype = np.int32)
#         for key, _ in data.items():
#             data[key] = pad(data[key], self.feature_lengths)
#         return data

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
  """Shift inputs and replace EOS by 0 for packed inputs."""
  x['inputs'] = shift_right(x['inputs'], axis=axis)
  targets_nonzero = (x['targets'] != 0)
  x['inputs_segmentation'] *= targets_nonzero
  x['targets_segmentation'] *= targets_nonzero
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.  
  x['inputs'] *= (x['inputs_segmentation'] == shift_right(x['inputs_segmentation'], axis=axis))

  return x

@dataclasses.dataclass
class ShiftData(pygrain.MapTransform):
    def __init__(self, axis = 1):
        self.axis = axis
    def map(self, data):
        return shift_and_refine(data, axis=self.axis)
