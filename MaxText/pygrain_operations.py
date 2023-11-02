from typing import Dict
import grain.python as pygrain
import numpy as np
import tensorflow as tf
Features = Dict[str, tf.Tensor]

class normalize_features():
  """Normalize text feature keys.
  """
  def __call__(self, features):
    def _normalize_features(features):
      # features['inputs'] = features.pop('text')
      # features['targets'] = features['inputs']
      return {'inputs':features, 'targets': features}
    return _normalize_features(features)

class TokenizeOperation():
    """ TokenizeOp
    """
    def __init__(self, sp_tokenizer):
        self.sp_tokenizer = sp_tokenizer

    def __call__(self, features: Features):
        data_keys = ('inputs', 'targets')
        for k in data_keys:
            features[k] = np.asarray(self.sp_tokenizer.tokenize(str(features[k])))
            # features[k] = self.sp_tokenizer.tokenize(str(features[k]))
        # import pdb;pdb.set_trace()
        return features

class length_filter():
    """pygrain max length filter
    """
    def __init__(self,max_length):
        self.max_len = max_length
    def __call__(self, x):
        source, target = x['inputs'], x['targets']
        l = np.maximum(np.shape(source)[0], np.shape(target)[0])
        return np.less(l, self.max_len + 1)

class PadToMaxLength():
    """Pads each input to the specified length
    """
    def __init__(self, feature_lengths):
        self.feature_lengths = feature_lengths

    def __call__(self, data):
        def pad(x, max_length):
            pad_amount = max(max_length - x.shape[0], 0)
            pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
            return np.pad(x, pad_amount)
        data['inputs_segmentation'] = np.ones(data['inputs'].shape)
        data['inputs_position'] = np.ones(data['inputs'].shape, dtype = np.int32)
        for key, _ in data.items():
            data[key] = pad(data[key], self.feature_lengths)
        return data

class CombineKeys():
    """ Combine tuples of sequence packing output in different keys
    """
    def __call__(self, data):
        combined_data = data[0]
        segments = data[1]
        segments['inputs_segmentation'] = segments.pop('inputs')
        segments['targets_segmentation'] = segments.pop('targets')
        positions = data[2]
        positions['inputs_position'] = positions.pop('inputs')
        positions['targets_position'] = positions.pop('targets')
        combined_data.update(segments)
        combined_data.update(positions)
        return combined_data

def shift_right_tf(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [slice(None),] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = tf.pad(
      x,
      tf.constant(pad_widths),
      mode='constant',
      constant_values=tf.constant(0, x.dtype))
  return padded[tuple(slices)]

def shift_inputs_tf(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right_tf(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= tf.cast(
        segment_ids == shift_right_tf(segment_ids, axis=axis), x.dtype
    )
  return shifted

class ShiftData():
    def __init__(self, axis = 0, segmented=True):
        self.axis = axis
        self.segmented = segmented

    def __call__(self, x):
        segment_ids = x['inputs_segmentation'] if self.segmented else None
        x['inputs'] = shift_inputs_tf(x['inputs'], segment_ids=segment_ids, axis=self.axis)
        return x        