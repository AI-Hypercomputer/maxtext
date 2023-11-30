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

"""Provides op for tokenizing a dataset."""

from typing import Any, Dict, Iterable

import dataclasses
from absl import logging
import tensorflow as tf
import tensorflow_text as tftxt

import max_logging
import numpy as np

Features = Dict[str, tf.Tensor]



def _load_sentencepiece_tokenizer(model_path: str,
                                  add_bos: bool = False,
                                  add_eos: bool = True,
                                  reverse: bool = False):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  max_logging.log(f"Model path: {model_path}")
  with tf.io.gfile.GFile(model_path, 'rb') as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
      model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
  return sp_tokenizer

def load_tokenizer(vocab_path: str, vocab_size: int,):
  """Loads the tokenizer at `vocab_path` or trains a one from `dataset`."""
  try:
    sp_tokenizer = _load_sentencepiece_tokenizer(vocab_path)
    sp_size = int(sp_tokenizer.vocab_size())
    if sp_size != vocab_size:
      raise ValueError(f'Existing sentencepiece vocabulary size {sp_size} '
                       f'does not match specified vocab size {vocab_size}.')
    return sp_tokenizer
  except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError):
    logging.info('SentencePiece vocab not found, Run train_tokenizer.py')
    return None


@dataclasses.dataclass
class TokenizeOp:

  sp_tokenizer: Any
  data_keys: Iterable[str] = ('inputs', 'targets')
  # data_keys: Iterable[str] = ('inputs')
  # def __call__(self, features):  
  #   # This function wraps tokenize_function with tf.py_function  
  #   dtypes = [tf.int32]*len(features)
  #   print(f"features in TokenizeOp = {features}")
  #   values =  tf.py_function(func=self.tokenize_function, inp=features, Tout=dtypes)  
  #   return {k:v for k,v in zip(features.keys(), values)}
  def __call__(self, features):  
    # This function wraps tokenize_function with tf.py_function  

    for k in self.data_keys:
      features_to_tokenize = {name: value for name, value in features.items() if name==k}

      dtypes = [tf.int32]*len(features_to_tokenize)
      print(f"features in TokenizeOp = {features}")
      print(f"features in features_to_tokenize = {features_to_tokenize}")
      
      values =  tf.py_function(func=self.tokenize_function, inp=features_to_tokenize.values(), Tout=dtypes) 

      tokenized_features = {k:v for k,v in zip(features_to_tokenize.keys(), values)}
      for k in tokenized_features.keys():
          features.update({k: tokenized_features[k]})

    return features

  def tokenize_function(self, features: Features) -> Features:
    print(f"features in tokenize_function = {features}")
    tokenized_feature = self.sp_tokenizer.encode(features.numpy().decode('utf-8'))
    print(f"tokenized_feature = {tokenized_feature}")
    print(f"tokenized_feature[0] type= {type(tokenized_feature[0])}")
    tokenized_feature_numpy = np.array(tokenized_feature).astype(np.int32)
    return tf.convert_to_tensor(tokenized_feature_numpy)
    for k in features.keys():
      print(f"features[{k}] = {features[k]}")
      # features[k] = self.sp_tokenizer.tokenize(features[k])
      features[k] = tf.convert_to_tensor(self.sp_tokenizer.tokenize(features[k].numpy().decode('utf-8')))
    return tuple(features.values())

  # def tokenize_function(self, **features: Features) -> Features:
  #   for k in self.data_keys:
  #     print(f"features[{k}] = {features[k]}")
  #     # features[k] = self.sp_tokenizer.tokenize(features[k])
  #     features[k] = tf.convert_to_tensor(self.sp_tokenizer.tokenize(features[k].numpy().decode('utf-8')))
  #   return tuple(features.values())
