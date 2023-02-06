# pylint: disable=missing-module-docstring
from collections import OrderedDict

import sys
import yaml

import jax

def string_to_bool(s : str):
  if s.lower() == "true":
    return True
  if s.lower() == "false":
    return False
  raise ValueError(f"Can't convert {s} to bool")

_yaml_types_to_parser = {str : str, int : int, float : float, bool : string_to_bool}

_config = None
config = None

def _lists_to_tuples(l):
  return tuple(_lists_to_tuples(x) for x in l) if isinstance(l, list) else l

class _HyperParameters():
  # pylint: disable=missing-class-docstring
  def __init__(self, argv, **kwargs):
    with open(argv[1], "r", encoding="utf-8") as yaml_file:
      raw_data_from_yaml = yaml.safe_load(yaml_file)
    raw_data_from_cmd_line = self._load_kwargs(argv, **kwargs)

    for k in raw_data_from_cmd_line:
      if k not in raw_data_from_yaml:
        raise ValueError(
            f"Key {k} was passed at the command line but isn't in config."
        )

    raw_keys = OrderedDict()
    for k in raw_data_from_yaml:
      if k in raw_data_from_cmd_line and not isinstance(raw_data_from_cmd_line[k], type(raw_data_from_yaml[k])) and \
                                         type(raw_data_from_yaml[k]) not in _yaml_types_to_parser:
        raise ValueError(
            f"For key '{k}', type {type(raw_data_from_yaml[k])} not in {_yaml_types_to_parser.keys()}, can't pass"
            " at the command line"
        )

      if k in raw_data_from_cmd_line and isinstance(raw_data_from_cmd_line[k], type(raw_data_from_yaml[k])):
        raw_keys[k] = raw_data_from_cmd_line[k] # take the raw data, no type conversion
      elif k in raw_data_from_cmd_line:
        raw_keys[k] = _yaml_types_to_parser[type(raw_data_from_yaml[k])](
            raw_data_from_cmd_line[k]
        )  # take the command line value, but type it like the config value.
      else:
        raw_keys[k] = raw_data_from_yaml[k]

    _HyperParameters.user_init(raw_keys)
    self.keys = raw_keys

  def _load_kwargs(self, argv, **kwargs):
    args_dict = dict(a.split("=") for a in argv[2:])
    args_dict.update(kwargs)
    return args_dict

  @staticmethod
  def user_init(raw_keys):
    '''Transformations between the config data and configs used at runtime'''
    raw_keys["dtype"] = jax.numpy.dtype(raw_keys["dtype"])
    run_name = raw_keys["run_name"]
    assert run_name, "Erroring out, need a real run_name"
    base_output_directory = raw_keys["base_output_directory"]
    raw_keys["tensorboard_dir"] = (
        f"{base_output_directory}/{run_name}/tensorboard/"
    )
    raw_keys["checkpoint_dir"] = (
        f"{base_output_directory}/{run_name}/checkpoints/"
    )
    raw_keys["logical_axis_rules"] = _lists_to_tuples(raw_keys["logical_axis_rules"])
    raw_keys["data_sharding"] = _lists_to_tuples(raw_keys["data_sharding"])
    raw_keys['emb_dim'] = raw_keys['scale'] * raw_keys['base_emb_dim']
    raw_keys['num_heads'] = raw_keys['scale'] * raw_keys['base_num_heads']
    raw_keys['mlp_dim'] = raw_keys['scale'] * raw_keys['base_mlp_dim']
    raw_keys['num_decoder_layers'] = raw_keys['scale'] * raw_keys['base_num_decoder_layers']



class HyperParameters(): # pylint: disable=missing-class-docstring
  def __init__(self):
    pass

  def __getattr__(self, attr):
    if attr not in _config.keys:
      raise ValueError(f"Requested key {attr}, not in config")
    return _config.keys[attr]

  def __setattr__(self, attr, value):
    raise ValueError


def initialize(argv, **kwargs):
  global _config, config
  _config = _HyperParameters(argv, **kwargs)
  config = HyperParameters()

if __name__ == "__main__":
  initialize(sys.argv)
  print(config.steps)
  r = range(config.steps)
