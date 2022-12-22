from collections import OrderedDict

import sys
import yaml

import jax


_allowed_command_line_types = [str, int, float]

_config = None
config = None

class _HyperParameters():
    def __init__(self, argv):
        raw_data_from_yaml = yaml.safe_load(open(argv[1]))
        raw_data_from_cmd_line = self._load_kwargs(argv)

        for k in raw_data_from_cmd_line:
            if k not in raw_data_from_yaml:
                raise ValueError(f"Key {k} was passed at the command line but isn't in config.")

        raw_keys = OrderedDict()
        for k in raw_data_from_yaml:
            if type(k) not in _allowed_command_line_types:
                raise ValueError(f"Type {type(k)} not in {_allowed_command_line_types}, can't pass as at the command line")

            if k in raw_data_from_cmd_line:
                raw_keys[k] = type(raw_data_from_yaml[k])(raw_data_from_cmd_line[k]) # take the command line value, but type it like the config value.
            else:
                raw_keys[k] = raw_data_from_yaml[k]

        _HyperParameters.user_init(raw_keys)
        self.keys = raw_keys
    
    def _load_kwargs(self, argv):
        return dict(a.split('=') for a in sys.argv[2:])

    @staticmethod
    def user_init(raw_keys):
        raw_keys["dtype"] = jax.numpy.dtype(raw_keys["dtype"])
        run_name = raw_keys["run_name"]
        assert run_name != "", "Erroring out, need a real run_name"
        base_output_directory = raw_keys["base_output_directory"]
        raw_keys["tensorboard_dir"] = f"{base_output_directory}/{run_name}/tensorboard/"
        raw_keys["checkpoint_dir"] = f"{base_output_directory}/{run_name}/checkpoints/"


class HyperParameters():
    def __init__(self):
        pass
    
    def __getattr__(self, attr):
        global _config
        if attr not in _config.keys:
            raise ValueError(f"Requested key {attr}, not in config")
        return _config.keys[attr]

    def __setattr__(self, attr, value):
        raise ValueError

def initialize(argv):
    global _config, config
    _config = _HyperParameters(argv)
    config = HyperParameters()

if __name__ == "__main__":
    initialize(sys.argv)
    print(config.steps)
    r = range(config.steps)

