#! /bin/bash -eu

cfg="$1"
MaxText/train.py "$cfg" steps=10 run_name=test_run global_parameter_scale=1 load_from_other_directory="" load_from_other_directory_step=-1
