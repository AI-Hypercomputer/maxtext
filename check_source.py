import torch
from safetensors import safe_open
import argparse
import os
import pathlib
import sys

import logging
import time
import re
from tqdm import tqdm

"""
# llama2-7b: pth
CKPT_TYPE=pth; CKPT_DIR=~/tmp/gcsfuse-maxtext-llama/llama2-7b/meta-ckpt; LOG=~/log/check-llama2-7b-meta-ckpt.log
python3 ~/maxtext/check_source.py --ckpt_dir=$CKPT_DIR --checkpoint_type=$CKPT_TYPE --log_path=$LOG

# scout: safetensor
CKPT_TYPE=safetensors; CKPT_DIR=/home/shuningjin/llama4-17b-16e/hf-bf16; LOG=~/log/check-llama4-17b-16e-hf-bf16.log
python3 ~/maxtext/check_source.py --ckpt_dir=$CKPT_DIR --checkpoint_type=$CKPT_TYPE --log_path=$LOG

# maverick: safetensor
CKPT_TYPE=safetensors; CKPT_DIR=/home/shuningjin/llama4-17b-128e/hf-bf16; LOG=~/log/check-llama4-17b-128e-hf-bf16.log
python3 ~/maxtext/check_source.py --ckpt_dir=$CKPT_DIR --checkpoint_type=$CKPT_TYPE --log_path=$LOG

# scout: pth
CKPT_TYPE=pth; CKPT_DIR=/home/shuningjin/llama4-17b-16e/meta-bf16; LOG=~/log/check-llama4-17b-16e-meta-bf16.log
python3 ~/maxtext/check_source.py --ckpt_dir=$CKPT_DIR --checkpoint_type=$CKPT_TYPE --log_path=$LOG

# maverick: pth
CKPT_TYPE=pth; CKPT_DIR=/home/shuningjin/llama4-17b-128e/meta-bf16; LOG=~/log/check-llama4-17b-128e-meta-bf16.log
python3 ~/maxtext/check_source.py --ckpt_dir=$CKPT_DIR --checkpoint_type=$CKPT_TYPE --log_path=$LOG
"""


def init_log(log_path=""):
  # logging.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=logging.INFO)
  # logging.getLogger().setLevel(logging.INFO)
  # formatter
  log_format = "%(asctime)s %(levelname)s: %(message)s"
  date_format = "%Y-%M-%D %I:%M:%S %p"
  formatter = logging.Formatter(log_format, date_format)
  # logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  # clear handlers
  logger.handlers.clear()
  # stream_hander
  stream_handler = logging.StreamHandler()
  # stream_handler.setFormatter(formatter)
  logger.addHandler(stream_handler)

  # file_handler
  if log_path:
    file_handler = logging.FileHandler(log_path, mode="w")
    # file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def natural_sort_key(s):
  """
  Utility to sort parameter names, by splitting the string into text and number chunks.
  """
  s = str(s)
  return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)]


def print_nested_keys(data, prefix=""):
  """
  Prints nested keys of a dictionary-like structure in a directory-like format.
  Args:
      data: The dictionary-like structure to traverse.
      prefix: The current path prefix.
  """
  if isinstance(data, dict):
    for key in sorted(data, key=natural_sort_key):
      current_path = f"{prefix}{key}."
      print_nested_keys(data[key], current_path)
  else:
    logging.info(f"key | {prefix.rstrip('.')} | {tuple(data.shape)}")
    # logging.info(f"\t{tuple(data.shape)}")


def load_pth_checkpoint(ckpt_paths):
  chkpt_vars_raw = {}
  for i, ckpt_path in tqdm(enumerate(ckpt_paths)):
    logging.info(f"Loading checkpointpath {i+1} of {len(ckpt_paths)}: {ckpt_path}")
    try:
      checkpoint = torch.load(ckpt_path, map_location="cpu")
      chkpt_vars_raw[int(ckpt_path.name.split(".", maxsplit=2)[1])] = checkpoint
    except Exception as e:
      logging.error(e)
  logging.info("")
  print_nested_keys(chkpt_vars_raw)


def load_safetensors_checkpoint(ckpt_paths):
  chkpt_vars_raw = {}
  for i, ckpt_path in tqdm(enumerate(ckpt_paths)):
    logging.info(f"Loading checkpoint path {i+1} of {len(ckpt_paths)}: {ckpt_path}")
    try:
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          chkpt_vars_raw[k] = f.get_tensor(k)
    except Exception as e:
      logging.error(e)
  logging.info("")
  print_nested_keys(chkpt_vars_raw)


def main(argv):
  parser = argparse.ArgumentParser(
      description="Print the contents (keys and shapes) of safetensors and PyTorch checkpoint files."
  )
  parser.add_argument("--ckpt_dir", type=str, required=True)
  parser.add_argument("--checkpoint_type", type=str, choices=["pth", "safetensors"], required=True)
  parser.add_argument("--log_path", type=str, default="", required=False)
  args = parser.parse_args(argv)

  # print(args)
  assert os.path.isdir(args.ckpt_dir), f"{args.ckpt_dir} is not a directory"

  init_log(args.log_path)
  logging.info(args)

  # if args.checkpoint_type not in CHECKPOINT_TYPES:
  #   raise NotImplementedError

  ckpt_paths = sorted(pathlib.Path(args.ckpt_dir).glob(f"[!.]*.{args.checkpoint_type}"))
  start_time = time.time()

  if args.checkpoint_type == "safetensors":
    load_safetensors_checkpoint(ckpt_paths)
  else:
    assert args.checkpoint_type == "pth"
    load_pth_checkpoint(ckpt_paths)

  end_time = time.time()
  elapse = end_time - start_time
  logging.info(f"\nelapse: {elapse / 60: .2f} minutes")


main(sys.argv[1:])
