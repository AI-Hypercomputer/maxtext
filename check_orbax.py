"""check_orbax.py
Print the contents (keys and shapes) of orbax checkpoint by reading metadata

Example command:
python check_orbax.py /tmp/gcsfuse/llama2-7b/2025-06-22/scanned/0/items
python check_orbax.py /tmp/gcsfuse/llama2-7b/2025-06-22/unscanned/checkpoints/0/items
"""

from etils import epath
import orbax.checkpoint as ocp
import re
import sys


def natural_sort_key(s: str):
  """
  Utility to sort parameter names, by splitting the string into text and number chunks.
  """
  return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)]


def check_orbax_structure(path: str):
  """
  inspect orbax checkpoint by reading metadata
  """
  path = epath.Path(path)
  print(path)
  metadata = ocp.StandardCheckpointer().metadata(path)

  # k: name tuple, v: array meta data
  dictionary = ocp.tree.to_flat_dict(metadata)
  # check params only (skip opt_state): name and shape
  dictionary = {".".join(k): v.shape for k, v in dictionary.items() if k[0] == "params"}
  # sort layer name 1, 2 ...., 10 (rather than 1, 10, 2, ...)
  for k in sorted(dictionary, key=natural_sort_key):
    print(f"key | {k} | {dictionary[k]}")

  return metadata


if __name__ == "__main__":
  check_orbax_structure(sys.argv[1])