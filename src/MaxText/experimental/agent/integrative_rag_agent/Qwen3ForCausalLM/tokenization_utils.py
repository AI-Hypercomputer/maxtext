# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Check out the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""UROMAN_IMPORT_ERROR = """
{0} requires the uroman library but it was not found in your environment. You can install it with pip:
`pip install uroman`. Please note that you may need to restart your runtime after installation.
"""
# The _sentencepiece_available global variable is assumed to be defined in the same
# way as in the original PyTorch file, using standard Python importlib utilities.
# No direct JAX or PyTorch dependencies are involved in this function.

def is_sentencepiece_available():
  return _sentencepiece_available

import importlib.util

# The availability of the 'rjieba' package is determined once at module import time.
_rjieba_available = importlib.util.find_spec("rjieba") is not None


def is_rjieba_available() -> bool:
  """Checks if the rjieba package is installed and available."""
  return _rjieba_available
