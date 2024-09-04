"""This module provides a HuggingFace dataiterator CheckpointHandler for integration with Orbax."""
import dataclasses
import json
from typing import Any, Optional, Dict

from etils import epath
from datasets import IterableDataset
import jax

# Ipmlements orbax.checkpoint.CheckpointHandler.
class HFCheckpointHandler:
  """Orbax CheckpointHandler for HF IterableDataset."""

  def save(
      self,
      directory: epath.Path,
      # `item` is for backwards compatibility with older Orbax API, see
      # https://orbax.readthedocs.io/en/latest/api_refactor.html.
      item: Optional[Dict] = None,
      args: Any = None,
  ):
    """Saves the given iterator to the checkpoint in `directory`."""
    item = item or args.item
    #import pdb; pdb.set_trace()
    state = json.dumps(item, indent=4)

    filename = (
        directory
        / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    )
    filename.write_text(state)

  def restore(
      self,
      directory: epath.Path,
      item: Optional[IterableDataset] = None,
      args: Any = None,
  ) -> IterableDataset:
    """Restores the given iterator from the checkpoint in `directory`."""
    item = item or args.item  # pytype:disable=attribute-error
    filename = (
        directory
        / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    )
    if not filename.exists():
      raise ValueError(f"File {filename} does not exist.")
    state = filename.read_text()
    state = json.loads(state)
    item.load_state_dict(state)
    return item

  # Required by interface but not supported by PyGrain checkpoints.
  def structure(self, directory: epath.Path) -> Any:
    del directory
    return None

  # Required by interface.

  def metadata(self, directory: epath.Path) -> Optional[Any]:
    del directory
    return None

  def finalize(self, directory: epath.Path):
    pass

  def close(self):
    pass

try:
  # Register the handler to be used with the new checkpointing API if Orbax is
  # present.
  import orbax.checkpoint as ocp  # pylint:disable=g-import-not-at-top # pytype:disable=import-error

  @ocp.args.register_with_handler(HFCheckpointHandler, for_save=True)  # pytype:disable=wrong-arg-types
  @dataclasses.dataclass
  class HFCheckpointSave(ocp.args.CheckpointArgs):
    item: Any

  @ocp.args.register_with_handler(HFCheckpointHandler, for_restore=True)  # pytype:disable=wrong-arg-types
  @dataclasses.dataclass
  class HFCheckpointRestore(ocp.args.CheckpointArgs):
    item: Any

except (ImportError, TypeError):
  pass
