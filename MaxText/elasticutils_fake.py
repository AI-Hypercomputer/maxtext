"""Utilities for elastic training."""
import logging
from typing import Any, Optional, Sequence

from elasticutils import ElasticUtils
from elasticutils import timeit
import jax


PyTree = Any

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

#  pylint: disable=logging-fstring-interpolation


class FakeElasticUtils(ElasticUtils):
  """Utility class for elastic training.

  This class will simulate slices going down and coming back up.
  """

  def __init__(
      self,
      devices: Sequence[jax.Device],
      total_slice_count: int,
      save_period: Optional[int] = None,
      reshard_check_period: Optional[int] = None,
      max_failures: Optional[int] = None,
  ):
    self.fake_good_slice_indices = set(d.slice_index for d in devices)

    super().__init__(
        devices,
        total_slice_count,
        save_period,
        reshard_check_period,
        max_failures,
    )

  def update_good_slice_indices(self, good_slice_indices: set[int]):
    """Start step handler."""
    self.fake_good_slice_indices = good_slice_indices
    logger.info(f"Updated: {self.fake_good_slice_indices=}")

  @timeit
  def get_slice_availability(self) -> set[int]:
    """Returns the set of good and bad slices."""
    good_slice_indices = self.fake_good_slice_indices

    logger.info(f"{good_slice_indices=}")

    return good_slice_indices

  # Does not work
  @staticmethod
  def put_array_jit(
      arr: jax.Array,
      dst_sharding: jax.sharding.Sharding,
      donate_input: bool,
  ):
    if not isinstance(dst_sharding, jax.sharding.Sharding):
      raise ValueError("`sharding` must contain only `Sharding` instances.")

    return jax.jit(
        lambda x: x,
        out_shardings=dst_sharding,
        donate_argnums=(0,) if donate_input else (),
    )(arr)

  # Slower than actually resharding
  @staticmethod
  def put_array_fake0(
      arr: jax.Array,
      dst_sharding: jax.sharding.Sharding,
      donate_input: bool,
  ):
    if not isinstance(dst_sharding, jax.sharding.Sharding):
      raise ValueError("`sharding` must contain only `Sharding` instances.")
    return jax.numpy.ones_like(arr, device=dst_sharding)

