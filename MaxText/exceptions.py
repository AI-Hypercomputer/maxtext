# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for MaxText."""


class StopTraining(Exception):
  """Custom exception to halt a training process."""

  def __init__(self, reason):
    super().__init__(reason)
