# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for decomposed_transport.py."""

import threading
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.trainers.diloco.decomposed_transport import (
    LearnerTransport,
    ThreadedTransportManager,
    TransportClosedError,
    TransportProtocolError,
)


def _single_device_mesh():
  return jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("model",))


def _direction(manager, direction):
  if direction == "learner_to_syncer":
    return (
        lambda step, data: manager.send_to_syncer(0, step, 0, data),
        lambda step: manager.recv_from_learner(0, step, 0),
    )
  return (
      lambda step, data: manager.send_to_learner(0, step, 0, data),
      lambda step: manager.recv_from_syncer(0, step, 0),
  )


@pytest.mark.parametrize("direction", ("learner_to_syncer", "syncer_to_learner"))
def test_channel_backpressures_at_capacity(direction):
  manager = ThreadedTransportManager(num_learners=1, max_pending_fragments=1)
  send, recv = _direction(manager, direction)
  send(step=1, data="first")

  started = threading.Event()
  finished = threading.Event()
  errors = []

  def send_second():
    started.set()
    try:
      send(step=2, data="second")
    except Exception as error:  # pylint: disable=broad-exception-caught
      errors.append(error)
    finally:
      finished.set()

  producer = threading.Thread(target=send_second)
  producer.start()

  assert started.wait(timeout=1)
  assert not finished.wait(timeout=0.2), "second payload bypassed the channel capacity"
  assert recv(step=1) == "first"
  assert finished.wait(timeout=1), "producer did not resume after capacity was released"
  producer.join(timeout=1)

  assert not errors
  assert recv(step=2) == "second"


@pytest.mark.parametrize("direction", ("learner_to_syncer", "syncer_to_learner"))
def test_channel_is_strict_fifo(direction):
  manager = ThreadedTransportManager(num_learners=1, max_pending_fragments=2)
  send, recv = _direction(manager, direction)

  send(step=1, data="first")
  send(step=2, data="second")

  assert recv(step=1) == "first"
  assert recv(step=2) == "second"


def test_out_of_order_payload_raises_protocol_error_without_leaking_capacity():
  manager = ThreadedTransportManager(num_learners=1, max_pending_fragments=1)
  manager.send_to_syncer(learner_idx=0, step=2, fragment_id=0, data="unexpected")

  with pytest.raises(TransportProtocolError, match=r"sent \(2, 0\); expected \(1, 0\)"):
    manager.recv_from_learner(learner_idx=0, step=1, fragment_id=0)

  manager.send_to_syncer(learner_idx=0, step=3, fragment_id=0, data="next")
  assert manager.recv_from_learner(learner_idx=0, step=3, fragment_id=0) == "next"


def test_close_cancels_a_producer_waiting_for_capacity():
  manager = ThreadedTransportManager(num_learners=1, max_pending_fragments=1)
  manager.send_to_syncer(learner_idx=0, step=1, fragment_id=0, data="first")

  started = threading.Event()
  finished = threading.Event()
  errors = []

  def blocked_send():
    started.set()
    try:
      manager.send_to_syncer(learner_idx=0, step=2, fragment_id=0, data="second")
    except Exception as error:  # pylint: disable=broad-exception-caught
      errors.append(error)
    finally:
      finished.set()

  producer = threading.Thread(target=blocked_send)
  producer.start()
  assert started.wait(timeout=1)
  assert not finished.wait(timeout=0.2)

  manager.close()

  assert finished.wait(timeout=1), "transport close did not wake the blocked producer"
  producer.join(timeout=1)
  assert len(errors) == 1
  assert isinstance(errors[0], TransportClosedError)


def test_async_send_reserves_capacity_before_device_put():
  manager = ThreadedTransportManager(num_learners=1, max_pending_fragments=1)
  cpu_mesh = _single_device_mesh()
  sharding = jax.sharding.NamedSharding(cpu_mesh, jax.sharding.PartitionSpec())
  fragment = {"weight": jax.device_put(jnp.arange(8), sharding)}
  learner = LearnerTransport(manager, learner_idx=0, local_cpu_mesh=cpu_mesh)

  real_device_put = jax.device_put
  offload_started = threading.Event()

  def counted_device_put(*args, **kwargs):
    offload_started.set()
    return real_device_put(*args, **kwargs)

  second_started = threading.Event()
  second_finished = threading.Event()
  errors = []

  def send_second():
    second_started.set()
    try:
      learner.send_to_syncer_async(step=2, fragment_id=0, data=fragment)
    except Exception as error:  # pylint: disable=broad-exception-caught
      errors.append(error)
    finally:
      second_finished.set()

  with mock.patch(
      "maxtext.trainers.diloco.decomposed_transport.jax.device_put",
      side_effect=counted_device_put,
  ) as device_put:
    learner.send_to_syncer_async(step=1, fragment_id=0, data=fragment)
    assert offload_started.wait(timeout=1)
    assert device_put.call_count == 1

    producer = threading.Thread(target=send_second)
    producer.start()
    assert second_started.wait(timeout=1)
    assert not second_finished.wait(timeout=0.2)
    assert device_put.call_count == 1, "second CPU payload was allocated before capacity was available"

    assert manager.recv_from_learner(learner_idx=0, step=1, fragment_id=0) is not None
    assert second_finished.wait(timeout=1)
    producer.join(timeout=1)
    assert device_put.call_count == 2
    assert manager.recv_from_learner(learner_idx=0, step=2, fragment_id=0) is not None

  assert not errors
  learner.close()


def test_async_send_error_is_propagated_and_cancels_protocol():
  manager = ThreadedTransportManager(num_learners=1, max_pending_fragments=1)
  cpu_mesh = _single_device_mesh()
  sharding = jax.sharding.NamedSharding(cpu_mesh, jax.sharding.PartitionSpec())
  fragment = {"weight": jax.device_put(jnp.arange(8), sharding)}
  learner = LearnerTransport(manager, learner_idx=0, local_cpu_mesh=cpu_mesh)
  failure_reached = threading.Event()

  def fail_block_until_ready(_):
    failure_reached.set()
    raise RuntimeError("offload failed")

  with mock.patch(
      "maxtext.trainers.diloco.decomposed_transport.jax.block_until_ready",
      side_effect=fail_block_until_ready,
  ):
    learner.send_to_syncer_async(step=1, fragment_id=0, data=fragment)
    assert failure_reached.wait(timeout=1)
    with pytest.raises(RuntimeError, match="offload failed"):
      learner.close()

  # The protocol cannot recover from a missing fragment. The worker must close
  # every channel immediately so that syncer/peer waits cannot deadlock while
  # the learner is still unwinding toward its next Future poll.
  with pytest.raises(TransportClosedError):
    manager.recv_from_learner(learner_idx=0, step=1, fragment_id=0)
  with pytest.raises(TransportClosedError):
    manager.send_to_syncer(learner_idx=0, step=2, fragment_id=0, data="recovered")
