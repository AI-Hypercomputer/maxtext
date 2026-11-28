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
r"""Numerical parity between the MPMD pipeline and its SPMD reference.

Pipeline parallelism must not change what the model computes -- only where it
computes it. This binary runs the *same* training step (see
:mod:`train_step`) twice from the same initial parameters and the same data:

  1. partitioned across stage meshes by PartIR MPMD, and
  2. as an ordinary single program.

It then compares the per-step losses, the step-1 gradients and the full
post-update parameter tree. Both runs share the model and the loss code; the
MPMD step additionally pre-gathers each stage's weights once per step and, with
data parallelism, reduce-scatters their gradients in the compute dtype
(``models.pregather_params`` / ``build_grad_carry``), where the unsharded
reference accumulates float32. In float32 the remaining differences are
therefore the partitioning's own reassociation; in bfloat16 they also include
that precision change.

How much reassociation moves the numbers depends on the hardware and the model,
so the SPMD reference also runs a second time with a different microbatch
split (the same arithmetic, summed in another order): the *noise floor*. Each
step's loss and each tensor passes within its fixed tolerance or within
``--noise_floor_factor`` times the floor at the same step or tensor. On TPU
even ``dtype=float32`` with ``matmul_precision=highest`` leaves MoE gradients
split-dependent at the percent level (SPMD with 8 vs 4 microbatches: 0.8-3.5%
on the router at step 1, against 5e-7 on CPU). Top-k routing adds jumps: where
two experts' scores nearly tie, MPMD's rounding can pick the other one, which
an SPMD floor need not do (the report notes it, with the override that routes
every token to all experts). A non-finite loss, gradient or parameter always
fails, and a floor that is not finite is ignored.

The MPMD side compiles only on Pathways, so run it against a Pathways backend,
e.g. a TPU VM's Pathways in docker (``local_pathways.sh up``); this checks
float32 at the default matmul precision (see :func:`parity_notes` and the
README's Parity section)::

    JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
      python3 -m maxtext.experimental.mpmd.partir.parity --pp=4 \
      --num_microbatches=4 --per_device_batch_size=4 --steps=3 \
      --model_name=simple --maxtext_overrides=dtype=float32
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
from typing import Any, NamedTuple

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pathwaysutils

from maxtext.common.common_types import DecoderBlockType
from maxtext.utils import max_logging

from maxtext.experimental.mpmd.partir import hparams as hparams_lib
from maxtext.experimental.mpmd.partir import runner as runner_lib
from maxtext.experimental.mpmd.partir import train_step as train_step_lib


def define_flags(flag_values: flags.FlagValues = flags.FLAGS) -> None:
  """Registers the parity check's flags, the hyperparameters' included, on `flag_values`."""
  hparams_lib.define_flags(flag_values)
  fv = {"flag_values": flag_values}
  flags.DEFINE_float(
      "loss_tolerance",
      1e-4,
      "Fixed tolerance on each step's relative MPMD-vs-SPMD loss difference (absolute below a loss of 1); a step also"
      " passes within --noise_floor_factor times the floor's difference up to that step.",
      **fv,
  )
  flags.DEFINE_float(
      "param_tolerance",
      1e-2,
      "Fixed tolerance on each parameter tensor's relative L2 difference after the last step; a tensor also passes"
      " within --noise_floor_factor times its own floor. Looser than the loss tolerance: AdamW divides by sqrt(v),"
      " which amplifies the difference for weights whose gradient is near zero.",
      **fv,
  )
  flags.DEFINE_float(
      "grad_tolerance",
      1e-2,
      "Fixed tolerance on each step-1 gradient tensor's relative L2 difference (read from AdamW's first moment after"
      " step 1, so before any update has amplified a difference); a tensor also passes within --noise_floor_factor"
      " times its own floor.",
      **fv,
  )
  flags.DEFINE_integer(
      "noise_floor_microbatches",
      -1,
      "Microbatches of the second SPMD run that measures the noise floor. -1: num_microbatches over its smallest"
      " prime factor (8 -> 4, 9 -> 3, 5 -> 1); 0: no floor, only the fixed tolerances.",
      **fv,
  )
  flags.DEFINE_float(
      "noise_floor_factor",
      4.0,
      "A step or tensor also passes within this many times the noise floor at the same step (the floor's largest"
      " difference up to it) or tensor. Once AdamW has changed an MoE model's routing, the later steps' losses"
      " diverge chaotically, so the floor gives their scale, not their value. Top-k MoE routing can flip under"
      " MPMD's rounding where an SPMD floor does not; see the report's note.",
      **fv,
  )


# optax.adamw's default b1 (train_step.make_optimizer keeps the defaults): after
# one step its first moment is (1 - b1) * g.
_ADAM_B1 = 0.9


def relative_difference(value: float, reference: float) -> float:
  """Returns the loss difference relative to ``max(|reference|, 1)``.

  Absolute below a loss of 1: a model that memorizes its batch drives the loss
  toward 0, where relative noise grows without bound (deepseek2-16b cut to two
  layers went 10.87 -> 2.30 -> 0.031 at lr 1e-3, and a 1.3e-5 difference at
  0.031 read 4.3e-4). Inf if either value is not finite.
  """
  if not (math.isfinite(value) and math.isfinite(reference)):
    return math.inf
  return abs(value - reference) / max(abs(reference), 1.0)


class TreeDiff(NamedTuple):
  """What :func:`_compare_params` found; see its docstring."""

  rel_l2: float
  worst_l2_path: str
  max_abs: float
  max_rel: float
  worst_path: str
  # (rel_l2, path, reference_norm) of every leaf, worst first; rel_l2 is inf
  # for a leaf with a non-finite value on either side.
  leaves: tuple[tuple[float, str, float], ...]

  @property
  def finite(self) -> bool:
    """Whether every leaf's difference is finite."""
    return all(math.isfinite(rel) for rel, _, _ in self.leaves)


@dataclasses.dataclass(frozen=True)
class Comparison:
  """How far one run is from the SPMD reference.

  Attributes:
    losses: Per-step losses of the run.
    reference_losses: Per-step losses of the SPMD reference.
    grads: Step-1 gradients compared, recovered from AdamW's first moment
      (``(1 - b1) * g`` after one step): unlike the parameters they have not
      been through an update, so they compare the backward passes themselves.
    params: Parameters after the last step compared. The relative L2 is gated,
      not the elementwise maximum; see :func:`_compare_params`.
  """

  losses: tuple[float, ...]
  reference_losses: tuple[float, ...]
  grads: TreeDiff
  params: TreeDiff

  @property
  def loss_diffs(self) -> tuple[float, ...]:
    """Per-step loss differences (see :func:`relative_difference`)."""
    return tuple(relative_difference(loss, reference) for loss, reference in zip(self.losses, self.reference_losses))

  @property
  def max_rel_loss_diff(self) -> float:
    """Largest per-step loss difference."""
    return max(self.loss_diffs, default=0.0)

  @property
  def finite(self) -> bool:
    """Whether every loss, gradient and parameter difference is finite."""
    return math.isfinite(self.max_rel_loss_diff) and self.grads.finite and self.params.finite


class Gate(NamedTuple):
  """One gated metric at its worst step or tensor, relative to its bound.

  Attributes:
    name: ``loss``, ``grads`` or ``params``.
    value: The MPMD difference there.
    bound: Its bound there: ``max(tolerance, factor * floor)``.
    floor: The floor's difference there, or None without a usable floor.
    where: The step or tensor.
  """

  name: str
  value: float
  bound: float
  floor: float | None
  where: str

  @property
  def passed(self) -> bool:
    """Whether the value is finite and within its bound."""
    return math.isfinite(self.value) and self.value <= self.bound

  @property
  def excess(self) -> float:
    """``value / bound``: above 1 fails; inf for a non-finite value."""
    if not math.isfinite(self.value):
      return math.inf
    if self.bound <= 0:
      return math.inf if self.value > 0 else 0.0
    return self.value / self.bound


def _worst(candidates: list[Gate]) -> Gate:
  """Returns the candidate furthest over (or least under) its bound."""
  return max(candidates, key=lambda gate: gate.excess)


@dataclasses.dataclass(frozen=True)
class ParityResult:
  """Outcome of a parity comparison.

  Attributes:
    mpmd: The MPMD run against the SPMD reference.
    floor: The SPMD reference run with ``floor_microbatches`` microbatches
      against the reference, or None when no floor was measured.
    floor_microbatches: Microbatch count of the noise-floor run.
    num_params: Number of parameter leaves compared.
    notes: Caveats about this configuration, printed with the report.
  """

  mpmd: Comparison
  floor: Comparison | None
  floor_microbatches: int
  num_params: int
  notes: tuple[str, ...] = ()

  @property
  def usable_floor(self) -> Comparison | None:
    """The floor, unless it is missing or has a non-finite difference."""
    return self.floor if self.floor is not None and self.floor.finite else None

  def gates(
      self,
      loss_tolerance: float,
      param_tolerance: float,
      grad_tolerance: float = math.inf,
      floor_factor: float = 0.0,
  ) -> tuple[Gate, Gate, Gate]:
    """Returns the loss, gradient and parameter gates at their worst points.

    Each step's loss is bounded by ``max(loss_tolerance, floor_factor x the
    floor's largest loss difference up to that step)``, so a chaotic late step
    of the floor cannot loosen step 1. Each tensor is bounded by its own
    tolerance or ``floor_factor`` x its own floor, so one noisy MoE router
    cannot loosen the bound of every other tensor.

    Args:
      loss_tolerance: Fixed tolerance on each step's loss difference (see
        :func:`relative_difference`).
      param_tolerance: Fixed tolerance on each parameter tensor's relative L2
        difference.
      grad_tolerance: Fixed tolerance on each step-1 gradient tensor's relative
        L2 difference.
      floor_factor: A step or tensor also passes within this many times the
        floor at the same step or tensor.
    """
    floor = self.usable_floor
    floor_losses = floor.loss_diffs if floor is not None else ()
    loss_gates = []
    running = 0.0
    for index, value in enumerate(self.mpmd.loss_diffs):
      floor_value = None
      if index < len(floor_losses):
        running = max(running, floor_losses[index])
        floor_value = running
      bound = loss_tolerance
      if floor_value is not None:
        bound = max(loss_tolerance, floor_factor * floor_value)
      loss_gates.append(Gate("loss", value, bound, floor_value, f"step {index + 1}"))

    def tensor_gate(name, mine, theirs, tolerance):
      floors = {path: rel for rel, path, _ in theirs.leaves} if theirs else {}
      candidates = []
      for rel, path, _ in mine.leaves:
        floor_value = floors.get(path)
        bound = tolerance
        if floor_value is not None:
          bound = max(tolerance, floor_factor * floor_value)
        candidates.append(Gate(name, rel, bound, floor_value, path))
      return _worst(candidates) if candidates else Gate(name, 0.0, tolerance, None, "<none>")

    return (
        _worst(loss_gates) if loss_gates else Gate("loss", 0.0, loss_tolerance, None, "<none>"),
        tensor_gate(
            "grads",
            self.mpmd.grads,
            floor.grads if floor is not None else None,
            grad_tolerance,
        ),
        tensor_gate(
            "params",
            self.mpmd.params,
            floor.params if floor is not None else None,
            param_tolerance,
        ),
    )

  def passed(
      self,
      loss_tolerance: float,
      param_tolerance: float,
      grad_tolerance: float = math.inf,
      floor_factor: float = 0.0,
  ) -> bool:
    """Returns whether every step's loss and every tensor is within bounds.

    Args:
      loss_tolerance: See :meth:`gates`.
      param_tolerance: See :meth:`gates`.
      grad_tolerance: See :meth:`gates`.
      floor_factor: See :meth:`gates`.
    """
    return all(gate.passed for gate in self.gates(loss_tolerance, param_tolerance, grad_tolerance, floor_factor))

  def report(
      self,
      loss_tolerance: float,
      param_tolerance: float,
      grad_tolerance: float = math.inf,
      floor_factor: float = 0.0,
  ) -> str:
    """Returns a human readable comparison table.

    Args:
      loss_tolerance: See :meth:`gates`.
      param_tolerance: See :meth:`gates`.
      grad_tolerance: See :meth:`gates`.
      floor_factor: See :meth:`gates`.
    """
    floor = self.usable_floor
    header = f"{'step':>6}  {'MPMD loss':>16}  {'SPMD loss':>16}  {'diff':>12}"
    if floor is not None:
      header += f"  {'floor diff':>14}"
    lines = [header]
    floor_losses = floor.loss_diffs if floor is not None else ()
    for index, (mpmd_loss, spmd_loss, difference) in enumerate(
        zip(self.mpmd.losses, self.mpmd.reference_losses, self.mpmd.loss_diffs)
    ):
      line = f"{index + 1:>6}  {mpmd_loss:>16.9f}  {spmd_loss:>16.9f}" f"  {difference:>12.3e}"
      if index < len(floor_losses):
        line += f"  {floor_losses[index]:>14.3e}"
      lines.append(line)

    tolerances = {
        "loss": loss_tolerance,
        "grads": grad_tolerance,
        "params": param_tolerance,
    }
    lines += [
        "",
        # Not "rel diff": absolute below a loss of 1 (relative_difference).
        "loss diff                 : |MPMD - SPMD| / max(|SPMD|, 1)",
        f"parameter leaves compared : {self.num_params}",
    ]
    if self.floor is not None and floor is None:
      lines.append("noise floor               : ignored, it has a non-finite difference")
    elif floor is not None:
      lines.append(
          "noise floor               : SPMD with" f" {self.floor_microbatches} microbatches against the SPMD reference"
      )
    lines += ["", "gated (worst step or tensor, against its own bound):"]
    for gate in self.gates(loss_tolerance, param_tolerance, grad_tolerance, floor_factor):
      if gate.floor is None:
        rule = "the tolerance"
      else:
        rule = f"max(tolerance {tolerances[gate.name]:.1e}," f" {floor_factor:g} x floor {gate.floor:.3e})"
      lines.append(
          f"  {gate.name:<7}: {gate.value:.3e} at {gate.where}  (bound"
          f" {gate.bound:.3e} = {rule})  {'ok' if gate.passed else 'FAIL'}"
      )
    floor_grads = {path: rel for rel, path, _ in floor.grads.leaves} if floor else {}
    lines += [
        "",
        "diagnostic (elementwise; see _compare_params):",
        f"  max abs param diff      : {self.mpmd.params.max_abs:.3e}",
        (f"  max rel param diff      : {self.mpmd.params.max_rel:.3e}" f"  (at {self.mpmd.params.worst_path})"),
        "  step-1 grads, worst rel L2 (floor; reference ||g||):",
    ]
    for rel, path, norm in self.mpmd.grads.leaves[:5]:
      floor_text = f"{floor_grads[path]:.3e}" if path in floor_grads else "n/a"
      lines.append(f"    {rel:.3e}  {path}  ({floor_text}; {norm:.3e})")
    if self.notes:
      lines += ["", "notes:", *(f"  - {note}" for note in self.notes)]
    return "\n".join(lines)


def noise_floor_microbatches(num_microbatches: int, flag_value: int) -> int:
  """Returns the noise-floor run's microbatch count (0: none); see the flag."""
  if flag_value >= 0:
    return flag_value
  for factor in range(2, num_microbatches + 1):
    if num_microbatches % factor == 0:
      return num_microbatches // factor
  return 0


def _first_moment(opt_state):
  """Returns AdamW's first moment (``mu``) from an optax state tree."""

  def is_adam(node):
    return isinstance(node, optax.ScaleByAdamState)

  for node in jax.tree_util.tree_leaves(opt_state, is_leaf=is_adam):
    if is_adam(node):
      return node.mu
  raise ValueError("The optimizer state holds no ScaleByAdamState.")


def _step1_gradients(opt_state):
  """Returns the step-1 gradients on the host, after exactly one step.

  In the leaves' own dtype: three runs' trees stay alive until they are
  compared, and _compare_params widens one leaf at a time.
  """
  return jax.tree.map(
      lambda mu: np.asarray(jax.device_get(mu)) / (1 - _ADAM_B1),
      _first_moment(opt_state),
  )


def _compare_params(mpmd_params, spmd_params, norm_floor=1.0) -> TreeDiff:
  """Compares two parameter trees.

  The gating metric is the per-tensor relative L2 difference
  ``||mpmd - spmd||_2 / ||spmd||_2`` rather than the largest elementwise
  relative difference, because the latter is not a usable signal once the
  parameters have been through AdamW. AdamW's update is
  ``m / (sqrt(v) + eps)``, so for a weight whose gradient is near zero an
  arbitrarily small perturbation of that gradient -- for instance the
  reassociation of a bf16 reduction caused by sharding -- flips the update by
  up to a full learning rate. A single such weight sends the elementwise
  relative difference to O(1) while the tensor as a whole is unchanged. The
  L2 ratio, in contrast, only grows if the difference is spread over the
  tensor, which is what an actual partitioning bug produces.

  Args:
    mpmd_params: Parameters after the MPMD run.
    spmd_params: Parameters after the SPMD reference run.
    norm_floor: Floor of the relative L2's denominator (see below); gradients,
      whose norms are far below 1, use a small one.

  Returns:
    A :class:`TreeDiff`.

  Raises:
    ValueError: If the trees differ in size or in any leaf's shape.
  """
  max_abs = 0.0
  max_rel = 0.0
  rel_l2 = 0.0
  worst_path = "<none>"
  worst_l2_path = "<none>"
  per_leaf: list[tuple[float, str, float]] = []

  mpmd_leaves = jax.tree_util.tree_flatten_with_path(mpmd_params)[0]
  spmd_leaves = jax.tree_util.tree_flatten_with_path(spmd_params)[0]
  if len(mpmd_leaves) != len(spmd_leaves):
    raise ValueError(f"Parameter trees differ in size: {len(mpmd_leaves)} (MPMD) vs" f" {len(spmd_leaves)} (SPMD).")

  for (path, mpmd_leaf), (_, spmd_leaf) in zip(mpmd_leaves, spmd_leaves):
    mpmd_array = np.asarray(jax.device_get(mpmd_leaf), dtype=np.float64)
    spmd_array = np.asarray(jax.device_get(spmd_leaf), dtype=np.float64)
    if mpmd_array.shape != spmd_array.shape:
      raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(path)}:" f" {mpmd_array.shape} vs {spmd_array.shape}.")
    finite = bool(np.isfinite(mpmd_array).all() and np.isfinite(spmd_array).all())
    absolute = np.abs(mpmd_array - spmd_array)
    # A NaN compares as neither larger nor smaller, so a non-finite leaf is
    # recorded as inf: it must fail every gate, never count as no difference.
    leaf_max_abs = float(absolute.max(initial=0.0)) if finite else math.inf
    max_abs = max(max_abs, leaf_max_abs)

    relative = absolute / np.maximum(np.abs(spmd_array), 1e-6)
    leaf_max_rel = float(relative.max(initial=0.0)) if finite else math.inf
    if leaf_max_rel > max_rel:
      max_rel = leaf_max_rel
      worst_path = jax.tree_util.keystr(path)

    # Floor the denominator at 1.0 so zero-initialised offset parameters (such
    # as Qwen3NextRMSNorm.scale, which starts at 0.0 and is applied as
    # `1.0 + scale`) do not divide a single-learning-rate AdamW step by ~0.
    reference_norm = float(np.linalg.norm(spmd_array.ravel())) if finite else math.nan
    leaf_rel_l2 = float(np.linalg.norm(absolute.ravel())) / max(reference_norm, norm_floor) if finite else math.inf
    per_leaf.append((leaf_rel_l2, jax.tree_util.keystr(path), reference_norm))
    if leaf_rel_l2 > rel_l2:
      rel_l2 = leaf_rel_l2
      worst_l2_path = jax.tree_util.keystr(path)

  per_leaf.sort(key=lambda leaf: leaf[0], reverse=True)
  return TreeDiff(rel_l2, worst_l2_path, max_abs, max_rel, worst_path, tuple(per_leaf))


def _run_reference(runnable, steps):
  """Runs the unsharded SPMD step; returns (losses, step-1 grads, params)."""
  # The reference is deliberately left unsharded: it exists to say what the
  # arithmetic should produce, so the fewer transformations applied to it the
  # better. Its inputs and outputs are committed to the one device it runs on:
  # left uncommitted, a model that constrains shardings inside (DeepSeek's
  # layers) returned step 1's state committed to another sharding, and step 2
  # compiled the program again (3.5 min for deepseek3-tiny with mHC on TPU7x).
  device = jax.sharding.SingleDeviceSharding(runnable.plan.layer_mesh.devices.flat[0])
  step_fn = jax.jit(runnable.train_step, in_shardings=device, out_shardings=device)
  state, batch = jax.device_put((runnable.state, runnable.batch), device)
  losses, grads = [], None
  for step in range(steps):
    state, loss = step_fn(state, batch)
    losses.append(float(loss))
    if step == 0:
      grads = _step1_gradients(state.opt_state)
  return tuple(losses), grads, state.params


def check_parity(
    hparams: hparams_lib.Hparams,
    *,
    config_overrides: dict[str, object] | None = None,
    floor_microbatches: int = 0,
) -> ParityResult:
  """Runs both execution modes (and the noise-floor run) and compares them.

  Args:
    hparams: Hyperparameters; ``steps`` controls how many optimizer steps are
      compared. More steps make the check stricter for dense models, because
      any divergence compounds through the optimizer; an MoE model's routing
      turns it chaotic instead, which the noise floor measures.
    config_overrides: Optional MaxText config overrides forwarded to
      :func:`runner.build`.
    floor_microbatches: Microbatches of the SPMD run that measures the noise
      floor; 0 skips it.

  Returns:
    The :class:`ParityResult`.

  Raises:
    ValueError: If ``hparams.steps < 1`` (nothing would be compared) or
      `floor_microbatches` does not divide the global batch.
  """
  if hparams.steps < 1:
    raise ValueError(f"steps must be at least 1, got {hparams.steps}.")
  mpmd_runnable = runner_lib.build(
      hparams,
      mode=train_step_lib.ExecutionMode.MPMD,
      config_overrides=config_overrides,
      check_microbatches=True,
  )
  spmd_runnable = runner_lib.build(
      hparams,
      mode=train_step_lib.ExecutionMode.SPMD,
      config_overrides=config_overrides,
  )

  if floor_microbatches and mpmd_runnable.batch_size % floor_microbatches:
    # Checked before anything compiles: runner.build would only reject it
    # after the MPMD and SPMD runs.
    raise ValueError(
        f"--noise_floor_microbatches={floor_microbatches} does not divide the"
        f" global batch of {mpmd_runnable.batch_size}."
    )

  max_logging.log("Configuration under test:")
  max_logging.log(mpmd_runnable.describe())

  max_logging.log("Compiling MPMD step...")
  mpmd_lowered = mpmd_runnable.lower()
  mpmd_step = mpmd_lowered.compile()
  mpmd_state, mpmd_batch = runner_lib.place_on_stage_meshes(
      (mpmd_runnable.state, mpmd_runnable.batch),
      mpmd_lowered.function_named_shardings.input_specs,
  )

  mpmd_losses, mpmd_grads = [], None
  for step in range(hparams.steps):
    mpmd_state, mpmd_loss = mpmd_step(mpmd_state, mpmd_batch)
    mpmd_losses.append(float(mpmd_loss))
    max_logging.log(f"  MPMD step {step + 1}/{hparams.steps}: {mpmd_losses[-1]:.9f}")
    if step == 0:
      # Read before the next step donates the state.
      mpmd_grads = _step1_gradients(mpmd_state.opt_state)
  mpmd_params = jax.device_get(mpmd_state.params)
  del mpmd_state

  max_logging.log("Running the SPMD reference...")
  spmd_losses, spmd_grads, spmd_params = _run_reference(spmd_runnable, hparams.steps)
  for step, loss in enumerate(spmd_losses):
    max_logging.log(f"  SPMD step {step + 1}/{hparams.steps}: {loss:.9f}")
  spmd_params = jax.device_get(spmd_params)
  del spmd_runnable

  floor = None
  if floor_microbatches > 0:
    max_logging.log(f"Running the SPMD reference with {floor_microbatches} microbatches" " (noise floor)...")
    floor_runnable = runner_lib.build(
        dataclasses.replace(hparams, num_microbatches=floor_microbatches),
        mode=train_step_lib.ExecutionMode.SPMD,
        config_overrides=config_overrides,
    )
    floor_losses, floor_grads, floor_params = _run_reference(floor_runnable, hparams.steps)
    for step, loss in enumerate(floor_losses):
      max_logging.log(f"  floor step {step + 1}/{hparams.steps}: {loss:.9f}")
    floor = (floor_losses, floor_grads, jax.device_get(floor_params))

  notes = parity_notes(
      mpmd_runnable.maxtext_config,
      mpmd_runnable.plan.topology.data_parallel_size,
      platform=mpmd_runnable.plan.layer_mesh.devices.flat[0].platform,
  )
  return assemble_result(
      reference=(spmd_losses, spmd_grads, spmd_params),
      mpmd_run=(mpmd_losses, mpmd_grads, mpmd_params),
      floor=floor,
      floor_microbatches=floor_microbatches,
      notes=notes,
  )


def parity_notes(maxtext_config: Any, data_parallel_size: int, platform: str = "") -> list[str]:
  """Returns the caveats a configuration's report should print (`platform`: the devices', e.g. ``tpu``)."""
  notes = []
  compute_dtype = jnp.dtype(maxtext_config.dtype)
  exact = ("highest", "float32")
  missing = []
  if maxtext_config.matmul_precision not in exact:
    # deepseek3-tiny in float32 with all 16 experts routed at PP=4 x FSDP=2 on
    # TPU7x: step-2 loss 4.3e-4 from the reference against a floor of 1.6e-5,
    # and attention-norm gradients 1.9%, at the default precision; 1.4e-4
    # (floor 2.7e-4) and 0.42% (bound 1.14%) at highest.
    missing.append("--maxtext_overrides=matmul_precision=highest")
  if maxtext_config.decoder_block == DecoderBlockType.SIMPLE_MLP and jax.config.jax_default_matmul_precision not in exact:
    # matmul_precision reaches only the simple model's output head: its
    # layers' jnp.dot take JAX's default precision (simple in float32 on
    # TPU7x: step-1 gradients 1.15e-3 from the reference with
    # matmul_precision=highest alone, 3.2e-7 with both).
    missing.append("JAX_DEFAULT_MATMUL_PRECISION=highest (the simple model's layers take JAX's default precision)")
  if platform == "tpu" and compute_dtype == jnp.float32 and missing:
    notes.append(
        "On TPU a float32 matmul at the default precision multiplies in bfloat16 passes, whose rounding the"
        " differently fused MPMD fragments and the single program do not share: for a like-for-like check add"
        f" {' and '.join(missing)}."
    )
  if data_parallel_size > 1 and compute_dtype != jnp.float32:
    # `models._bind_pregathered` / `_cast_sharded` reduce-scatter the gathered
    # weights' gradients in the compute dtype, for speed; the unsharded
    # reference accumulates float32 gradients. The floor run is unsharded too.
    notes.append(
        f"MPMD reduce-scatters gradients over the stage's devices in"
        f" {compute_dtype.name}, the unsharded reference accumulates them in"
        " float32, and the noise floor does not see that: for a like-for-like"
        " check run with --maxtext_overrides=dtype=float32."
    )
  num_experts = maxtext_config.num_experts
  top_k = maxtext_config.num_experts_per_tok
  if 1 < num_experts and top_k < num_experts:
    # deepseek3-tiny in float32 at PP=4 x FSDP=2 (or --expert=2) on a v5p-16:
    # step-1 loss 6.9e-6 from the reference (floor 3.7e-7), router gradients
    # 4.4% (floor 0.8%); with all 16 experts routed, 1.6e-6 (floor 3.1e-6)
    # and every gradient within 0.71%.
    notes.append(
        f"Each token goes to {top_k} of {num_experts} experts. Where two"
        " experts' scores nearly tie, the MPMD program's rounding (its"
        " fragments fuse differently from the single program) can pick the"
        " other one, which moves the loss and the router and expert gradients"
        " in a jump that the floor, an SPMD run, need not show. If only those"
        " fail, check the MoE machinery without the discontinuity by"
        " rerunning with"
        f" --maxtext_overrides=num_experts_per_tok={num_experts}."
    )
  return notes


def assemble_result(
    *,
    reference: tuple[Sequence[float], Any, Any],
    mpmd_run: tuple[Sequence[float], Any, Any],
    floor: tuple[Sequence[float], Any, Any] | None = None,
    floor_microbatches: int = 0,
    notes: Sequence[str] = (),
) -> ParityResult:
  """Compares the runs' ``(losses, step-1 gradients, final params)``.

  Args:
    reference: The SPMD reference run.
    mpmd_run: The MPMD run.
    floor: The noise-floor run (the reference with another microbatch split),
      or None.
    floor_microbatches: The floor run's microbatch count.
    notes: Caveats printed with the report.

  Returns:
    The :class:`ParityResult`.
  """
  ref_losses, ref_grads, ref_params = reference

  def compare(run) -> Comparison:
    losses, grads, params = run
    return Comparison(
        losses=tuple(losses),
        reference_losses=tuple(ref_losses),
        # Gradient norms are far below 1: floor the denominator at 1e-5.
        grads=_compare_params(grads, ref_grads, norm_floor=1e-5),
        params=_compare_params(params, ref_params),
    )

  return ParityResult(
      mpmd=compare(mpmd_run),
      floor=compare(floor) if floor is not None else None,
      floor_microbatches=floor_microbatches if floor is not None else 0,
      num_params=len(jax.tree_util.tree_leaves(ref_params)),
      notes=tuple(notes),
  )


def main(argv: Sequence[str], flag_values: flags.FlagValues = flags.FLAGS) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected command line arguments: {argv[1:]}")
  # Registers the Pathways proxy backend when JAX_PLATFORMS contains "proxy";
  # it must run before the backend is first used.
  pathwaysutils.initialize()

  fv = flag_values
  hparams = hparams_lib.from_flags(fv)
  result = check_parity(
      hparams,
      floor_microbatches=noise_floor_microbatches(hparams.num_microbatches, fv.noise_floor_microbatches),
  )

  max_logging.log("=" * 72)
  max_logging.log("MPMD vs SPMD numerical parity")
  max_logging.log("=" * 72)
  tolerances = (fv.loss_tolerance, fv.param_tolerance, fv.grad_tolerance, fv.noise_floor_factor)
  max_logging.log(result.report(*tolerances))
  max_logging.log("=" * 72)

  if result.passed(*tolerances):
    max_logging.log("PARITY PASSED")
    return
  raise SystemExit("PARITY FAILED: see the differences reported above.")


if __name__ == "__main__":
  define_flags()
  jax.config.config_with_absl()
  app.run(main)
