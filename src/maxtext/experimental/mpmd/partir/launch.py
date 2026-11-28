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

r"""Prints (or runs) the command that runs a partir entry point on GKE through Pathways.

The command is ``gcluster job submit --pathways`` (Cluster Toolkit, MaxText's
launcher for GKE; ``--launcher=gcluster``, the default), after ``gcluster job
config set`` of the project, cluster and location, or ``xpk workload
create-pathways`` (``--launcher=xpk``; MaxText has deprecated XPK): one JobSet
whose head pod runs the Pathways resource manager, the IFRT proxy and the entry
point, while the TPU slices run Pathways workers. ``--device_type`` names each
slice; gcluster's ``--compute-type`` and ``--topology`` are each derived from it
(:func:`gcluster_slice`) unless ``--gcluster_compute_type`` or
``--gcluster_topology`` gives it. ``--module`` picks the entry point:
``train`` (the default), ``parity`` (MPMD vs SPMD) or ``train_compile`` (compile
on the slices' Pathways backend and log the memory report;
``--compile_topology=local`` is added). Arguments after ``--`` go to the entry
point verbatim, e.g. ``-- --loss_tolerance=1e-3`` for ``parity``.

The image (``--docker_image``) must hold this MaxText tree and the Shardy MPMD
frontend that matches its jaxlib (``sdy_mpmd_compat.py``), and the Pathways
images (``--pathways_tag``) must match its jax. gcluster v1.108.0 and v1.109.0
run a successful Pathways job twice and then mark it Failed
(:data:`GCLUSTER_RESTART_WARNING`, which the launcher logs).

The model and pipeline flags are not redeclared here: :func:`hparams.define_flags`
registers them, and :func:`hparams.to_job_args` serialises them back out for the
entry point, so the launcher can never pass a flag the entry point does not
accept. The launcher checks what would otherwise fail only once the slices are
up: the workload name's length, how stages and slices divide, and whether each
microbatch splits over a stage's batch-sharding devices.

Example (PP=4 x FSDP=2 on a v5p-16, i.e. 8 chips on 2 hosts)::

    python3 -m maxtext.experimental.mpmd.partir.launch \
      --cluster=<cluster> --project=<project> --zone=<zone> \
      --docker_image=<image> --pathways_gcs_location=gs://<bucket>/pathways \
      --device_type=v5p-16 --workload=mpmd-pp4 \
      --pp=4 --fsdp=2 --num_microbatches=8 --per_device_batch_size=8 \
      --model_name=deepseek3-tiny --num_layers=8      # prints the command
    ... --submit                                       # runs it
    ... --module=parity --submit -- --loss_tolerance=1e-3
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import re
import shlex
import subprocess

from absl import app
from absl import flags

from maxtext.utils import accelerator_to_spec_map
from maxtext.utils import max_logging

from maxtext.experimental.mpmd.partir import hparams as hparams_lib

# The newest Pathways images known to compile Shardy MPMD's IFRT IR from a jax
# 0.12.0.1 client: the 20261118, 20261120 and 20261124 jax_0.12.0.1 images
# reject it ("Failed to parse IFRT IR module string").
DEFAULT_PATHWAYS_TAG = "20261117-jax_0.12.0.1"
# Since cluster-toolkit #6528 (in v1.108.0 and v1.109.0), a Pathways batch job's workers have restartPolicy Never and
# backoffLimit 0. A Pathways worker exits when its client disconnects, so at the end of a successful run the worker Job
# fails before the head Job completes, and JobSet, which acts on a failed Job first, restarts the whole JobSet.
GCLUSTER_RESTART_WARNING = (
    "gcluster v1.108.0 and v1.109.0 give a Pathways job's workers restartPolicy Never, and a Pathways worker exits when"
    " its client disconnects: when the entry point ends, even successfully, the worker Job fails, the JobSet restarts,"
    " the entry point runs a second time, and the JobSet ends Failed. cluster-toolkit PR #6739 (merged into develop on"
    " 2026-11-25; not in v1.109.0) fixes it: use a gcluster built with it, or --launcher=xpk."
)
PATHWAYS_IMAGES = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways"
# xpk's Pathways JobSet labels every pod `<name>-pathways-head-0-0.<name>`, a
# Kubernetes label of at most 63 characters: 2 * len(name) + 19 <= 63.
MAX_PATHWAYS_NAME = 22
# A DNS-1035 label, which starts with a letter: xpk's workload names, and the
# Job names JobSet's admission webhook builds from the name (`<name>-<job>-<i>`).
_NAME = re.compile(r"[a-z]([a-z0-9-]*[a-z0-9])?")
# `gcluster job submit --compute-type` of each TPU generation (the GCE machine types MaxText's
# run_maxtext_via_pathways.md and run_maxtext_via_cluster_toolkit.md name), by --device_type prefix.
GCLUSTER_COMPUTE_TYPES = {
    "tpu7x": "tpu7x-standard-4t",
    "v6e": "ct6e-standard-4t",
    "v5p": "ct5p-hightpu-4t",
    "v5e": "ct5lp-hightpu-4t",
    "v5litepod": "ct5lp-hightpu-4t",
    "v4": "ct4p-hightpu-4t",
}
# Every compute type above is a host of 4 chips, in a chip box of 3 dimensions (TPU7x, v5p, v4) or 2 (v6e, v5e).
_GCLUSTER_CHIPS_PER_HOST = 4
_TOPOLOGY_DIMENSIONS = {"tpu7x": 3, "v5p": 3, "v4": 3, "v6e": 2, "v5e": 2, "v5litepod": 2}
# The flags every launcher needs.
_REQUIRED = ("cluster", "project", "zone", "docker_image", "pathways_gcs_location", "device_type", "workload")
# The launcher's own flags: none of them may be passed after `--`.
_OWN_FLAGS = (
    "module",
    "launcher",
    "cluster",
    "project",
    "zone",
    "docker_image",
    "pathways_gcs_location",
    "pathways_tag",
    "device_type",
    "num_slices",
    "workload",
    "gcluster_compute_type",
    "gcluster_topology",
    "extra_xla_flags",
    "xla_flags_file",
    "launcher_args",
    "enable_profile",
    "profile_steps",
    "profile_dir",
    "peak_tflops_per_device",
    "submit",
)


def define_flags(flag_values: flags.FlagValues = flags.FLAGS) -> None:
  """Registers the launcher's flags, the hyperparameters' included, on `flag_values`."""
  hparams_lib.define_flags(flag_values)
  fv = {"flag_values": flag_values}
  flags.DEFINE_enum(
      "module",
      "train",
      ["train", "parity", "train_compile"],
      "Entry point to run: maxtext.experimental.mpmd.partir.<module>.",
      **fv,
  )
  flags.DEFINE_enum(
      "launcher",
      "gcluster",
      ["gcluster", "xpk"],
      "gcluster: `gcluster job submit --pathways` (Cluster Toolkit), after `gcluster job config set` of the project,"
      " cluster and zone. xpk: `xpk workload create-pathways` (MaxText has deprecated XPK).",
      **fv,
  )
  flags.DEFINE_string("cluster", "", "GKE cluster (required).", **fv)
  flags.DEFINE_string("project", "", "GCP project of the cluster (required).", **fv)
  flags.DEFINE_string("zone", "", "Zone (or region) of the cluster (required).", **fv)
  flags.DEFINE_string(
      "docker_image",
      "",
      "Image of the head pod's user container (required): this MaxText tree, and the shardy-mpmd frontend that"
      " matches its jaxlib.",
      **fv,
  )
  flags.DEFINE_string(
      "pathways_gcs_location",
      "",
      "gs:// scratch location of the Pathways servers (required); it holds their persistent compilation cache.",
      **fv,
  )
  flags.DEFINE_string(
      "pathways_tag",
      DEFAULT_PATHWAYS_TAG,
      f"Tag of the Pathways server and proxy_server images ({PATHWAYS_IMAGES}/...); it must match the jax version in"
      " --docker_image.",
      **fv,
  )
  flags.DEFINE_string(
      "device_type",
      "",
      "TPU type of each slice (required), e.g. v5p-16, v6e-16, tpu7x-128 or tpu7x-4x4x4: xpk's --tpu-type, and what"
      " gcluster's --compute-type and --topology are derived from.",
      **fv,
  )
  flags.DEFINE_integer("num_slices", 1, "Number of TPU slices.", **fv)
  flags.DEFINE_string(
      "workload",
      "",
      f"Workload name (required): a lowercase letter, then lowercase letters, digits and '-', at most"
      f" {MAX_PATHWAYS_NAME} characters for a Pathways JobSet; also the trainer's --run_name.",
      **fv,
  )
  flags.DEFINE_string(
      "gcluster_compute_type",
      "",
      "--launcher=gcluster: the slices' --compute-type, e.g. ct5p-hightpu-4t; empty derives it from --device_type.",
      **fv,
  )
  flags.DEFINE_string(
      "gcluster_topology",
      "",
      "--launcher=gcluster: the slices' --topology, e.g. 2x2x2; empty derives it from --device_type.",
      **fv,
  )
  flags.DEFINE_multi_string(
      "extra_xla_flags",
      [],
      "key=value XLA flag for the Pathways proxy server (which compiles the TPU programs), resource manager and"
      " workers; repeatable, applied after --xla_flags_file.",
      **fv,
  )
  flags.DEFINE_string(
      "xla_flags_file",
      "",
      "File of --key=value XLA flags, one per line with # comments, for the Pathways proxy server, resource manager"
      " and workers.",
      **fv,
  )
  flags.DEFINE_multi_string(
      "launcher_args", [], "Argument appended verbatim to the xpk or gcluster command, e.g. --priority=low.", **fv
  )
  flags.DEFINE_boolean("enable_profile", False, "train: capture an XProf trace (needs --profile_dir).", **fv)
  flags.DEFINE_integer("profile_steps", 3, "train: extra steps run under the profiler (not timed).", **fv)
  flags.DEFINE_string("profile_dir", "", "train: gs:// directory for the trace (<profile_dir>/<workload>).", **fv)
  flags.DEFINE_float(
      "peak_tflops_per_device",
      0.0,
      "train: hardware peak TFLOP/s per JAX device (on TPU7x, half a chip's), for MFU.",
      **fv,
  )
  flags.DEFINE_boolean("submit", False, "Run the command(s) instead of printing them.", **fv)


def xla_server_args(flag_values: flags.FlagValues) -> str:
  """Returns the ``--key=value`` XLA flags for the Pathways servers, joined by single spaces."""
  xla_args: dict[str, str] = {}
  if flag_values.xla_flags_file:
    xla_args.update(hparams_lib.read_xla_flags_file(flag_values.xla_flags_file))
  for item in flag_values.extra_xla_flags:
    key, sep, value = item.lstrip("-").partition("=")
    if not sep or not key:
      raise app.UsageError(f"Expected --extra_xla_flags entry of the form key=value, got {item!r}.")
    xla_args[key] = value
  return " ".join(f"--{key}={value}" for key, value in xla_args.items())


def module_args(hparams: hparams_lib.Hparams, flag_values: flags.FlagValues, extra: Sequence[str] = ()) -> list[str]:
  """Returns the command line of ``maxtext.experimental.mpmd.partir.<module>`` for this run.

  Args:
    hparams: The configuration to run.
    flag_values: The launcher's parsed flags.
    extra: Arguments passed after ``--``: flags only the entry point defines
      (e.g. parity's ``--loss_tolerance``), as ``--flag=value``, appended
      verbatim.

  Raises:
    app.UsageError: If an extra argument is not a ``--flag``, or is a
      hyperparameter or launcher flag. Those go before ``--``, where they are
      validated; after it they would silently override the validated values.
  """
  module = flag_values.module
  args = dict(hparams_lib.to_job_args(hparams))
  if module == "train":
    args.update(
        enable_profile=flag_values.enable_profile,
        profile_steps=flag_values.profile_steps,
        profile_dir=flag_values.profile_dir,
        peak_tflops_per_device=flag_values.peak_tflops_per_device,
        run_name=flag_values.workload,
    )
  elif module == "train_compile":
    # The head pod's only backend is the Pathways proxy: no fake CPU devices.
    args["compile_topology"] = "local"
  reserved = set(hparams_lib.FORWARDED_FLAGS) | {"pp"} | set(_OWN_FLAGS)
  for item in extra:
    if not item.startswith("--"):
      raise app.UsageError(f"Arguments after -- go to {module} and must be --flags, got {item!r}.")
    name = item[2:].partition("=")[0]
    if name in reserved or name.removeprefix("no") in reserved:
      raise app.UsageError(f"{item!r} is a hyperparameter or launcher flag: pass it before --, where it is validated.")
  return hparams_lib.job_argv(args) + list(extra)


def gcluster_topology(device_type: str) -> tuple[str, int]:
  """Returns gcluster's ``--topology`` for a slice of `device_type`, and the number of chips on each of its hosts.

  The topology is the chip box: the name's own (``tpu7x-4x4x4``), or MaxText's for the type
  (``accelerator_to_spec_map``: ``v5p-16`` is 2x2x2, ``v6e-16`` 4x4, ``tpu7x-128`` 4x4x4).

  Args:
    device_type: A TPU type such as ``v5p-16`` or ``tpu7x-4x4x4``.

  Raises:
    ValueError: If the slice is not known: a generation not in :data:`GCLUSTER_COMPUTE_TYPES`, a type MaxText does not
      know (``v5p-12345``), or a topology with the wrong number of dimensions for its generation (``tpu7x-4x4``) or
      with a dimension that is not a positive integer (``v5p-2x2x0``).
  """
  generation, _, size = device_type.partition("-")
  if generation not in GCLUSTER_COMPUTE_TYPES:
    raise ValueError(f"{generation!r} is not one of the known TPU generations ({', '.join(GCLUSTER_COMPUTE_TYPES)})")
  if "x" in size:
    dimensions = size.split("x")
    expected = _TOPOLOGY_DIMENSIONS[generation]
    if len(dimensions) != expected:
      raise ValueError(f"a {generation} topology has {expected} dimensions ({'x'.join('ABC'[:expected])}), not {size!r}")
    for dimension in dimensions:
      if not (dimension.isdecimal() and int(dimension) > 0):
        raise ValueError(f"a topology's dimensions are positive integers, not {dimension!r} (in {size!r})")
    return size, min(math.prod(map(int, dimensions)), _GCLUSTER_CHIPS_PER_HOST)
  name = f"v5e-{size}" if generation == "v5litepod" else device_type
  spec = accelerator_to_spec_map.UserFacingNameToSystemCharacteristics.get(name)
  if spec is None or not spec.topology_name or not spec.chips_per_host_bounds:
    raise ValueError(f"MaxText knows no {device_type} slice (maxtext.utils.accelerator_to_spec_map)")
  return spec.topology_name.partition(":")[2], math.prod(spec.chips_per_host_bounds)


def gcluster_slice(device_type: str) -> tuple[str, str]:
  """Returns gcluster's ``(--compute-type, --topology)`` for a slice of `device_type`.

  The compute type is the generation's host of 4 chips (:data:`GCLUSTER_COMPUTE_TYPES`); the topology is
  :func:`gcluster_topology`'s.

  Args:
    device_type: A TPU type such as ``v5p-16`` or ``tpu7x-4x4x4``.

  Raises:
    ValueError: If the slice is not known (see :func:`gcluster_topology`), or its hosts do not have 4 chips
      (``v6e-1``, ``tpu7x-2``).
  """
  topology, chips_per_host = gcluster_topology(device_type)
  compute_type = GCLUSTER_COMPUTE_TYPES[device_type.partition("-")[0]]
  if chips_per_host != _GCLUSTER_CHIPS_PER_HOST:
    raise ValueError(
        f"a {topology} slice has {chips_per_host}-chip hosts, and {compute_type} is a host of"
        f" {_GCLUSTER_CHIPS_PER_HOST} chips"
    )
  return compute_type, topology


def gcluster_options(flag_values: flags.FlagValues) -> tuple[str, str]:
  """Returns ``(--compute-type, --topology)`` for gcluster: each the flag's, else derived from ``--device_type``.

  Only a value no flag gives is derived: with ``--gcluster_compute_type`` alone, just the topology, so that compute
  type is not checked against the slice's hosts.

  Raises:
    app.UsageError: If a value no flag gives cannot be derived. The error names only the flags that are needed: just
      ``--gcluster_compute_type`` when the slice is known but its hosts do not have 4 chips.
  """
  fv = flag_values
  compute_type, topology = fv.gcluster_compute_type, fv.gcluster_topology
  if compute_type and topology:
    return compute_type, topology
  try:
    derived_topology, _ = gcluster_topology(fv.device_type)
  except ValueError as error:
    # Neither value can be derived from a slice that is not known.
    missing = [f"--{name}" for name in ("gcluster_compute_type", "gcluster_topology") if not fv[name].value]
    raise app.UsageError(f"--device_type={fv.device_type}: {error}; pass {' and '.join(missing)}.") from error
  if not compute_type:
    try:
      compute_type, _ = gcluster_slice(fv.device_type)
    except ValueError as error:
      raise app.UsageError(f"--device_type={fv.device_type}: {error}; pass --gcluster_compute_type.") from error
  return compute_type, topology or derived_topology


def gcluster_config_commands(project: str, cluster: str, zone: str) -> list[list[str]]:
  """Returns the ``gcluster job config set`` commands that point ``gcluster job submit`` at the cluster."""
  return [
      ["gcluster", "job", "config", "set", "project", project],
      ["gcluster", "job", "config", "set", "cluster", cluster],
      ["gcluster", "job", "config", "set", "location", zone],
  ]


def check_workload_name(workload: str) -> None:
  """Raises unless `workload` is a valid Pathways JobSet name (see :data:`MAX_PATHWAYS_NAME` and ``_NAME``)."""
  if not _NAME.fullmatch(workload) or len(workload) > MAX_PATHWAYS_NAME:
    raise app.UsageError(
        f"--workload {workload!r}: a lowercase letter, then lowercase letters, digits and '-' (not last), at most"
        f" {MAX_PATHWAYS_NAME} characters for a Pathways JobSet."
    )


def check_slices(hparams: hparams_lib.Hparams, device_type: str, num_slices: int) -> None:
  """Fails fast when the slices cannot hold the stages, or a microbatch cannot be split.

  Args:
    hparams: The configuration to launch.
    device_type: TPU type of each slice, e.g. ``v5p-16``.
    num_slices: Number of slices.

  Raises:
    app.UsageError: If the device count, the slices or the microbatch do not
      divide.
  """
  devices = hparams_lib.devices_in_device_type(device_type) * num_slices
  if devices <= 0 or devices % hparams.num_stages:
    raise app.UsageError(
        f"{device_type} x {num_slices} ({devices} devices) does not split into {hparams.num_stages} stages."
    )
  if num_slices > 1 and hparams.num_stages % num_slices and num_slices % hparams.num_stages:
    # mesh_utils never lets a stage straddle a slice.
    raise app.UsageError(
        f"{num_slices} slices and {hparams.num_stages} stages must divide one another: each slice holds whole stages,"
        " or each stage whole slices."
    )
  try:
    hparams_lib.check_microbatch_split_on(hparams, devices)
  except ValueError as error:
    raise app.UsageError(str(error)) from error


def entry_point_command(hparams: hparams_lib.Hparams, flag_values: flags.FlagValues, extra: Sequence[str] = ()) -> str:
  """Returns the head pod's command: the entry point, as ``python3 -u -m``."""
  module = f"maxtext.experimental.mpmd.partir.{flag_values.module}"
  return shlex.join(["python3", "-u", "-m", module, *module_args(hparams, flag_values, extra)])


def launcher_commands(
    hparams: hparams_lib.Hparams, flag_values: flags.FlagValues, extra: Sequence[str] = ()
) -> list[list[str]]:
  """Returns the argvs that launch the run, in order (config, then submit, for gcluster; one for xpk)."""
  fv = flag_values
  command = entry_point_command(hparams, fv, extra)
  server_image = f"{PATHWAYS_IMAGES}/server:{fv.pathways_tag}"
  proxy_image = f"{PATHWAYS_IMAGES}/proxy_server:{fv.pathways_tag}"
  # The proxy server hosts the Pathways client, which compiles: a compiler flag
  # given only to the resource manager and workers does not apply.
  xla = xla_server_args(fv)
  if fv.launcher == "gcluster":
    compute_type, topology = gcluster_options(fv)
    argv = ["gcluster", "job", "submit", "--pathways"] + [
        f"--name={fv.workload}",
        f"--image={fv.docker_image}",
        f"--compute-type={compute_type}",
        f"--topology={topology}",
        f"--num-slices={fv.num_slices}",
        f"--pathways-gcs-location={fv.pathways_gcs_location}",
        f"--pathways-server-image={server_image}",
        f"--pathways-worker-image={server_image}",
        f"--pathways-proxy-server-image={proxy_image}",
    ]
    if xla:
      argv += [f"--pathways-proxy-args={xla}", f"--pathways-server-args={xla}", f"--pathways-worker-args={xla}"]
    return gcluster_config_commands(fv.project, fv.cluster, fv.zone) + [
        argv + list(fv.launcher_args) + [f"--command={command}"]
    ]
  argv = ["xpk", "workload", "create-pathways"] + [
      f"--cluster={fv.cluster}",
      f"--project={fv.project}",
      f"--zone={fv.zone}",
      f"--workload={fv.workload}",
      f"--tpu-type={fv.device_type}",
      f"--num-slices={fv.num_slices}",
      f"--docker-image={fv.docker_image}",
      f"--server-image={server_image}",
      f"--proxy-server-image={proxy_image}",
      f"--pathways-gcs-location={fv.pathways_gcs_location}",
  ]
  if xla:
    argv += [
        f"--custom-pathways-proxy-server-args={xla}",
        f"--custom-pathways-server-args={xla}",
        f"--custom-pathways-worker-args={xla}",
    ]
  return [argv + list(fv.launcher_args) + [f"--command={command}"]]


def main(argv: Sequence[str], flag_values: flags.FlagValues = flags.FLAGS) -> None:
  fv = flag_values
  missing = [f"--{name}" for name in _REQUIRED if not fv[name].value]
  if missing:
    raise app.UsageError(f"Missing required flags: {', '.join(missing)}.")
  if not fv.pathways_gcs_location.startswith("gs://"):
    raise app.UsageError(f"--pathways_gcs_location must be gs://..., got {fv.pathways_gcs_location!r}.")
  if fv.enable_profile and not fv.profile_dir.startswith("gs://"):
    raise app.UsageError("--enable_profile needs a gs:// --profile_dir: Pathways writes traces only to GCS.")
  # Validates the hyperparameters (layers divisible by stages, schedule name,
  # decoder block, ...) here, so that a misconfiguration fails in a second
  # rather than after the slices have been provisioned.
  try:
    hparams = hparams_lib.from_flags(fv)
  except (ValueError, FileNotFoundError) as error:
    raise app.UsageError(str(error)) from error
  check_workload_name(fv.workload)
  check_slices(hparams, fv.device_type, fv.num_slices)
  note = hparams_lib.ragged_buffer_note(hparams)
  if note:
    max_logging.warning(f"launch: {note}")
  if fv.launcher == "gcluster":
    max_logging.warning(f"launch: {GCLUSTER_RESTART_WARNING}")

  commands = launcher_commands(hparams, fv, argv[1:])
  if not fv.submit:
    for command in commands:
      print(shlex.join(command))
    return
  for command in commands:
    returncode = subprocess.run(command, check=False).returncode
    if returncode:
      raise SystemExit(returncode)


if __name__ == "__main__":
  define_flags()
  app.run(main)
