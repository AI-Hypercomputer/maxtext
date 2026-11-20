# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Renders the RL trainer benchmark JobSet from a dry-run render of a Tunix dist_rl e2e run.

Every process of a Tunix dist_rl run (`tunix/dist_rl/launch/common/launch.sh`)
gets the same config arguments (`override_config_file=...` and
`section.key=value`). The trainer JobSet is kept as is (Pathways resource
manager, proxy and worker containers and their args, env, images, node
selectors, tolerations, reservations, Kueue labels and annotations) except:

  * names: JobSet name, namespace, gang-id label; failurePolicy.maxRestarts=0
    (a failed bench should not rerun the compile);
  * the client command: the same inline env prefix (LIBTPU_INIT_ARGS and the
    other `cluster_config.roles.trainer.env` variables, ...) in front of
    run_trainer.py instead of the Tunix runtime main, with the e2e config
    arguments in CONFIG_ARGV; the Tunix fail-fast wrapper and the gRPC port
    are dropped;
  * config: `cluster_config.k8s.trainer_id` (the MaxText run name) is the bench
    name and `actor_model_config.kwargs.maxtext_output_dir` points at
    `--output-prefix/<name>`. run_trainer.py clears the checkpoint path itself
    (random init, the one intended difference);
  * optional knobs for A/B variants: --libtpu-extra / --libtpu-drop (written to
    every place the e2e render carries libtpu flags: the client's
    LIBTPU_INIT_ARGS, the pathways-proxy --xla_* args and the pw-node workers'
    LIBTPU_INIT_ARGS, and to the config keys they come from:
    `cluster_config.roles.trainer.env`,
    `cluster_config.pathways.proxy_extra_args` and
    `cluster_config.pathways.worker_env`, so CONFIG_ARGV describes the
    rendered JobSet), --extra-maxtext-flags (set
    `dist_training_config.maxtext_extra_config` keys), --replace-maxtext-flags
    (replace a key the resolved `maxtext_extra_config` already has),
    --set-config (set any config key), --compute-logps-chunk-size,
    --profiler-skip / --profiler-steps;
  * optional --trainer-slice (e.g. 4x8x8): the trainer slice and the FSDP size
    that fills it, with expert, context, dp and tp kept. It changes the
    Pathways resource manager's --instance_type, the pw-node
    completions/parallelism (hosts = chips / chips per host), the pw-node
    gke-tpu-slice-topology annotation, and `actor_model_config.mesh` (the
    assembler's pack size, fsdp x dp x expert rows, follows from it).

The final config arguments are resolved with Tunix's
`resolve_config.initialize`, so a bad override fails here rather than in the
pod; relative `override_config_file=` paths are read from --tunix-root.

Writes a file and nothing else; it never talks to a cluster. With
--print-aot-env it prints shell exports of the trainer's MaxText and XLA flags
and mesh for run_trainer_compile.sh instead of rendering.
"""

# pylint: disable=protected-access

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import shlex
import sys
from typing import Any

from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import trainer as trainer_node
import yaml

RUNTIME_MAIN = "python -m tunix.dist_rl.distributed.runtime.main"
TRAINER_MAIN = "tunix.dist_rl.launch.nodes.trainer.main"
# pod hostnames <name>-pw-node-0-<idx>-<5 chars> must stay <= 63 chars.
MAX_NAME_LEN = 30
TOPOLOGY_ANNOTATION = "cloud.google.com/gke-tpu-slice-topology"
NODE_SELECTOR_TOPOLOGY = "cloud.google.com/gke-tpu-topology"
MAXTEXT_EXTRA_CONFIG_KEY = "dist_training_config.maxtext_extra_config"
TRAINER_LIBTPU_KEY = "cluster_config.roles.trainer.env.LIBTPU_INIT_ARGS"
PROXY_EXTRA_ARGS_KEY = "cluster_config.pathways.proxy_extra_args"
WORKER_LIBTPU_KEY = "cluster_config.pathways.worker_env.LIBTPU_INIT_ARGS"


class _LiteralDumper(yaml.SafeDumper):
  """Dumps multi-line strings as literal blocks so rendered YAML is readable."""


def _str_representer(dumper, data):
  """Formats strings containing newlines with YAML literal block style."""
  style = "|" if "\n" in data else None
  return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_LiteralDumper.add_representer(str, _str_representer)


def find(items, name):
  """Returns the first mapping in `items` whose `name` field equals `name`."""
  for it in items:
    if it["name"] == name:
      return it
  raise KeyError(name)


def compose_libtpu(base: str, drop: str, extra: str) -> list[str]:
  """Returns `base` flags minus `drop` names, overridden/extended by `extra`."""

  def key(f):
    return f.split("=", 1)[0]

  drop_keys = {key(f) for f in drop.split()}
  extra_flags = extra.split()
  extra_keys = {key(f) for f in extra_flags}
  out = [f for f in base.split() if key(f) not in drop_keys and key(f) not in extra_keys]
  return out + extra_flags


def _proc_main(js):
  """Returns the `(pod_spec, main_container)` pair from a JobSet's `proc` job."""
  proc = find(js["spec"]["replicatedJobs"], "proc")["template"]["spec"]["template"]["spec"]
  return proc, find(proc["containers"], "main")


def load_documents(path: str) -> list[dict[str, Any]]:
  """Loads YAML JobSet documents from a launcher dry-run output file."""
  docs = []
  with open(path, encoding="utf-8") as f:
    text = f.read()
  for chunk in re.split(r"(?m)^---\s*$", text):
    try:
      d = yaml.safe_load(chunk)
    except yaml.YAMLError:
      continue
    if isinstance(d, dict) and d.get("kind") and d.get("spec"):
      docs.append(d)
  return docs


def process_main_of(js) -> str | None:
  """Extracts the `--process_main=` value of a JobSet's main container."""
  try:
    _, main_c = _proc_main(js)
  except (KeyError, TypeError):
    return None
  m = re.search(r"--process_main=(\S+)", " ".join(main_c.get("command", [])))
  return m.group(1).rstrip("\\") if m else None


def _main_aliases(name: str) -> set[str]:
  """Returns both `tunix.dist_rl.*` and `tunix.experimental.*` module names."""
  return {
      name,
      name.replace("tunix.dist_rl.", "tunix.experimental."),
      name.replace("tunix.experimental.", "tunix.dist_rl."),
  }


def select_jobset(docs: list[dict[str, Any]], trainer_main: str = TRAINER_MAIN) -> dict[str, Any]:
  """Selects the unique trainer JobSet (`--process_main=trainer_main`) from `docs`."""
  aliases = _main_aliases(trainer_main)
  hits = [d for d in docs if process_main_of(d) in aliases]
  if len(hits) != 1:
    names = [d["metadata"]["name"] for d in hits]
    raise ValueError(f"expected one JobSet with --process_main={trainer_main}, found {len(hits)}: {names}")
  return hits[0]


def split_command(cmd: str) -> tuple[str, list[str]]:
  """Splits a Tunix runtime command into `(env_prefix, flags_after_main)`."""
  i = cmd.index("(\n") + 2
  j = cmd.index(RUNTIME_MAIN)
  k = cmd.index("\n)", j)
  env_prefix = re.sub(r"^\s*trap\b[^\n]*\n", "", cmd[i:j])
  tokens = shlex.split(cmd[j + len(RUNTIME_MAIN) : k].replace("\\\n", " "))
  for n, t in enumerate(tokens):
    if t.startswith("--process_main="):
      return env_prefix, tokens[n + 1 :]
  raise ValueError("no --process_main in command")


def format_config_arg(key: str, value: str | int) -> str:
  """Returns the `key=value` override that sets `key` to `value`.

  The resolver parses each value as YAML, so a string is written as a one-line
  double-quoted YAML scalar and reads back unchanged (`""`, `"0"` or `"true"`
  stay strings instead of becoming None, 0 or True), as
  `tunix.cli.config.format_cli_arg` writes them.

  Args:
    key: Dotted config path.
    value: The value to set.
  """
  if isinstance(value, str):
    # `width=math.inf` keeps long strings on one line.
    quoted = yaml.safe_dump(value, default_style='"', width=math.inf)
    return f"{key}={quoted.rstrip()}"
  return f"{key}={value}"


def set_config(args: list[str], key: str, value: str | int) -> list[str]:
  """Removes every `key=` override from `args` and appends one setting `key` to `value`."""
  return [t for t in args if not t.startswith(f"{key}=")] + [format_config_arg(key, value)]


def set_raw_configs(args: list[str], kvs: list[str]) -> list[str]:
  """Replaces or appends each raw `key=value` override of `kvs`, its value read as YAML."""
  for kv in kvs:
    key, sep, _ = kv.partition("=")
    if not key or not sep:
      raise ValueError(f"expected key=value, got {kv!r}")
    args = [t for t in args if not t.startswith(f"{key}=")] + [kv]
  return args


def resolve(config_args: list[str], tunix_root: str) -> resolve_config.DistHyperParameters:
  """Resolves `config_args`, reading relative `override_config_file=` paths from `tunix_root`.

  The k8s launcher passes override files relative to the repo root, the
  container's working directory.

  Args:
    config_args: The run's config arguments.
    tunix_root: Root of the Tunix checkout that holds the override files.

  Returns:
    The validated configuration.
  """
  args = []
  for t in config_args:
    if t.startswith("override_config_file="):
      path = t.split("=", 1)[1]
      if not os.path.isabs(path):
        t = f"override_config_file={os.path.join(tunix_root, path)}"
    args.append(t)
  return resolve_config.initialize(args)


def libtpu_init_args(prefix: str) -> str:
  """Extracts the `LIBTPU_INIT_ARGS` string from an inline env prefix."""
  m = re.findall(r"LIBTPU_INIT_ARGS='([^']*)'", prefix)
  if len(m) > 1:
    raise ValueError(f"expected at most one LIBTPU_INIT_ARGS in env prefix, found {len(m)}")
  return m[0] if m else ""


def rewrite_env_prefix(prefix: str, libtpu: list[str]) -> str:
  """Rewrites `LIBTPU_INIT_ARGS` in `prefix`."""
  # LIBTPU_INIT_ARGS='...' (single-quoted in the e2e render).
  prefix, n = re.subn(
      r"LIBTPU_INIT_ARGS='[^']*'",
      lambda _: "LIBTPU_INIT_ARGS=" + shlex.quote(" ".join(libtpu)),
      prefix,
  )
  if n == 0 and libtpu:
    prefix = prefix.rstrip() + " LIBTPU_INIT_ARGS=" + shlex.quote(" ".join(libtpu)) + " "
  elif n > 1:
    raise ValueError(f"expected at most one LIBTPU_INIT_ARGS in the env prefix, found {n}")
  return prefix


def rewrite_pathways_libtpu(
    js: dict[str, Any],
    hp: resolve_config.DistHyperParameters,
    config_args: list[str],
    base_libtpu: str,
    libtpu: list[str],
) -> list[str]:
  """Writes `libtpu` into the pathways-proxy and pw-node worker flags of `js`.

  The proxy keeps its other args and takes the `--xla_*` flags of `libtpu`. If
  the render gives the workers a LIBTPU_INIT_ARGS (from
  `cluster_config.pathways.worker_env`, empty by default), it becomes `libtpu`
  plus the worker-only flags (those not in `base_libtpu`); otherwise the
  workers keep none. The returned config arguments set the keys these come
  from (`cluster_config.pathways.proxy_extra_args` and
  `.worker_env.LIBTPU_INIT_ARGS`) to the rewritten values.

  Args:
    js: The trainer JobSet, updated in place.
    hp: The resolved config of `config_args`.
    config_args: The run's config arguments.
    base_libtpu: The client's LIBTPU_INIT_ARGS in the e2e render.
    libtpu: The bench's libtpu flags.

  Returns:
    `config_args` with the rewritten Pathways keys set.
  """
  xla = [f for f in libtpu if f.startswith("--xla_")]
  proc, _ = _proc_main(js)
  proxy = find(proc["initContainers"], "pathways-proxy")
  head = [x for x in proxy["args"] if not x.startswith("--xla_") and not x.startswith("--sidecar_name")]
  tail = [x for x in proxy["args"] if x.startswith("--sidecar_name")]
  proxy["args"] = head + xla + tail
  pathways = hp.config_mapping("cluster_config")["pathways"]
  proxy_extra = [x for x in pathways["proxy_extra_args"].split() if not x.startswith("--xla_")]
  config_args = set_config(config_args, PROXY_EXTRA_ARGS_KEY, " ".join(proxy_extra + xla))
  pw_node = find(js["spec"]["replicatedJobs"], "pw-node")
  worker_pod = pw_node["template"]["spec"]["template"]["spec"]
  worker = find(worker_pod["containers"], "pathways-worker")
  if not any(e["name"] == "LIBTPU_INIT_ARGS" for e in worker["env"]):
    return config_args
  env = find(worker["env"], "LIBTPU_INIT_ARGS")
  base_keys = {x.split("=", 1)[0] for x in base_libtpu.split()}
  extras = [f for f in env["value"].split() if f.split("=", 1)[0] not in base_keys]
  env["value"] = " ".join(libtpu + extras)
  return set_config(config_args, WORKER_LIBTPU_KEY, env["value"])


def _mesh_context(hp: resolve_config.DistHyperParameters) -> int:
  """Returns the trainer's MaxText context-parallel size, as the trainer reads it."""
  return trainer_node._kwarg(hp.config_mapping("actor_model_config")["kwargs"], "mesh_context", 1)


def current_slice(js: dict[str, Any]) -> tuple[str, str]:
  """Returns `(accelerator, topology)` of the trainer JobSet."""
  proc, _ = _proc_main(js)
  rm = find(proc["initContainers"], "pathways-rm")
  it = [x.split("=", 1)[1] for x in rm["args"] if x.startswith("--instance_type=")]
  if len(it) != 1 or ":" not in it[0]:
    raise ValueError("expected one --instance_type=<accelerator>:<topology> in pathways-rm" f" args, found {it}")
  accel, topo = it[0].split(":", 1)
  pw = find(js["spec"]["replicatedJobs"], "pw-node")["template"]["spec"]
  ann = pw["template"].get("metadata", {}).get("annotations") or {}
  ns = pw["template"]["spec"].get("nodeSelector") or {}
  pw_topo = ann.get(TOPOLOGY_ANNOTATION) or ns.get(NODE_SELECTOR_TOPOLOGY)
  if pw_topo != topo:
    raise ValueError(f"pathways-rm --instance_type {it[0]} disagrees with pw-node topology:" f" {pw_topo!r}")
  return accel, topo


def compile_topology(accel: str, topo: str) -> str:
  """Maps a Pathways `(accelerator, topology)` pair to MaxText `compile_topology`."""
  chips = 1
  for d in topo.split("x"):
    chips *= int(d)
  if accel == "tpu7x":
    return f"tpu7x-{2 * chips}"
  if accel in ("tpuv6e", "tpuv6ea", "v6e"):
    return f"v6e-{chips}"
  if accel in ("tpuv5", "tpuv5p", "v5", "v5p"):
    return f"v5p-{2 * chips}"
  if accel in ("tpuv5e", "v5e"):
    return f"v5e-{chips}"
  return f"{accel}-{chips}"


def apply_trainer_slice(
    js: dict[str, Any],
    hp: resolve_config.DistHyperParameters,
    config_args: list[str],
    topo: str,
    devices_per_chip: int = 0,
) -> tuple[list[str], str]:
  """Re-targets the trainer JobSet and `actor_model_config.mesh` to a `topo` slice."""
  accel, ref_topo = current_slice(js)
  if devices_per_chip <= 0:
    devices_per_chip = 2 if accel == "tpu7x" else 1
  dims = [int(x) for x in topo.split("x")]
  chips = 1
  for d in dims:
    chips *= d
  pw = find(js["spec"]["replicatedJobs"], "pw-node")["template"]["spec"]
  worker = find(pw["template"]["spec"]["containers"], "pathways-worker")
  chips_per_host = int(worker["resources"]["limits"]["google.com/tpu"])
  if chips % chips_per_host:
    raise ValueError(f"{chips} chips not divisible by {chips_per_host} chips per host")
  hosts = chips // chips_per_host
  devices = chips * devices_per_chip
  axes = hp.mesh_axes("actor_model_config")
  tp, ep, dp, cp = axes["tp"], axes["expert"], axes["dp"], _mesh_context(hp)
  others = tp * ep * dp * cp
  if devices % others:
    raise ValueError(f"{devices} devices not divisible by tp*expert*dp*context={others}")
  fsdp = devices // others
  if pw["completions"] != pw["parallelism"]:
    raise ValueError("unexpected pw-node completions/parallelism:" f" {pw['completions']}/{pw['parallelism']}")
  shape, names = hp.parse_mesh_config("actor_model_config")
  if "fsdp" not in names:
    raise ValueError(f"actor_model_config.mesh.axis_names {names} has no fsdp axis to fill the slice with")

  proc, _ = _proc_main(js)
  rm = find(proc["initContainers"], "pathways-rm")
  it = [n for n, x in enumerate(rm["args"]) if x.startswith("--instance_type=")]
  rm["args"][it[0]] = f"--instance_type={accel}:{topo}"
  ann = pw["template"].setdefault("metadata", {}).setdefault("annotations", {})
  ns = pw["template"]["spec"].get("nodeSelector") or {}
  if TOPOLOGY_ANNOTATION in ann or NODE_SELECTOR_TOPOLOGY not in ns:
    ann[TOPOLOGY_ANNOTATION] = topo
  if NODE_SELECTOR_TOPOLOGY in ns:
    ns[NODE_SELECTOR_TOPOLOGY] = topo
  pw["completions"] = pw["parallelism"] = hosts
  # `mesh.shape` and `mesh.axis_names` are inherited as a pair, so both are set.
  new_shape = [fsdp if name == "fsdp" else size for size, name in zip(shape, names)]
  config_args = set_config(config_args, "actor_model_config.mesh.shape", f"({','.join(map(str, new_shape))})")
  config_args = set_config(
      config_args, "actor_model_config.mesh.axis_names", "(" + ",".join(f"'{name}'" for name in names) + ")"
  )
  config_args = set_config(config_args, "cluster_config.k8s.trainer.tpu_slice", f"{accel}:{topo}")
  summary = (
      f"trainer slice {accel}:{ref_topo} -> {accel}:{topo}: {chips} chips,"
      f" {hosts} hosts, {devices} devices; mesh fsdp={fsdp} tp={tp}"
      f" expert={ep} dp={dp} context={cp}; pack size {fsdp * dp * ep} rows per micro-batch"
  )
  return config_args, summary


def client_script(
    env_prefix: str,
    driver_src: str,
    config_args: list[str],
    *,
    steps: int,
    micro_steps: int,
    driver_env: str = "",
) -> str:
  """Builds the client container's bash script that embeds and runs the driver."""
  bench_env = f"CONFIG_ARGV={shlex.quote(shlex.join(config_args))} STEPS={steps} MICRO_STEPS={micro_steps}"
  for kv in shlex.split(driver_env):
    k, _, v = kv.partition("=")
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k):
      raise ValueError(f"bad --driver-env entry {kv!r}")
    bench_env += f" {k}={shlex.quote(v)}"
  if "BENCH_DRIVER_EOF" in driver_src:
    raise ValueError("driver source contains the heredoc terminator")
  return f"""# PID 1 wrapper: forwards SIGTERM to the benchmark so the grace period shuts it down cleanly.
echo "Program Start: $(date)"
cat > /tmp/run_trainer.py <<'BENCH_DRIVER_EOF'
{driver_src}BENCH_DRIVER_EOF
_sigterm() {{ kill -SIGTERM "$!" 2>/dev/null; }}
trap _sigterm SIGTERM
(
{env_prefix} {bench_env} exec python /tmp/run_trainer.py
) & PID=$!
while kill -0 $PID 2>/dev/null; do
  sleep 5
done
wait $PID
EXIT_CODE=$?
echo "Program End: $(date)"
echo "EXIT_CODE=$EXIT_CODE"
exit $EXIT_CODE
"""


def _is_moe_model(model_name: str, hp: resolve_config.DistHyperParameters) -> bool:
  """Returns whether `model_name` or the resolved config configures an MoE model."""
  models_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "..",
      "..",
      "..",
      "configs",
      "models",
  )
  yml = os.path.join(models_dir, f"{model_name}.yml")
  if os.path.exists(yml):
    with open(yml, encoding="utf-8") as f:
      data = yaml.safe_load(f) or {}
    return int(data.get("num_experts") or 1) > 1
  return hp.mesh_axes("actor_model_config")["expert"] > 1 or bool(
      hp.config_mapping("vllm_config")["return_routed_experts"]
  )


def aot_env(
    libtpu: list[str],
    hp: resolve_config.DistHyperParameters,
    js: dict[str, Any] | None = None,
) -> str:
  """Formats shell exports of MaxText, XLA, and mesh flags for AOT compile."""
  model_name = str(hp.config_mapping("actor_model_config")["kwargs"]["maxtext_model_name"])
  rollout = hp.config_mapping("rollout_config")
  rl_train = hp.config_mapping("rl_training_config")
  prompt_len = int(rollout["max_prompt_length"])
  resp_len = int(rollout["total_generation_steps"])
  budget = rl_train["max_seq_tokens_per_device"]
  seq_tpu = int(budget) if budget else 0
  target_len = max(prompt_len + resp_len, seq_tpu)
  seq_token_per_tpu = seq_tpu if seq_tpu > 0 else target_len
  segments = rl_train["max_segments_per_packed_row"]
  # 16: the AOT check's segment cap when the config sets none, as before.
  max_segments = 16 if segments is None else int(segments)
  mini_batch = int(rl_train["mini_batch_size"])
  num_gen = int(hp.config_mapping("agentic_grpo_config")["num_generations"])
  micro_batch = int(rl_train["train_micro_batch_size"])
  global_batch = max(mini_batch * num_gen, micro_batch)
  router_replay = "true" if _is_moe_model(model_name, hp) else "false"
  # The MaxText arguments the trainer appends for `maxtext_extra_config`.
  maxtext_extra = " ".join(trainer_node._maxtext_extra_flags(hp))
  lines = [
      f"export MAXTEXT_EXTRA_FLAGS={shlex.quote(maxtext_extra)}",
      ("export" f" LIBTPU_INIT_ARGS={shlex.quote(' '.join(f for f in libtpu if f.startswith('--xla_')))}"),
      f"export MODEL_NAME=${{MODEL_NAME:-{shlex.quote(model_name)}}}",
      f"export MAX_PREFILL_PREDICT_LENGTH=${{MAX_PREFILL_PREDICT_LENGTH:-{prompt_len}}}",
      f"export MAX_TARGET_LENGTH=${{MAX_TARGET_LENGTH:-{target_len}}}",
      f"export MAX_SEQ_TOKEN_PER_TPU=${{MAX_SEQ_TOKEN_PER_TPU:-{seq_token_per_tpu}}}",
      f"export MAX_SEGMENTS_PER_PACKED_ROW=${{MAX_SEGMENTS_PER_PACKED_ROW:-{max_segments}}}",
      f"export GLOBAL_BATCH=${{GLOBAL_BATCH:-{global_batch}}}",
      f"export TRAIN_MICRO_BATCH_SIZE=${{TRAIN_MICRO_BATCH_SIZE:-{micro_batch}}}",
      f"export ROUTER_REPLAY=${{ROUTER_REPLAY:-{router_replay}}}",
  ]
  if js is not None:
    topo = shlex.quote(compile_topology(*current_slice(js)))
    lines.append(f"export COMPILE_TOPOLOGY=${{COMPILE_TOPOLOGY:-{topo}}}")
  axes = hp.mesh_axes("actor_model_config")
  mesh = {"fsdp": axes["fsdp"], "tp": axes["tp"], "expert": axes["expert"], "context": _mesh_context(hp)}
  for k, val in mesh.items():
    var = f"MESH_{k.upper()}"
    lines.append(f"export {var}=${{{var}:-{val}}}")
  chunk = int(rl_train["compute_logps_chunk_size"])
  if chunk > 0:
    lines.append(f"export LOGPS_CHUNK_SIZE=${{LOGPS_CHUNK_SIZE:-{chunk}}}")
  return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
  """Returns the command-line parser for `render_trainer_jobset.py`."""
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
  p.add_argument(
      "--launch-render",
      default="",
      help="dry-run output of the e2e launcher (multi-document YAML)",
  )
  p.add_argument(
      "--train-yaml",
      default="",
      help="the e2e trainer JobSet alone (instead of --launch-render)",
  )
  p.add_argument(
      "--trainer-main",
      default=TRAINER_MAIN,
      help="--process_main of the trainer JobSet",
  )
  p.add_argument(
      "--tunix-root",
      default=str(resolve_config.REPO_ROOT),
      help="Tunix checkout that relative override_config_file= paths are read from",
  )
  p.add_argument(
      "--driver",
      default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_trainer.py"),
  )
  p.add_argument("--out", default="")
  p.add_argument("--name", default="")
  p.add_argument("--namespace", default="")
  p.add_argument(
      "--output-prefix",
      default="",
      help="the bench writes to <output-prefix>/<name>",
  )
  p.add_argument("--steps", type=int, default=3)
  p.add_argument("--micro-steps", type=int, default=4)
  p.add_argument("--libtpu-drop", default="")
  p.add_argument("--libtpu-extra", default="")
  p.add_argument(
      "--extra-maxtext-flags",
      default="",
      help='"k=v ...": set dist_training_config.maxtext_extra_config keys',
  )
  p.add_argument(
      "--replace-maxtext-flags",
      default="",
      help='"k=v ...": replace keys the resolved maxtext_extra_config already has',
  )
  p.add_argument(
      "--set-config",
      default="",
      help='"section.key=value ...": set config keys',
  )
  p.add_argument(
      "--image",
      default="",
      help="client (main container) image; empty keeps the e2e render's",
  )
  # Driver knobs (run_trainer.py)
  p.add_argument("--driver-env", default="")
  p.add_argument(
      "--trainer-slice",
      default="",
      help="e.g. 4x8x8: FSDP fills it, expert/context/tp kept",
  )
  p.add_argument(
      "--devices-per-chip",
      type=int,
      default=0,
      help="JAX devices per chip (0 auto-detects: 2 for tpu7x, 1 otherwise)",
  )
  p.add_argument(
      "--compute-logps-chunk-size",
      default="",
      help="rl_training_config.compute_logps_chunk_size; empty keeps",
  )
  p.add_argument(
      "--profiler-skip",
      default="",
      help="rl_training_config.profiler_options.skip_first_n_steps (counts fwd_bwd calls)",
  )
  p.add_argument(
      "--profiler-steps",
      default="",
      help="rl_training_config.profiler_options.profiler_steps (fwd_bwd calls in the window)",
  )
  p.add_argument(
      "--print-aot-env",
      action="store_true",
      help="print exports for run_trainer_compile.sh instead of rendering",
  )
  return p


def main(argv=None):
  """CLI entrypoint that renders the benchmark JobSet or prints AOT exports."""
  a = build_parser().parse_args(argv)
  if a.launch_render:
    if a.train_yaml:
      sys.exit("pass either --launch-render or --train-yaml")
    train = select_jobset(load_documents(a.launch_render), a.trainer_main)
  elif a.train_yaml:
    with open(a.train_yaml, encoding="utf-8") as f:
      train = yaml.safe_load(f)
  else:
    sys.exit("pass --launch-render or --train-yaml")
  if not a.print_aot_env:
    missing = [k for k in ("name", "namespace", "output_prefix") if not getattr(a, k)]
    if missing:
      sys.exit("missing required:" f" {', '.join('--' + k.replace('_', '-') for k in missing)}")
    if len(a.name) > MAX_NAME_LEN or not re.fullmatch(r"[a-z0-9-]+", a.name):
      sys.exit(f"NAME must be lowercase alphanumeric/dashes, <= {MAX_NAME_LEN}" f" chars: {a.name!r}")
    if not a.out:
      a.out = f"rendered/{a.name}.yaml"

  js = copy.deepcopy(train)
  _, main_c = _proc_main(js)
  env_prefix, config_args = split_command(main_c["command"][-1])
  base_libtpu = libtpu_init_args(env_prefix)
  libtpu = compose_libtpu(base_libtpu, a.libtpu_drop, a.libtpu_extra)
  env_prefix = rewrite_env_prefix(env_prefix, libtpu)
  if a.libtpu_drop or a.libtpu_extra:
    config_args = set_config(config_args, TRAINER_LIBTPU_KEY, " ".join(libtpu))

  if not a.print_aot_env:
    config_args = set_config(config_args, "cluster_config.k8s.trainer_id", a.name)
    config_args = set_config(config_args, "actor_model_config.kwargs.maxtext_output_dir", f"{a.output_prefix}/{a.name}")
  if a.compute_logps_chunk_size:
    config_args = set_config(config_args, "rl_training_config.compute_logps_chunk_size", int(a.compute_logps_chunk_size))
  config_args = set_raw_configs(config_args, [f"{MAXTEXT_EXTRA_CONFIG_KEY}.{kv}" for kv in a.extra_maxtext_flags.split()])
  hp = resolve(config_args, a.tunix_root)
  extra = hp.config_mapping("dist_training_config")["maxtext_extra_config"]
  for kv in a.replace_maxtext_flags.split():
    if kv.split("=", 1)[0] not in extra:
      raise ValueError(f"--replace-maxtext-flags {kv!r}: key not in the resolved {MAXTEXT_EXTRA_CONFIG_KEY}")
    config_args = set_raw_configs(config_args, [f"{MAXTEXT_EXTRA_CONFIG_KEY}.{kv}"])
  # The AOT exports read no Pathways container or key.
  if (a.libtpu_drop or a.libtpu_extra) and not a.print_aot_env:
    config_args = rewrite_pathways_libtpu(js, hp, config_args, base_libtpu, libtpu)
  if a.trainer_slice and a.trainer_slice != current_slice(js)[1]:
    config_args, summary = apply_trainer_slice(js, hp, config_args, a.trainer_slice, a.devices_per_chip)
    print(summary, file=sys.stderr)
  config_args = set_raw_configs(config_args, shlex.split(a.set_config))
  if a.profiler_steps:
    config_args = set_config(config_args, "rl_training_config.profiler_options.profiler_steps", int(a.profiler_steps))
  if a.profiler_skip:
    config_args = set_config(config_args, "rl_training_config.profiler_options.skip_first_n_steps", int(a.profiler_skip))
  # Validates the final arguments here rather than in the pod.
  hp = resolve(config_args, a.tunix_root)

  if a.print_aot_env:
    print(aot_env(libtpu, hp, js))
    return 0

  md = js["metadata"]
  md["name"], md["namespace"] = a.name, a.namespace
  md.setdefault("labels", {})["scheduling.x-k8s.io/gang-id"] = a.name
  js["spec"].setdefault("failurePolicy", {})["maxRestarts"] = 0

  with open(a.driver, encoding="utf-8") as f:
    driver_src = f.read()
  main_c["command"] = [
      "bash",
      "-c",
      client_script(
          env_prefix,
          driver_src,
          config_args,
          steps=a.steps,
          micro_steps=a.micro_steps,
          driver_env=a.driver_env,
      ),
  ]
  main_c.pop("ports", None)
  if a.image:
    main_c["image"] = a.image

  if os.path.dirname(a.out):
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
  with open(a.out, "w", encoding="utf-8") as f:
    yaml.dump(js, f, Dumper=_LiteralDumper, sort_keys=False, width=120)
  print(a.out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
