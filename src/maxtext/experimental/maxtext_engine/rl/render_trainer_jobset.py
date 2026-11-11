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

"""Renders the RL trainer benchmark JobSet from a DRY_RUN render of a Tunix GRPO e2e run.

The trainer JobSet is kept as is (Pathways resource manager, proxy and worker
containers and their args, env, images, node selectors, tolerations,
reservations, Kueue labels and annotations) except:

  * names: JobSet name, namespace, gang-id label; failurePolicy.maxRestarts=0
    (a failed bench should not rerun the compile);
  * the client command: the same inline env prefix (MAXTEXT_EXTRA_FLAGS,
    LIBTPU_INIT_ARGS, ...) in front of run_trainer.py instead of the
    Tunix runtime main, with the e2e trainer's flags (TRAINER_ARGV) and the e2e
    orchestrator's flags (ORCH_ARGV); the Tunix fail-fast wrapper and the gRPC
    port are dropped;
  * trainer flags: --maxtext_ckpt_path is cleared (random init, the one
    intended difference), --maxtext_output_directory points at
    `--output-prefix/<name>` and --worker_id is the bench name;
  * optional knobs for A/B variants: --libtpu-extra / --libtpu-drop (written to
    every place the e2e render carries libtpu flags: the client's
    LIBTPU_INIT_ARGS, the pathways-proxy --xla_* args and the pw-node workers'
    LIBTPU_INIT_ARGS), --extra-maxtext-flags (appended to MAXTEXT_EXTRA_FLAGS),
    --replace-maxtext-flags (replace an existing MAXTEXT_EXTRA_FLAGS key in
    place), --set-trainer-flags / --set-orch-flags (replace an existing trainer
    or orchestrator flag in place), --compute-logps-chunk-size,
    --profiler-skip / --profiler-steps;
  * optional --trainer-slice (e.g. 4x8x8): the trainer slice and the FSDP size
    that fills it, with expert, context and tp kept. It changes the Pathways
    resource manager's --instance_type, the pw-node completions/parallelism
    (hosts = chips / chips per host), the pw-node gke-tpu-slice-topology
    annotation, the trainer's --mesh_fsdp and the orchestrator's
    --trainer_fsdp (which sets the assembler's pack size, fsdp x expert rows).

Writes a file and nothing else; it never talks to a cluster. With
--print-aot-env it prints shell exports of the trainer's MaxText and XLA flags
and mesh for run_trainer_compile.sh instead of rendering.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import shlex
import sys
from typing import Any

import yaml

RUNTIME_MAIN = "python -m tunix.dist_rl.distributed.runtime.main"
TRAINER_MAIN = "tunix.dist_rl.examples.common.run_trainer_node.main"
ORCH_MAIN = "tunix.dist_rl.examples.deepswe_dist.run_deepswe_dist.main"
# pod hostnames <name>-pw-node-0-<idx>-<5 chars> must stay <= 63 chars.
MAX_NAME_LEN = 30
TOPOLOGY_ANNOTATION = "cloud.google.com/gke-tpu-slice-topology"
NODE_SELECTOR_TOPOLOGY = "cloud.google.com/gke-tpu-topology"


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
  """Loads YAML JobSet documents from a `DRY_RUN` output file."""
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


def select_jobsets(
    docs: list[dict[str, Any]],
    trainer_main: str = TRAINER_MAIN,
    orch_main: str = ORCH_MAIN,
):
  """Selects the unique `(trainer, orchestrator)` JobSet pair from `docs`."""
  out = []
  for want in (trainer_main, orch_main):
    aliases = _main_aliases(want)
    hits = [d for d in docs if process_main_of(d) in aliases]
    if len(hits) != 1:
      names = [d.get("metadata", {}).get("name") for d in hits]
      raise ValueError(f"expected one JobSet with --process_main={want}, found" f" {len(hits)}: {names}")
    out.append(hits[0])
  return out[0], out[1]


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


def set_flag(tokens: list[str], name: str, value: str) -> list[str]:
  """Removes every `--name=` entry from `tokens` and appends `--name=value`."""
  out = [t for t in tokens if not t.startswith(f"--{name}=")]
  return out + [f"--{name}={value}"]


def set_in_place(tokens: list[str], kvs: list[str], what: str = "trainer") -> list[str]:
  """Replaces existing `--k=` flags in place while preserving argv order."""
  tokens = list(tokens)
  for kv in kvs:
    k, _, v = kv.partition("=")
    hits = [i for i, t in enumerate(tokens) if t.startswith(f"--{k}=")]
    if len(hits) != 1:
      raise ValueError(f"--set-{what}-flags {kv!r}: --{k}= found {len(hits)} times in the" f" {what} argv")
    tokens[hits[0]] = f"--{k}={v}"
  return tokens


def upsert_flag(tokens: list[str], name: str, value: str) -> list[str]:
  """Replaces `--name=` in place if present once, or appends it if absent."""
  hits = [i for i, t in enumerate(tokens) if t.startswith(f"--{name}=")]
  if len(hits) > 1:
    raise ValueError(f"--{name}= found {len(hits)} times")
  if not hits:
    return list(tokens) + [f"--{name}={value}"]
  return set_in_place(tokens, [f"{name}={value}"])


def maxtext_extra_flags(prefix: str) -> str:
  """Extracts the `MAXTEXT_EXTRA_FLAGS` string from an inline env prefix."""
  m = re.findall(r'MAXTEXT_EXTRA_FLAGS="([^"]*)"', prefix)
  if len(m) > 1:
    raise ValueError("expected at most one MAXTEXT_EXTRA_FLAGS in env prefix, found" f" {len(m)}")
  return m[0] if m else ""


def libtpu_init_args(prefix: str) -> str:
  """Extracts the `LIBTPU_INIT_ARGS` string from an inline env prefix."""
  m = re.findall(r"LIBTPU_INIT_ARGS='([^']*)'", prefix)
  if len(m) > 1:
    raise ValueError(f"expected at most one LIBTPU_INIT_ARGS in env prefix, found {len(m)}")
  return m[0] if m else ""


def rewrite_env_prefix(
    prefix: str,
    libtpu: list[str],
    extra_maxtext: str,
    replace_maxtext: str = "",
) -> str:
  """Rewrites `LIBTPU_INIT_ARGS` and `MAXTEXT_EXTRA_FLAGS` in `prefix`."""
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
  for kv in replace_maxtext.split():
    # Replaces an existing key=value of MAXTEXT_EXTRA_FLAGS in place,
    # so the rendered flag set carries one value per key.
    # Fails if the key is not there exactly once.
    key = kv.split("=", 1)[0]
    m = re.search(r'MAXTEXT_EXTRA_FLAGS="([^"]*)"', prefix)
    if m is None:
      raise ValueError("no MAXTEXT_EXTRA_FLAGS in the env prefix")
    toks = m.group(1).split(" ")
    hits = [i for i, t in enumerate(toks) if t.split("=", 1)[0] == key]
    if len(hits) != 1:
      raise ValueError(f"--replace-maxtext-flags {kv!r}: key found {len(hits)} times in" " MAXTEXT_EXTRA_FLAGS")
    toks[hits[0]] = kv
    prefix = prefix[: m.start(1)] + " ".join(toks) + prefix[m.end(1) :]
  if extra_maxtext:
    prefix, n = re.subn(
        r'MAXTEXT_EXTRA_FLAGS="([^"]*)"',
        lambda m: f'MAXTEXT_EXTRA_FLAGS="{m.group(1)} {extra_maxtext}"',
        prefix,
    )
    if n == 0:
      prefix = f'{prefix.rstrip()} MAXTEXT_EXTRA_FLAGS="{extra_maxtext}" '
    elif n > 1:
      raise ValueError("expected at most one MAXTEXT_EXTRA_FLAGS in the env prefix," f" found {n}")
  return prefix


def _flag_value(tokens: list[str], *names: str, default: str | None = None) -> str:
  """Returns the value of a unique `--name=value` flag in `tokens`."""
  prefixes = tuple(f"--{n}=" for n in names)
  vals = list(dict.fromkeys(t.split("=", 1)[1] for t in tokens if t.startswith(prefixes)))
  if not vals and default is not None:
    return default
  if len(vals) != 1:
    raise ValueError(f"expected one --{names[0]}= flag, found {vals}")
  return vals[0]


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
    trainer_flags: list[str],
    orch_flags: list[str],
    topo: str,
    devices_per_chip: int = 0,
):
  """Re-targets the trainer JobSet and FSDP flags to a `topo` slice."""
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
  tp = int(_flag_value(trainer_flags, "mesh_tp"))
  ep = int(_flag_value(trainer_flags, "mesh_expert"))
  cp = int(_flag_value(trainer_flags, "mesh_context", "maxtext_mesh_context", default="1"))
  if devices % (tp * ep * cp):
    raise ValueError(f"{devices} devices not divisible by tp*expert*context={tp * ep * cp}")
  fsdp = devices // (tp * ep * cp)
  if any(t.startswith("--trainer_expert=") for t in orch_flags):
    if _flag_value(orch_flags, "trainer_expert") != str(ep):
      raise ValueError("orchestrator --trainer_expert differs from the trainer's" " --mesh_expert")
  if pw["completions"] != pw["parallelism"]:
    raise ValueError("unexpected pw-node completions/parallelism:" f" {pw['completions']}/{pw['parallelism']}")

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
  trainer_flags = set_flag(trainer_flags, "mesh_fsdp", str(fsdp))
  if any(t.startswith("--num_data_shards=") for t in orch_flags):
    orch_flags = set_flag(orch_flags, "num_data_shards", str(fsdp * ep))
  else:
    orch_flags = set_flag(orch_flags, "trainer_fsdp", str(fsdp))
  summary = (
      f"trainer slice {accel}:{ref_topo} -> {accel}:{topo}: {chips} chips,"
      f" {hosts} hosts, {devices} devices; mesh fsdp={fsdp} tp={tp}"
      f" expert={ep} context={cp}; pack size {fsdp * ep} rows per micro-batch"
  )
  return trainer_flags, orch_flags, summary


def client_script(
    env_prefix: str,
    driver_src: str,
    trainer_argv: list[str],
    orch_argv: list[str],
    *,
    steps: int,
    micro_steps: int,
    driver_env: str = "",
) -> str:
  """Builds the client container's bash script that embeds and runs the driver."""
  bench_env = (
      f"TRAINER_ARGV={shlex.quote(shlex.join(trainer_argv))} "
      f"ORCH_ARGV={shlex.quote(shlex.join(orch_argv))} STEPS={steps}"
      f" MICRO_STEPS={micro_steps}"
  )
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


def _is_moe_model(model_name: str, trainer_flags: list[str], orch_flags: list[str]) -> bool:
  """Returns whether `model_name` or the launch flags configure an MoE model."""
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
  ep = int(_flag_value(trainer_flags, "mesh_expert", default="1"))
  return ep > 1 or "--return_routed_experts" in orch_flags


def aot_env(
    env_prefix: str,
    libtpu: list[str],
    trainer_flags: list[str],
    orch_flags: tuple[str, ...] | list[str] = (),
    js: dict[str, Any] | None = None,
) -> str:
  """Formats shell exports of MaxText, XLA, and mesh flags for AOT compile."""
  model_name = _flag_value(trainer_flags, "maxtext_model_name")
  prompt_len = int(_flag_value(trainer_flags, "max_prompt_length", default="8192"))
  resp_len = int(_flag_value(trainer_flags, "max_response_length", default="57344"))
  seq_tpu = int(
      _flag_value(
          list(trainer_flags) + list(orch_flags),
          "max_seq_token_per_tpu",
          default="0",
      )
  )
  target_len = max(prompt_len + resp_len, seq_tpu)
  seq_token_per_tpu = seq_tpu if seq_tpu > 0 else target_len
  max_segments = _flag_value(
      list(orch_flags) + list(trainer_flags),
      "max_segments_per_packed_row",
      default="16",
  )
  mini_batch = int(
      _flag_value(
          list(orch_flags) + list(trainer_flags),
          "mini_batch_size",
          "batch_size",
          default="64",
      )
  )
  num_gen = int(
      _flag_value(
          list(orch_flags) + list(trainer_flags),
          "num_generations",
          default="16",
      )
  )
  micro_batch = int(
      _flag_value(
          list(orch_flags) + list(trainer_flags),
          "train_micro_batch_size",
          default="64",
      )
  )
  global_batch = max(mini_batch * num_gen, micro_batch)
  router_replay = "true" if _is_moe_model(model_name, trainer_flags, list(orch_flags)) else "false"
  lines = [
      ("export" f" MAXTEXT_EXTRA_FLAGS={shlex.quote(maxtext_extra_flags(env_prefix))}"),
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
  for k in ("fsdp", "tp", "expert", "context"):
    names = ("mesh_context", "maxtext_mesh_context") if k == "context" else (f"mesh_{k}",)
    default = "1" if k == "context" else None
    var = f"MESH_{k.upper()}"
    val = _flag_value(trainer_flags, *names, default=default)
    lines.append(f"export {var}=${{{var}:-{val}}}")
  chunk = [t.split("=", 1)[1] for t in trainer_flags if t.startswith("--compute_logps_chunk_size=")]
  if len(chunk) == 1:
    lines.append(f"export LOGPS_CHUNK_SIZE=${{LOGPS_CHUNK_SIZE:-{chunk[0]}}}")
  return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
  """Returns the command-line parser for `render_trainer_jobset.py`."""
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
  p.add_argument(
      "--launch-render",
      default="",
      help="DRY_RUN output of the e2e launcher (multi-document YAML)",
  )
  p.add_argument(
      "--train-yaml",
      default="",
      help="the e2e trainer JobSet alone (instead of --launch-render)",
  )
  p.add_argument(
      "--orch-yaml",
      default="",
      help="the e2e orchestrator JobSet alone (instead of --launch-render)",
  )
  p.add_argument(
      "--trainer-main",
      default=TRAINER_MAIN,
      help="--process_main of the trainer JobSet",
  )
  p.add_argument(
      "--orch-main",
      default=ORCH_MAIN,
      help="--process_main of the orchestrator JobSet",
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
  p.add_argument("--extra-maxtext-flags", default="")
  p.add_argument(
      "--replace-maxtext-flags",
      default="",
      help='"k=v ...": replace existing keys in place',
  )
  p.add_argument(
      "--set-trainer-flags",
      default="",
      help='"k=v ...": replace existing trainer flags in place',
  )
  p.add_argument(
      "--set-orch-flags",
      default="",
      help='"k=v ...": replace existing orchestrator flags in place',
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
      help="trainer --compute_logps_chunk_size; empty keeps",
  )
  p.add_argument(
      "--profiler-skip",
      default="",
      help="trainer --skip_first_n_profiler_steps (counts fwd_bwd calls)",
  )
  p.add_argument(
      "--profiler-steps",
      default="",
      help="trainer --profiler_steps (fwd_bwd calls in the window)",
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
    if a.train_yaml or a.orch_yaml:
      sys.exit("pass either --launch-render or --train-yaml/--orch-yaml")
    train, orch = select_jobsets(load_documents(a.launch_render), a.trainer_main, a.orch_main)
  elif a.train_yaml and a.orch_yaml:
    with open(a.train_yaml, encoding="utf-8") as f:
      train = yaml.safe_load(f)
    with open(a.orch_yaml, encoding="utf-8") as f:
      orch = yaml.safe_load(f)
  else:
    sys.exit("pass --launch-render, or both --train-yaml and --orch-yaml")
  if not a.print_aot_env:
    missing = [k for k in ("name", "namespace", "output_prefix") if not getattr(a, k)]
    if missing:
      sys.exit("missing required:" f" {', '.join('--' + k.replace('_', '-') for k in missing)}")
    if len(a.name) > MAX_NAME_LEN or not re.fullmatch(r"[a-z0-9-]+", a.name):
      sys.exit(f"NAME must be lowercase alphanumeric/dashes, <= {MAX_NAME_LEN}" f" chars: {a.name!r}")
    if not a.out:
      a.out = f"rendered/{a.name}.yaml"

  js = copy.deepcopy(train)
  _, orch_main = _proc_main(orch)
  _, orch_flags = split_command(orch_main["command"][-1])

  proc, main_c = _proc_main(js)
  env_prefix, trainer_flags = split_command(main_c["command"][-1])
  base_libtpu = libtpu_init_args(env_prefix)
  libtpu = compose_libtpu(base_libtpu, a.libtpu_drop, a.libtpu_extra)
  env_prefix = rewrite_env_prefix(env_prefix, libtpu, a.extra_maxtext_flags, a.replace_maxtext_flags)

  trainer_flags = set_flag(trainer_flags, "maxtext_ckpt_path", "")
  trainer_flags = set_flag(trainer_flags, "maxtext_output_directory", f"{a.output_prefix}/{a.name}")
  trainer_flags = set_flag(trainer_flags, "worker_id", a.name)
  if a.compute_logps_chunk_size:
    trainer_flags = set_in_place(
        trainer_flags,
        [f"compute_logps_chunk_size={int(a.compute_logps_chunk_size)}"],
    )
  trainer_flags = [t for t in trainer_flags if not t.startswith(("--discovery_addrs=", "--port="))]
  if a.trainer_slice and a.trainer_slice != current_slice(js)[1]:
    trainer_flags, orch_flags, summary = apply_trainer_slice(
        js, trainer_flags, orch_flags, a.trainer_slice, a.devices_per_chip
    )
    print(summary, file=sys.stderr)
  trainer_flags = set_in_place(trainer_flags, a.set_trainer_flags.split(), "trainer")
  orch_flags = set_in_place(orch_flags, a.set_orch_flags.split(), "orch")
  if a.profiler_steps:
    trainer_flags = upsert_flag(trainer_flags, "profiler_steps", str(int(a.profiler_steps)))
  if a.profiler_skip:
    trainer_flags = upsert_flag(trainer_flags, "skip_first_n_profiler_steps", str(int(a.profiler_skip)))

  if a.print_aot_env:
    print(aot_env(env_prefix, libtpu, trainer_flags, orch_flags, js))
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
          trainer_flags,
          orch_flags,
          steps=a.steps,
          micro_steps=a.micro_steps,
          driver_env=a.driver_env,
      ),
  ]
  main_c.pop("ports", None)
  if a.image:
    main_c["image"] = a.image

  if a.libtpu_drop or a.libtpu_extra:
    proxy = find(proc["initContainers"], "pathways-proxy")
    orig = proxy["args"]
    head = [x for x in orig if not x.startswith("--xla_") and not x.startswith("--sidecar_name")]
    tail = [x for x in orig if x.startswith("--sidecar_name")]
    proxy["args"] = head + [f for f in libtpu if f.startswith("--xla_")] + tail
    worker_pod = find(js["spec"]["replicatedJobs"], "pw-node")["template"]["spec"]["template"]["spec"]
    env = find(
        find(worker_pod["containers"], "pathways-worker")["env"],
        "LIBTPU_INIT_ARGS",
    )
    base_keys = {x.split("=", 1)[0] for x in base_libtpu.split()}
    extras = [f for f in env["value"].split() if f.split("=", 1)[0] not in base_keys]
    env["value"] = " ".join(libtpu + extras)

  if os.path.dirname(a.out):
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
  with open(a.out, "w", encoding="utf-8") as f:
    yaml.dump(js, f, Dumper=_LiteralDumper, sort_keys=False, width=120)
  print(a.out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
