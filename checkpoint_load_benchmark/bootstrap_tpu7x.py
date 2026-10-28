"""Pod bootstrap and coordination script for `bodaborg-tpu7x-nap` (`tpu7x-standard-4t`, `2x2x1`) benchmark pods.

Responsibilities:
  1. Environment Setup (`setup_environment`):
     - Installs missing Python dependencies from `gs://mesa-maxtext/mesakh_tpu7x_bench/code/site_packages_overlay.tar.gz`
       without shadowing the container's native TPU-matched `jax`, `jaxlib`, `flax`, or `orbax` packages.
     - Stages MaxText worktrees on each pod:
         * `/tmp/maxtext`: `origin/docs/checkpoint-load-benchmark` (includes Lance Wang's 3 PRs: `#5706`, `#5828`, `#5705`)
         * `/tmp/maxtext_origin_main`: `origin/main` (`f596f5369`, before Lance Wang's 3 PRs) for Experiment 1A comparison.
         * `/tmp/maxtext_hbmfix`: `/tmp/maxtext` with the HBM-peak fix for composite keys (`load_dynamic.py` and
           `tensor_handling.py` from `gs://mesa-maxtext/mesakh_tpu7x_bench/code/`), used only by runs labeled hbmfix.
       A run selects its tree by running `<tree>/checkpoint_load_benchmark/benchmark_ckpt_load_time.py`; the harness
       copies itself to the same path on the slice-1 daemon, so both slices use the same tree.
     - Patches `load_dynamic.py` to use single-GET streaming (`1` HTTP GET per `.safetensors` shard with 32 MB GCS upload chunks)
       so `>200B` models (`94` to `241` shards, `470 GB` to `961 GB`) stream from HuggingFace Hub to GCS without HTTP 429 rate limits.
  2. Multi-Slice Role Dispatch (`main`):
     - `ROLE=daemon` (Slice 1 pod `mesakh-tpu7x-<id>1`): Starts `benchmark_ckpt_load_time.py --remote-slice-daemon --daemon-port 55555`.
     - `ROLE=runner` (Slice 0 pod `mesakh-tpu7x-<id>0`): Waits for the Slice 1 daemon via headless Kubernetes Service DNS
       (`WAIT_FOR_DAEMON_HOST=mesakh-w1-<id>`), exports `REMOTE_SLICE1_ADDR=<ip>:55555`, runs `RUNNER_SCRIPT`, and sends a clean
       shutdown signal to Slice 1 upon completion.
"""

import json
import os
import shutil
import site
import socket
import subprocess
import sys
import tarfile
import time
from etils import epath

GCS_CODE_DIR = "gs://mesa-maxtext/mesakh_tpu7x_bench/code"


def download_from_gcs(gcs_uri: str, local_path: str) -> None:
  os.makedirs(os.path.dirname(local_path), exist_ok=True)
  data = epath.Path(gcs_uri).read_bytes()
  with open(local_path, "wb") as f:
    f.write(data)
  print(f"[Bootstrap] Downloaded {gcs_uri} -> {local_path} ({len(data)/1e6:.2f} MB)", flush=True)


def _patch_single_get_streaming(ld_file: str) -> None:
  """Makes `build_gcs_cache_worker` stream each shard from HF Hub with 1 GET (no HTTP 429s on 100+ shards)."""
  with open(ld_file, "r", encoding="utf-8") as f:
    ld_txt = f.read()
  old_block = """      with fs.open(fpath, "rb") as remote_f:
        blob.chunk_size = 1024 * 1024 * 32  # 32MB chunks
        blob.upload_from_file(remote_f, client=storage_client)"""
  new_block = """      import requests, shutil
      _parts = fpath.strip("/").split("/")
      _repo_id, _fname = "/".join(_parts[:2]), "/".join(_parts[2:])
      _url = huggingface_hub.hf_hub_url(repo_id=_repo_id, filename=_fname)
      _is_gated = _repo_id.startswith(("meta-llama/", "google/"))
      _hdrs = {"Authorization": f"Bearer {hf_access_token}"} if (hf_access_token and (_is_gated or attempt % 2 == 1)) else {}
      with requests.get(_url, headers=_hdrs, stream=True, timeout=(30, 300)) as _resp:
        _resp.raise_for_status()
        _resp.raw.decode_content = True
        with blob.open("wb", chunk_size=32 * 1024 * 1024) as _gcs_out:
          shutil.copyfileobj(_resp.raw, _gcs_out, length=16 * 1024 * 1024)"""
  ld_txt = ld_txt.replace(old_block, new_block)
  ld_txt = ld_txt.replace("max_workers=32", "max_workers=16")
  with open(ld_file, "w", encoding="utf-8") as f:
    f.write(ld_txt)
  print(f"[Bootstrap] Patched single-GET HF->GCS streaming in {ld_file}", flush=True)


def setup_environment() -> None:
  t0 = time.time()
  os.makedirs("/tmp/site_packages_raw", exist_ok=True)
  os.makedirs("/tmp/maxtext", exist_ok=True)

  download_from_gcs(f"{GCS_CODE_DIR}/site_packages_overlay.tar.gz", "/tmp/site_packages_overlay.tar.gz")
  with tarfile.open("/tmp/site_packages_overlay.tar.gz", "r:gz") as tf:
    tf.extractall("/tmp/site_packages_raw")

  # Only install overlay packages that do not already exist in the container's native site-packages
  # so we never shadow the container's matched jax/jaxlib/flax/orbax installation.
  site_dirs = site.getsitepackages()
  target_sp = site_dirs[0] if site_dirs else "/usr/local/lib/python3.12/site-packages"
  copied_pkgs = []
  for item in os.listdir("/tmp/site_packages_raw"):
    src = os.path.join("/tmp/site_packages_raw", item)
    dst = os.path.join(target_sp, item)
    if not os.path.exists(dst):
      if os.path.isdir(src):
        shutil.copytree(src, dst)
      else:
        shutil.copy2(src, dst)
      if not item.endswith(".dist-info"):
        copied_pkgs.append(item)
  print(f"[Bootstrap] Added missing packages to {target_sp}: {sorted(copied_pkgs)}", flush=True)

  download_from_gcs(f"{GCS_CODE_DIR}/maxtext_lance_prs.tar.gz", "/tmp/maxtext_lance_prs.tar.gz")
  with tarfile.open("/tmp/maxtext_lance_prs.tar.gz", "r:gz") as tf:
    tf.extractall("/tmp/maxtext")

  download_from_gcs(
      f"{GCS_CODE_DIR}/benchmark_ckpt_load_time.py",
      "/tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py",
  )

  # Also create /tmp/maxtext_origin_main with origin/main checkpoint_conversion code (Before Lance's 3 PRs)
  if os.path.exists("/tmp/maxtext_origin_main"):
    shutil.rmtree("/tmp/maxtext_origin_main")
  shutil.copytree("/tmp/maxtext", "/tmp/maxtext_origin_main")
  download_from_gcs(
      f"{GCS_CODE_DIR}/checkpoint_conversion_origin_main.tar.gz",
      "/tmp/checkpoint_conversion_origin_main.tar.gz",
  )
  with tarfile.open("/tmp/checkpoint_conversion_origin_main.tar.gz", "r:gz") as tf:
    tf.extractall("/tmp/maxtext_origin_main")
  shutil.rmtree("/tmp/site_packages_raw", ignore_errors=True)
  for _tar_f in (
      "/tmp/site_packages_overlay.tar.gz",
      "/tmp/maxtext_lance_prs.tar.gz",
      "/tmp/checkpoint_conversion_origin_main.tar.gz",
  ):
    try:
      os.remove(_tar_f)
    except OSError:
      pass

  for root_dir in ("/tmp/maxtext", "/tmp/maxtext_origin_main"):
    ld_file = os.path.join(root_dir, "src/maxtext/checkpoint_conversion/utils/load_dynamic.py")
    if os.path.exists(ld_file):
      _patch_single_get_streaming(ld_file)

  # Patch composite tuple (wi_0, wi_1) split-before-reshard and intermediate buffer deletion in /tmp/maxtext
  th_file = "/tmp/maxtext/src/maxtext/checkpoint_conversion/utils/tensor_handling.py"
  ld_main = "/tmp/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py"
  if os.path.exists(th_file) and os.path.exists(ld_main):
    with open(th_file, "r", encoding="utf-8") as f:
      th_txt = f.read()
    old_reshard = """  pieces = [
      jax.device_put(jax.lax.slice_in_dim(array, start, min(start + step, size), axis=axis), sharding)
      for start in range(0, size, step)
  ]
  return _concatenate_fn(axis, sharding)(*pieces)"""
    new_reshard = """  pieces = [
      jax.device_put(jax.lax.slice_in_dim(array, start, min(start + step, size), axis=axis), sharding)
      for start in range(0, size, step)
  ]
  jax.block_until_ready(pieces)
  if isinstance(array, jax.Array) and not array.is_deleted():
    array.delete()
  res = _concatenate_fn(axis, sharding)(*pieces)
  jax.block_until_ready(res)
  for _p in pieces:
    if isinstance(_p, jax.Array) and not _p.is_deleted():
      _p.delete()
  return res"""
    th_txt = th_txt.replace(old_reshard, new_reshard)

    old_bstack = """  mid = len(tensors) // 2
  left = _binary_chunked_stack(tensors[:mid], axis=axis)
  right = _binary_chunked_stack(tensors[mid:], axis=axis)
  return np.concatenate([left, right], axis=axis)"""
    new_bstack = """  mid = len(tensors) // 2
  left = _binary_chunked_stack(tensors[:mid], axis=axis)
  right = _binary_chunked_stack(tensors[mid:], axis=axis)
  res = np.concatenate([left, right], axis=axis)
  jax.block_until_ready(res)
  for _side in (left, right):
    if isinstance(_side, jax.Array) and not _side.is_deleted():
      _side.delete()
  return res"""
    th_txt = th_txt.replace(old_bstack, new_bstack)
    th_txt = th_txt.replace(
        "  if target_sharding is not None:\n    stacked_array = _reshard(stacked_array, target_sharding)",
        "  if target_sharding is not None and not isinstance(mt_key, tuple):\n    stacked_array = _reshard(stacked_array, target_sharding)",
    )
    with open(th_file, "w", encoding="utf-8") as f:
      f.write(th_txt)

    with open(ld_main, "r", encoding="utf-8") as f:
      ld_m_txt = f.read()
    old_comp = """    load_fn = get_hf_loading_function(
        hf_source,
        tensor_getter,
        hook_fn,
        target_leaf,
        maxtext_config,
        mt_keys[0],
    )

    # Execute transformation and assign to flat_restored
    t_layer = time.time()
    loaded = load_fn()
    if isinstance(mt_key, tuple):
      for i, (name, leaf) in enumerate(zip(check_names, leaves)):
        part = loaded[..., i]
        flat_restored[name] = jax.device_put(part, leaf.sharding) if hasattr(leaf, "sharding") else part
    else:
      flat_restored[check_names[0]] = loaded"""
    new_comp = """    load_fn = get_hf_loading_function(
        hf_source,
        tensor_getter,
        hook_fn,
        target_leaf,
        maxtext_config,
        mt_key if isinstance(mt_key, tuple) else mt_keys[0],
    )

    # Execute transformation and assign to flat_restored
    t_layer = time.time()
    loaded = load_fn()
    if isinstance(mt_key, tuple):
      parts = [jax.lax.slice_in_dim(loaded, i, i + 1, axis=-1)[..., 0] for i in range(len(leaves))]
      jax.block_until_ready(parts)
      if isinstance(loaded, jax.Array) and not loaded.is_deleted():
        loaded.delete()
      del loaded
      for i, (name, leaf) in enumerate(zip(check_names, leaves)):
        part = parts[i]
        parts[i] = None
        flat_restored[name] = tensor_handling._reshard(part, leaf.sharding) if hasattr(leaf, "sharding") else part
    else:
      flat_restored[check_names[0]] = loaded"""
    ld_m_txt = ld_m_txt.replace(old_comp, new_comp)
    with open(ld_main, "w", encoding="utf-8") as f:
      f.write(ld_m_txt)
    print("[Bootstrap] Patched composite-leaf split-before-reshard HBM optimization in /tmp/maxtext", flush=True)

  try:
    from google.cloud import storage  # noqa: F401
  except ImportError:
    print("[Bootstrap] Installing google-cloud-storage...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "google-cloud-storage"],
        check=True,
    )

  stage_hbmfix_tree()
  print(f"[Bootstrap] Environment ready in {time.time() - t0:.2f}s", flush=True)


def stage_hbmfix_tree() -> None:
  """Stages `/tmp/maxtext_hbmfix`: `/tmp/maxtext` with the HBM-peak fix for composite keys (e.g. qwen3.5 `wi_0/wi_1`).

  The fixed `load_dynamic.py` and `tensor_handling.py` are read from GCS_CODE_DIR. The tree is separate so runs
  that compare against `origin/main` keep using `/tmp/maxtext` (Lance's PRs plus the string patches above), and
  runs that use the fix select it by running the harness from this tree.
  """
  utils_rel = "src/maxtext/checkpoint_conversion/utils"
  try:
    fixed = {
        name: epath.Path(f"{GCS_CODE_DIR}/{name}").read_bytes() for name in ("load_dynamic.py", "tensor_handling.py")
    }
  except Exception as ex:  # pylint: disable=broad-exception-caught
    print(f"[Bootstrap] Skipping /tmp/maxtext_hbmfix ({ex})", flush=True)
    return
  if os.path.exists("/tmp/maxtext_hbmfix"):
    shutil.rmtree("/tmp/maxtext_hbmfix")
  shutil.copytree("/tmp/maxtext", "/tmp/maxtext_hbmfix")
  for name, data in fixed.items():
    with open(os.path.join("/tmp/maxtext_hbmfix", utils_rel, name), "wb") as f:
      f.write(data)
  _patch_single_get_streaming(os.path.join("/tmp/maxtext_hbmfix", utils_rel, "load_dynamic.py"))
  print("[Bootstrap] Staged /tmp/maxtext_hbmfix with the fixed load_dynamic.py and tensor_handling.py", flush=True)


def wait_for_daemon(host: str, port: int = 55555, timeout_s: int = 7200) -> str:
  print(f"[Bootstrap] Waiting for remote slice daemon at {host}:{port} ...", flush=True)
  t0 = time.time()
  while time.time() - t0 < timeout_s:
    try:
      ip = socket.gethostbyname(host)
      with socket.create_connection((ip, port), timeout=5):
        print(f"[Bootstrap] Connected to remote slice daemon at {host} ({ip}:{port})!", flush=True)
        return f"{ip}:{port}"
    except Exception:
      time.sleep(2)
  raise TimeoutError(f"Timed out waiting for remote slice daemon {host}:{port}")


def shutdown_daemons(addrs: list[str]) -> None:
  for addr in addrs:
    try:
      host, port_s = addr.split(":", 1)
      with socket.create_connection((host, int(port_s)), timeout=10) as sock:
        sock.sendall((json.dumps({"shutdown": True}) + "\n").encode("utf-8"))
      print(f"[Bootstrap] Sent shutdown signal to remote slice daemon at {addr}", flush=True)
    except Exception as ex:  # pylint: disable=broad-exception-caught
      print(f"[Bootstrap] Warning: could not send shutdown to {addr}: {ex}", flush=True)


def main() -> None:
  setup_environment()
  os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

  role = os.environ.get("ROLE", "runner")
  if role == "daemon":
    port = int(os.environ.get("DAEMON_PORT", "55555"))
    cmd = [
        sys.executable,
        "-u",
        "/tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py",
        "--remote-slice-daemon",
        "--daemon-port",
        str(port),
    ]
    os.execvpe(sys.executable, cmd, os.environ)

  addrs: list[str] = []
  daemon_host = os.environ.get("WAIT_FOR_DAEMON_HOST", "").strip()
  if daemon_host:
    daemon_port = int(os.environ.get("DAEMON_PORT", "55555"))
    for dh in [h.strip() for h in daemon_host.split(",") if h.strip()]:
      addrs.append(wait_for_daemon(dh, daemon_port))
    os.environ["REMOTE_SLICE1_ADDR"] = ",".join(addrs)

  runner_script = os.environ.get("RUNNER_SCRIPT", "").strip()
  try:
    if runner_script:
      subprocess.run(["bash", "-c", runner_script], check=True, env=os.environ)
    else:
      print("[Bootstrap] No RUNNER_SCRIPT provided; exiting.", flush=True)
  finally:
    if addrs:
      shutdown_daemons(addrs)


if __name__ == "__main__":
  main()
