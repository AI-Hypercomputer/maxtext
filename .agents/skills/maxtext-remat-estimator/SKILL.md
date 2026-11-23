---
name: maxtext-remat-estimator
description: >-
  Finds rematerialization (remat) policies and per-device batch sizes that fit
  in TPU HBM for a MaxText training config, using the AOT-compile-based
  estimator (`maxtext.utils.estimator`) so the target hardware is not needed.
  Use when a MaxText training run runs out of HBM, when choosing
  `remat_policy` or custom per-tensor `remat`/`offload`/`device` settings, when
  looking for the largest `per_device_batch_size` that fits a model on a TPU
  topology, or when sizing a run for hardware you do not have yet. Don't use it
  to measure step time or MFU; the estimator only checks whether a
  configuration fits in memory.
---

# MaxText Remat & Batch Size Estimator

`maxtext.utils.estimator` searches custom remat policies and
`per_device_batch_size` values. It checks each candidate by compiling the
training step ahead of time (AOT) for `compile_topology`, so a CPU VM or a
single small TPU VM can size a run for a large pod.

Reference docs:
- [Tuning remat policy](../../../docs/guides/optimization/benchmark_and_performance.md#tuning-remat-policy),
  including estimator examples for Llama 3.1 405B and DeepSeek V3 671B
- [Ahead-of-time compilation](../../../docs/guides/monitoring_and_debugging/features_and_diagnostics.md#ahead-of-time-compilation-aot)
- Implementation: [`src/maxtext/utils/estimator.py`](../../../src/maxtext/utils/estimator.py),
  `is_oom` in [`train_compile.py`](../../../src/maxtext/trainers/pre_train/train_compile.py),
  and the custom remat keys in [`base.yml`](../../../src/maxtext/configs/base.yml)

---

## 1. How the search works

Each probe appends `per_device_batch_size=<pdb> remat_policy=custom <tensor>=remat|offload|device ...`
to your arguments and compiles `train_step` with `train_compile.is_oom`. If the
compile error contains `RESOURCE_EXHAUSTED` or `HBM`, the probe counts as out of
memory. Any other compile error stops the search.

The estimator varies the dense decoder checkpoints:
- `context`
- the Q/K/V projections: `query_proj`, `key_proj`, `value_proj`, and `kv_proj`, or `qkv_proj`
- the MLP input: `mlpwi`, or `mlpwi_0` and `mlpwi_1`
- `mlpwo` and `out_proj`

It ranks them by size from `max_target_length` and the layer widths. The largest
tensors are rematerialized first and kept on device last.

| Mode | When | Search | Result |
| :--- | :--- | :--- | :--- |
| **1. Batch size and policy** | `per_device_batch_size` not set | Finds the largest pdb (up to 32) that fits when every tensor is rematerialized. Then it moves tensors from `remat` to `offload`, smallest first, one at a time. When all of them are offloaded, it moves them to `device` the same way. After each move, it binary-searches the largest pdb that still fits. | Several `(pdb, policy)` pairs, from the largest batch with full remat to smaller batches with more tensors saved |
| **2. Policy only** | `per_device_batch_size=<pdb>` set | Starts with every tensor on `device`. It moves tensors from `device` to `offload`, largest first, then from `offload` to `remat`, until a policy fits. It raises `ValueError` if the batch runs out of memory even with full remat. | One `(pdb, policy)` pair: the least remat that fits |

Every probe is a full compile. A probe takes seconds for a small dense model and
minutes for a large MoE. Mode 2 needs roughly two probes per searched tensor at
most. Mode 1 can need several dozen probes; for example, `qwen3-0.6b` on
`v6e-4` took 53. Run Mode 1 under `tmux` or `nohup` for large models.

---

## 2. Before you run

1. **Environment:** use the same install as `train_compile`: a MaxText
   environment with `jax[tpu]`. A CPU VM or any TPU VM can compile for a TPU
   topology. On a TPU VM, set `JAX_PLATFORMS=cpu` so the probes don't claim
   the local chips. To run several estimator or `train_compile` processes at
   the same time, also set `ALLOW_MULTIPLE_LIBTPU_LOAD=1`.
2. **Copy every flag that affects memory from the real run:**
   - `model_name` and any model overrides, plus `max_target_length`
   - dtypes (`dtype`, `weight_dtype`, `grad_dtype`, `mu_dtype`) and `opt_type`
   - sharding: `ici_*` and `dcn_*` parallelism, `compile_topology`, and
     `compile_topology_num_slices`
   - attention and MoE kernel flags, and quantization
   - host offload: `optimizer_memory_host_offload` and
     `parameter_memory_host_offload`
   - `LIBTPU_INIT_ARGS`. XLA flags such as SparseCore collective offload
     change buffer assignment, which moves the OOM boundary.

   Use `dataset_type=synthetic`.
3. **Compile once by hand** with a setting that should fit. The estimator hides
   compiler output during probes and stops on any non-OOM error, so fix
   sharding and config errors first:
   ```bash
   python3 -m maxtext.trainers.pre_train.train_compile src/maxtext/configs/base.yml \
     <same args> remat_policy=full per_device_batch_size=1
   ```

---

## 3. Run the estimator

```bash
python3 -m maxtext.utils.estimator src/maxtext/configs/base.yml \
  model_name=<model> \
  compile_topology=<topology, e.g. v6e-256 or tpu7x-512> \
  compile_topology_num_slices=<num_slices> \
  max_target_length=<seq_len> \
  ici_fsdp_parallelism=-1 <other sharding flags> \
  <dtype, optimizer, kernel, and quantization flags from the real run> \
  dataset_type=synthetic log_config=False \
  write_estimator_result=True
```

- To run Mode 2 instead of Mode 1, add `per_device_batch_size=<pdb>`.
- **Pin a tensor** by passing it yourself, such as `context=device` or
  `mlpwo=offload`. A pinned tensor keeps that setting and leaves the search.
- `decoder_layer_input` stays on `device` unless you pass
  `decoder_layer_input=offload`. Offloading it moves each layer's saved input
  to host memory, which helps with long sequences and deep models.
- Mode 1 steps `per_device_batch_size` by `1/s`. Here `s` is the product of
  the `ici_*_parallelism` values over every mesh axis except `data`, `fsdp`,
  `fsdp_transpose`, `expert`, and `stage`. For example,
  `ici_tensor_parallelism=4` gives steps of `0.25`.
- You don't need to pass `remat_policy`. Every probe appends
  `remat_policy=custom`, and repeated flags take the last value.

---

## 4. Read the results

Each probe prints `Checking whether batch_size=<pdb> and policy={...} is OOM`
and then `Is OOM: True` or `Is OOM: False`. The search finishes with one or
more lines like this one:

```text
  - Found valid combo: pdb=2.0, policy={'mlpwo': 'offload', 'context': 'offload', ..., 'key_proj': 'device'}
```

With `write_estimator_result=True`, the estimator also writes
`remat_commands_from_estimator.txt` in the current directory. The file holds one
runnable `python -m maxtext.trainers.pre_train.train ...` command per result.
Each run overwrites it.

Before you launch a generated command:
- **Remove `compile_topology` and `compile_topology_num_slices`.** While
  `compile_topology` is set, MaxText sizes the mesh from the virtual topology
  and skips `jax.distributed` initialization.
- **Add the run-specific flags** the estimator didn't need. These include
  `run_name`, `base_output_directory`, the real dataset, `steps`, and
  checkpointing.

**Choose by measuring speed.** The estimator reports what fits, not what is
fastest. Remat recomputes activations in the backward pass. Offload spends
host-to-device bandwidth. A smaller batch can leave the TPU's matrix units idle.
Mode 1 reports every step of its walk, so many results share a pdb. Within each
group that shares a pdb, the last result rematerializes the fewest tensors.
Start by benchmarking the last result for each pdb on the target hardware, and
compare `Tokens/s/device` once step time is steady. MaxText
computes its TFLOP/s/device figure from the model, without counting
recomputation, so that figure is also comparable across remat policies.

---

## 5. Pitfalls

- **Tensors go to `offload` before `device`.** In both modes, every searched
  tensor goes to `offload` before any tensor goes to `device`, so many results
  rely heavily on offload. Host offload competes for PCIe bandwidth and can be
  slower than recomputing. If you suspect that, pin the tensors you want saved
  (for example `context=device`) and search again.
- **Some checkpoints are never searched.** The keys in the printed policy are
  the only tensors the estimator varies. Every other checkpoint keeps its value
  from the config or command line, which defaults to `remat`. That includes:
  - MoE: `moe_mlpwi_0`, `moe_mlpwi_1`, `moe_mlpwo`, `moe_x_sorted`,
    `moe_routing_maps`
  - MLA: `query_wa_proj`, `kv_wa_proj`, `mla_q`, `mla_kv`
  - `attention_out`
  - GDN: `gdn`, `gdn_states`, `gdn_cp_state`

  To include any of them, pin them on the command line. The complete list of
  names that `remat_policy=custom` honors is in the `remat_policy == "custom"`
  block of [`types.py`](../../../src/maxtext/configs/types.py).
- **Mode 1 stops at pdb 32.** If full remat still fits at pdb 32, the results
  start there.
- **`ValueError: Abnormal min_pdb=... or max_pdb=0.0 values` means nothing
  fits.** Mode 1 raises it right after printing `OOM at minimum batch size`,
  when even the smallest batch runs out of memory with full remat. Add more
  sharding, set `decoder_layer_input=offload`, turn on optimizer or parameter
  host offload, or apply the verification HBM escalation ladder in
  [`hbm_verification.md`](../../wiki/troubleshooting/hbm_verification.md), and
  then search again.
- **Errors other than OOM end the search with a traceback.** To see the full
  compiler log, rerun `train_compile` with the pdb and policy from the last
  probe.
- **Only `train_step` is compiled.** The probes don't check eval steps,
  checkpoint save and restore, or anything else outside `train_step`. Confirm
  the chosen configuration on hardware.

---

## 6. Verify on hardware

1. Remove the compile-only flags from the chosen command. Run it for about 20
   steps with the production data, checkpointing, and eval settings.
2. Confirm it doesn't run out of memory. Record step time and `Tokens/s/device`
   after warmup.
3. If two candidates are close, capture XProf traces (`profiler=xplane`) to see
   how much time goes to recomputation and host transfers.
