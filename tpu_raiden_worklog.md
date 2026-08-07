# TPU-Raiden Work Log

## Project: MaxText Parameter & Optimizer State Offloading Fix & Layer-Wise Interleaved Design

### Activity Log
- **2026-08-05**: Initialized project repository `maxtext_optoff` at `/usr/local/google/home/mohitkhatwani/maxtext_optoff`.
- **2026-08-05**: Checked out `darisoy-qwen3-next-muon-scan-1476823e9` and created development branch `mohit/opt-off-trial`.
- **2026-08-05**: Analyzed current `optimizer_memory_host_offload` flaw in `maxtext/trainers/pre_train/train.py` and `maxtext_utils.py`.
- **2026-08-05**: Researched Dave Lacey's host offload reference implementation in `//depot/google3/experimental/users/davelacey/host_offload_training_example/`.
- **2026-08-05**: Completed Phase 0-4 under `planning_swarm` protocol. Implemented double-buffered layer-wise parameter and optimizer state offloading for scanned models.
- **2026-08-05**: Verified implementation via unit tests (`maxtext_utils_test.py` and `train_compile_test.py`).
- **2026-08-05**: Executed TPU v5p-8 GKE workload `mohit-opt-offload-v5p9` on cluster `v5p-8-bodaborg-europe-west4-b`. Exported profile trace to GCS (`gs://runner-maxtext-logs/offload_profile`) and uploaded to XProf (`mohitkhatwani-6873267442692234662`). Verified TPU HBM usage was **0.06 GB / 95.74 GB (0.06%)**.
- **2026-08-05**: Implemented interleaved backward optimizer update + D2H state offloading (`TransferToMemoryKind('pinned_host')`) and forward H2D parameter prefetching (`TransferToMemoryKind('device')`). All unit tests passed cleanly.
- **2026-08-06**: Executed TPU v5p-8 benchmark workload `mohit-opt-offload-v5p11` with the new interleaved offloading engine. Exported profile trace to GCS (`gs://runner-maxtext-logs/offload_profile/...`) and uploaded to XProf (`mohitkhatwani-17313608384370693559`).
- **2026-08-06**: Implemented `interleaved_scanned_forward_backward_optimizer` in `src/maxtext/utils/maxtext_utils.py` and updated `train.py`.
- **2026-08-06**: Executed TPU v5p-8 benchmark workload `mohit-opt-offload-v5p12` with the integrated 3-way interleaved forward-backward-optimizer engine. Exported profile trace to GCS (`gs://runner-maxtext-logs/offload_profile/...`) and uploaded to XProf (`mohitkhatwani-15326528413640647064`).
- **2026-08-06**: Launching baseline workload `mohit-opt-offload-v5p13` WITHOUT host offloading (`parameter_memory_host_offload=false`, `optimizer_memory_host_offload=false`) to perform direct performance and XProf trace comparison against `v5p12`.
- **2026-08-06**: Ran full unit test suite on `mohit/opt-off-trial` (150 passed, 17 skipped, 13 failed). Invoked `tpu-raiden-debugger` to analyze root causes and debug failures in `maxtext_utils_test.py`, `nnx_scan_test.py`, and `train_compile_test.py`.
- **2026-08-06**: `tpu-raiden-debugger` resolved all 13 unit test failures (path isolation, multimodal image masks, libtpu multi-process lockfile, and mock tokenizer). Synchronized code to TPU-VM (`t1v-n-666717f0-w-0.europe-west4-b.cloud-tpu-multipod-dev`) and verified with 100% test pass rate (`14 passed, 166 deselected in 240.96s`).
- **2026-08-06**: Launched `mohit-qn80b-v6e-0806224006` on v6e-256 cluster `bodaborg-v6e-256-lcscld-c`. Diagnosed compilation crash: PyTree structure mismatch between `pjit out_shardings` and `state.opt_state` under `opt_type=muon` and `mhc.py` static tensors. Invoked `tpu-raiden-debugger` to align sharding trees and verify via targeted compilation test.
- **2026-08-07**: Stopped slow full-model CPU AOT compile process. Executed targeted optimizer offload and scanned layer unit tests (`TestOptimizerOffloading` and `TestApplyScannedLayers`) directly on TPU-VM with 100% pass rate (`9 passed in 2.82s`).
- **2026-08-07**: Successfully launched v6e-256 workload `mohit-qn80b-v6e-0807201639` on cluster `bodaborg-v6e-256-lcscld-c` (`run_v6e256_qwen3.sh`) with Qwen3-Next 80B Muon optimizer and double-buffered layer-wise host offloading. Verified all 64/64 pods Running (256 v6e chips) and PJRT mesh initialized cleanly.





