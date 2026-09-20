RWKV-7
======

RWKV-7 "Goose" (BlinkDL's x070 architecture) is an attention-free language model. Each layer's time mixing keeps a fixed-size per-head state updated by a generalized delta rule with dynamic, per-channel decay (the WKV7 recurrence), so training cost is linear in sequence length and decoding keeps constant memory per sequence. MaxText supports the x070 checkpoints BlinkDL publishes (validated: `rwkv7-g1d-0.1b-20260129-ctx8192.pth`, config `rwkv7-0.1b`). See [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) for the architecture.

* * * * *

Checkpoint conversion
---------------------

Download a BlinkDL `.pth` (e.g. `rwkv7-g1d-0.1b-20260129-ctx8192.pth` from the `BlinkDL/rwkv7-g1` Hugging Face repository) and convert it to an unscanned MaxText checkpoint:

```sh
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_rwkv7_unscanned \
    --checkpoint_path rwkv7-g1d-0.1b-20260129-ctx8192.pth \
    --output_dir ${CHECKPOINT_DIR}
```

It prints the model dimensions it found; load the result with `load_parameters_path=${CHECKPOINT_DIR}/0/items`.

Decoding
--------

```sh
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
    model_name=rwkv7-0.1b \
    load_parameters_path=${CHECKPOINT_DIR}/0/items \
    per_device_batch_size=1 max_prefill_predict_length=16 max_target_length=24 \
    decode_sampling_strategy=greedy \
    prompt="The Eiffel tower is in the city of"
```

The tokenizer defaults to BlinkDL's World vocabulary (`tokenizer_type=rwkv`, `src/maxtext/assets/tokenizers/rwkv_vocab_v20230424.txt`). It has no BOS token (`add_bos: false`); token 0 ends a document and is the tokenizer's stop token. MaxEngine keeps one fixed-size recurrent cache per decode slot; multi-prompt packed prefill (`prefill_concat`) is not supported.

Pre-training
------------

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=rwkv7_pre_training \
    model_name=rwkv7-0.1b \
    per_device_batch_size=8 max_target_length=4096 \
    learning_rate=8e-4 steps=1000 \
    dataset_type=synthetic
```

`rwkv7-0.1b.yml` carries BlinkDL's training recipe: weight decay only on the dense matrices, embedding and head (`adamw_mask`), 2x learning rate for the decay bias `w0` (`lr_multipliers`), Adam betas (0.9, 0.99) and eps 1e-18, and L2Wrap (`logits_l2wrap_factor: 1e-4`, gradient-only). The learning-rate schedule is left to the run.

Packing is supported. With `rwkv7_segment_resets=true` (the default) each document in a packed row starts from a zero state; `rwkv7_segment_resets=false` treats the row as one continuous stream, as BlinkDL's own training does.

Ahead-of-time compilation for a TPU topology works without TPU hardware:

```sh
python3 -m maxtext.trainers.pre_train.train_compile src/maxtext/configs/base.yml \
    model_name=rwkv7-0.1b compile_topology=v5e-8 compile_topology_num_slices=1 \
    per_device_batch_size=1 max_target_length=2048 dataset_type=synthetic
```

WKV implementations
-------------------

`rwkv7_wkv_impl` selects the recurrence:

- `naive` (default): a `lax.scan` reference, the correctness oracle.
- `pallas`: a Pallas TPU kernel that steps through each chunk of `rwkv7_wkv_chunk_size` timesteps (default 16, a multiple of 8).
- `pallas_chunked`: a Pallas TPU kernel that evaluates each chunk as matmuls; `rwkv7_wkv_chunk_size` at most 128.

Both kernels have custom backward passes that recompute states from per-chunk checkpoints, run under `shard_map` on multi-device meshes, and use interpret mode off-TPU. Their relative speed on TPU has not been measured yet.

Validation status and limitations
---------------------------------

- Validated on CPU (float32): logits against BlinkDL's reference implementation and the `rwkv` package on the 0.1B checkpoint, decoding through MaxEngine, training, and packed rows; all three WKV implementations. bf16 activations were measured on CPU at about 1e-2 maximum relative logit deviation from float32, with identical top-1 tokens.
- Compiled for TPU (v5e) ahead of time, including both Pallas kernels. Execution on TPU hardware and performance are not yet verified.
- Not supported: `scan_layers=true` and pipeline parallelism (the layers aren't interchangeable: layer 0 produces the value residual `v_first` used by every later layer), and multi-prompt packed prefill.
