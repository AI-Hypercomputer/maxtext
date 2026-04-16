# Copyright 2023–2026 Google LLC
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

"""Standalone script to benchmark BLEU and Self-BLEU for DBS vs Baseline."""

import os
import sys
import jax
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from maxtext.configs import pyconfig

def calculate_self_bleu(generated_beams, bleu_metric):
    """Calculates diversity by measuring pairwise BLEU among generated beams.
    Higher Self-BLEU means LESS diversity.
    """
    if len(generated_beams) <= 1:
        return 1.0
    
    scores = []
    for i in range(len(generated_beams)):
        prediction = generated_beams[i]
        # We provide all other beams as valid references for the current beam
        references = [[generated_beams[j] for j in range(len(generated_beams)) if i != j]]
        # Prediction: str, Reference: list[list[str]]
        res = bleu_metric.compute(predictions=[prediction], references=references)
        scores.append(res['bleu'])
    return float(np.mean(scores))

def main():
    # 1. Initialize MaxText Config
    # Usage: python3 scripts/benchmark_bleu.py src/maxtext/configs/base.yml ...
    is_mock = "--mock" in sys.argv
    if is_mock:
        sys.argv.remove("--mock")
    
    # Manually extract DBS-specific flags that aren't in the base MaxText config schema
    max_examples = 100
    decode_num_beams = 1
    decode_diversity_penalty = 0.0
    decode_num_beam_groups = 1
    # Path to write debug logs (defaults to ~/debug_log.txt)
    debug_log_path = os.path.expanduser("~/debug_log.txt")
    
    # Rebuild sys.argv to strip out DBS flags
    new_argv = []
    # Keys we want to strip from the command line because they aren't standard MaxTextConfig fields
    dbs_keys_to_strip = {"max_dataset_examples", "decode_num_beams", "decode_diversity_penalty", "decode_num_beam_groups", "debug_log_path"}
    
    for arg in sys.argv:
        # Check if this arg is a key=val pair we should strip
        is_dbs_flag = False
        if "=" in arg:
            # Handle both key=val and --key=val
            key_part = arg.split("=", 1)[0].lstrip("-")
            if key_part in dbs_keys_to_strip:
                val = arg.split("=", 1)[1]
                if key_part == "max_dataset_examples":
                    max_examples = int(val)
                elif key_part == "decode_num_beams":
                    decode_num_beams = int(val)
                elif key_part == "decode_diversity_penalty":
                    decode_diversity_penalty = float(val)
                elif key_part == "decode_num_beam_groups":
                    decode_num_beam_groups = int(val)
                elif key_part == "debug_log_path":
                    debug_log_path = os.path.expanduser(val)
                is_dbs_flag = True
        
        if not is_dbs_flag:
            new_argv.append(arg)
    
    print(f"DEBUG: Cleaned argv before pyconfig: {new_argv}")
    # Force sys.argv to be the cleaned version
    sys.argv = new_argv
    
    def debug_log(msg, mode="a"):
        if debug_log_path:
            with open(debug_log_path, mode) as f:
                f.write(msg + "\n")
                f.flush()
        print(msg)

    debug_log("DEBUG: pyconfig starting...", mode="w")
    config = pyconfig.initialize(sys.argv)
    
    debug_log(f"DEBUG: Config parallelisms - FSDP: {config.ici_fsdp_parallelism}, Tensor: {config.ici_tensor_parallelism}, Data: {config.ici_data_parallelism}")
    debug_log("DEBUG: pyconfig initialized. Injecting DBS flags...")
    
    # Manually inject the extracted flags back into the config object
    # (Using setattr because MaxConfig is a Pydantic model)
    object.__setattr__(config, 'decode_num_beams', decode_num_beams)
    object.__setattr__(config, 'decode_diversity_penalty', decode_diversity_penalty)
    debug_log("DEBUG: DBS flags injected.")
    
    if is_mock:
        debug_log(f"RUNNING IN MOCK MODE (No model loading) - Beams: {decode_num_beams}")
        engine = None
        params = None
    else:
        debug_log("DEBUG: Importing MaxEngine...")
        from maxtext.inference.maxengine import maxengine
        debug_log("DEBUG: Initializing MaxEngine...")
        # Note: MaxEngine initialization sets up the JAX mesh and loaded model
        engine = maxengine.MaxEngine(config)
        debug_log("DEBUG: MaxEngine initialized.")
    
    # 4. AOT Compile (Optional but recommended for TPU performance/memory)
    print("Compiling engine...")
    if engine is not None and not is_mock:
        try:
            # We don't need to use the returned values, just triggering the compilation
            # Use dummy rng and config for compilation if needed, or just let JIT handle it.
            # MaxEngine.aot_compile usually needs the actual params to finalize layouts.
            # Since we haven't loaded params yet, we can either move loading up or skip AOT here.
            print("AOT Compilation skipped at this step (will be handled by first JIT).")
        except Exception as e:
            print(f"AOT Compilation failed: {e}")
    
    # 5. Load Dataset (CNN/DailyMail)
    print(f"Loading CNN/DailyMail Goal: {max_examples} samples...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
    
    # 3. Load BLEU metric
    print("Loading BLEU metric...")
    try:
        bleu = evaluate.load("bleu")
    except Exception as e:
        print(f"Could not load BLEU metric: {e}.")
        return

    # 4. Initialize Model Params
    if not is_mock:
        rng = jax.random.PRNGKey(1234)
        rng, rng_load = jax.random.split(rng)
        params = engine.load_params(rng_load)
    else:
        rng = jax.random.PRNGKey(1234)
        params = None
    
    results = {
        "bleu": [],
        "self_bleu": []
    }

    print(f"Starting Benchmark on {max_examples} examples...")
    for i, example in enumerate(tqdm(dataset, total=max_examples)):
        if i >= max_examples:
            break
            
        prompt = f"Summarize the following article:\n{example['article']}\n\nSummary:"
        reference = example['highlights']

        if is_mock:
            # Mock behavior: generate num_beams strings based on reference
            beams = [f"Beam {j} summary variant {reference[:30]}" for j in range(decode_num_beams)]
            output_text_or_beams = beams
        else:
            tokenizer = engine.build_tokenizer(engine.get_tokenizer())
            tokens, true_length = tokenizer.encode(prompt, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
            
            # Pad input to total_batch_size to support FSDP sharding
            # Note: total_batch_size = devices * per_device_batch_size
            total_bs = int(jax.device_count() * config.per_device_batch_size)
            current_bs = tokens.shape[0]
            if current_bs < total_bs:
                padding = jnp.zeros((total_bs - current_bs, tokens.shape[1]), dtype=tokens.dtype)
                tokens = jnp.concatenate([tokens, padding], axis=0)
                padding_len = jnp.zeros((total_bs - current_bs,), dtype=true_length.dtype)
                true_length = jnp.concatenate([true_length, padding_len], axis=0)

            rng, rng_gen = jax.random.split(rng)
            # Prefill once for the whole beam group
            prefill_res, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
            
            # Prepare state with multiple slots for each beam
            decode_state = engine.init_decode_state(rng_gen)
            slots = list(range(config.batch_size))
            # bulk_insert replicates the prefill results into all beam slots
            decode_state = engine.bulk_insert(prefill_res, decode_state, slots=slots)
            # Safety: Ensure is_dbs is preserved (in case MaxEngine isn't updated)
            if "is_dbs" not in decode_state:
                decode_state["is_dbs"] = jnp.array([config.decode_sampling_strategy == "diverse_beam_search"])
            if "cumulative_logprobs" not in decode_state:
                decode_state["cumulative_logprobs"] = jnp.zeros_like(decode_state["tokens"], dtype=jnp.float32)
            
            # Start all beams with the same first token from prefill
            token0 = first_token.get_result_at_slot(0).tokens.item()
            beam_tokens = [[token0] for _ in slots]
            beam_finished = [False] * config.batch_size
            
            for _ in range(config.max_prefill_predict_length, config.max_target_length):
                decode_state, sampled_tokens = engine.generate(params, decode_state)
                for s in slots:
                    if not beam_finished[s]:
                        tok = sampled_tokens.get_result_at_slot(s).tokens.item()
                        beam_tokens[s].append(tok)
                        if tok == tokenizer.eos_id:
                            beam_finished[s] = True
                if all(beam_finished):
                    break
                    
            output_text_or_beams = [tokenizer.decode(b) for b in beam_tokens]
    
        # Handle predictions and scoring
        if isinstance(output_text_or_beams, list):
            top_prediction = output_text_or_beams[0]
            beams = output_text_or_beams
        else:
            top_prediction = output_text_or_beams
            beams = [output_text_or_beams]

        # Score Quality (Top-1 BLEU)
        score = bleu.compute(predictions=[top_prediction], references=[[reference]])
        results["bleu"].append(score['bleu'])
        
        # Score Diversity (Self-BLEU across beams)
        if len(beams) > 1:
            sb = calculate_self_bleu(beams, bleu)
            results["self_bleu"].append(sb)

    # 7. Print Final Metrics
    print("\n" + "="*40)
    print(f"BENCHMARK RESULTS\nStrategy: {config.decode_sampling_strategy}")
    print(f"Num Examples: {len(results['bleu'])}\nAvg BLEU-4: {np.mean(results['bleu']):.4f}")
    if results["self_bleu"]:
        print(f"Avg Self-BLEU: {np.mean(results['self_bleu']):.4f} (Lower = More Diverse)")
    print("="*40)

if __name__ == "__main__":
    main()
