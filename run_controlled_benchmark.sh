#!/usr/bin/env bash

# Launcher script: run_controlled_benchmark.sh
# Runs the controlled e2e benchmark with various configurations.
# Includes robust parsing and proactive cleanup of stale processes.

# --- VERSION CHECK ---
echo "--- EXECUTING SCRIPT VERSION: v7 (Proactive Process Cleanup) ---"

set -o pipefail

# --- Configuration ---
OUTPUT_DIR="./benchmark_outputs_controlled"
mkdir -p "${OUTPUT_DIR}"

BASE_CONFIG="MaxText/configs/base.yml"
CHECKPOINT=""
TOKENIZER="assets/tokenizer.llama2"

# Common parameters (as a bash array for robustness)
COMMON_ARGS=(
    "model_name=llama2-70b"
    "tokenizer_path=${TOKENIZER}"
    "load_parameters_path=${CHECKPOINT}"
    "quantization=int8"
    "checkpoint_is_quantized=True"
    "ici_fsdp_parallelism=1"
    "ici_tensor_parallelism=-1"
    "scan_layers=false"
    "weight_dtype=bfloat16"
)

# --- NEW: Cleanup Function ---
cleanup_stale_processes() {
    echo "--- Cleaning up any stale benchmark processes... ---"
    # This command finds any python3 process that has 'controlled_e2e_benchmark.py'
    # in its command line arguments and terminates it.
    # The '|| true' ensures that the script doesn't fail if no such processes are found.
    pkill -f "python3 controlled_e2e_benchmark.py" || true
    # Add a small delay to allow the OS to release resources
    sleep 2
}


# --- BENCHMARK CONFIGURATIONS ---
# This strategic list uses fewer requests for longer sequences to keep runtimes reasonable.
# Format: "attention:batch:pages:tpp:ppb:workload:requests:prefill:target"
declare -a SCENARIOS=(
    # ==========================================================================
    # === Suite 1: Full Matrix at 2K Sequence Length (target=2048)
    # ==========================================================================
    # Goal: Precisely map the performance curves for all workloads at a standard size.
    # Requests: 3000

    # --- 2K, CHAT Workload (short/short) ---
    "dot_product:16:::chat:3000:1024:2048"
    "dot_product:24:::chat:3000:1024:2048"
    "dot_product:28:::chat:3000:1024:2048" # Find wall
    "paged:32:15000:32:4:chat:3000:1024:2048"
    "paged:48:15000:32:4:chat:3000:1024:2048"
    "paged:64:15000:32:4:chat:3000:1024:2048" # Find knee

    # --- 2K, SUMMARIZE Workload (long/short) ---
    "dot_product:16:::summarize:3000:1024:2048"
    "dot_product:24:::summarize:3000:1024:2048"
    "dot_product:28:::summarize:3000:1024:2048"
    "paged:24:15000:32:4:summarize:3000:1024:2048"
    "paged:32:15000:32:4:summarize:3000:1024:2048"
    "paged:40:15000:32:4:summarize:3000:1024:2048"

    # --- 2K, GENERATIVE Workload (short/long) ---
    "dot_product:16:::generative:3000:1024:2048"
    "dot_product:24:::generative:3000:1024:2048"
    "dot_product:28:::generative:3000:1024:2048"
    "paged:24:15000:32:4:generative:3000:1024:2048"
    "paged:32:15000:32:4:generative:3000:1024:2048"
    "paged:40:15000:32:4:generative:3000:1024:2048"
    
    # --- 2K, LONG Workload (long/long, the dot_product best case) ---
    "dot_product:16:::long:3000:1024:2048"
    "dot_product:24:::long:3000:1024:2048"
    "dot_product:28:::long:3000:1024:2048"
    "paged:24:15000:32:4:long:3000:1024:2048"
    "paged:32:15000:32:4:long:3000:1024:2048"
    "paged:40:15000:32:4:long:3000:1024:2048"


    # ==========================================================================
    # === Suite 2: Full Matrix at 4K Sequence Length (target=4096)
    # ==========================================================================
    # Goal: Understand how the performance curves shift with longer context.
    # Requests: 1500
    # Page Pool: Use 25000 pages to maintain good token coverage.
    
    # --- 4K, SUMMARIZE Workload (long/short), a key long-context use case ---
    "dot_product:8:::summarize:1500:2048:4096"
    "dot_product:12:::summarize:1500:2048:4096"
    "dot_product:16:::summarize:1500:2048:4096" # Likely fails
    "paged:16:25000:32:4:summarize:1500:2048:4096"
    "paged:24:25000:32:4:summarize:1500:2048:4096"
    "paged:32:25000:32:4:summarize:1500:2048:4096"

    # --- 4K, GENERATIVE Workload (short/long) ---
    "dot_product:8:::generative:1500:2048:4096"
    "dot_product:12:::generative:1500:2048:4096"
    "paged:16:25000:32:4:generative:1500:2048:4096"
    "paged:24:25000:32:4:generative:1500:2048:4096"
    "paged:32:25000:32:4:generative:1500:2048:4096"


    # ==========================================================================
    # === Suite 3: Limit Testing at 8K Sequence Length (target=8192)
    # ==========================================================================
    # Goal: Test extreme long-context for paged attention.
    # Requests: 1000
    # Page Pool: Use 25000 pages.
    
    # --- 8K, SUMMARIZE Workload (long/short) ---
    "dot_product:4:::summarize:1000:4096:8192"  # The "can it even run?" test
    "paged:8:25000:32:4:summarize:1000:4096:8192"
    "paged:12:25000:32:4:summarize:1000:4096:8192"
    "paged:16:25000:32:4:summarize:1000:4096:8192" # Find the 8K peak
)


echo "Starting controlled benchmark sweep..."
echo "Output directory: ${OUTPUT_DIR}"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    # Call cleanup before every single run to ensure a clean state
    cleanup_stale_processes
    
    # Pipe awk's output to a 'while read' loop. This is the most compatible way.
    echo "$scenario" | awk -F: '{print $1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9}' | \
    while read -r attention batch pages tpp ppb workload requests prefill target; do

        # Build a descriptive run name
        run_name="controlled_${attention}_bs${batch}"
        if [ -n "$prefill" ] && [ -n "$target" ]; then
            run_name="${run_name}_pf${prefill}_tg${target}"
        fi
        if [ "$attention" = "paged" ]; then
            run_name="${run_name}_np${pages}_tpp${tpp}"
        fi
        run_name="${run_name}_${workload}_${requests}req"

        # Build the command array for execution
        CMD=(
            python3 controlled_e2e_benchmark.py
            --config_path "${BASE_CONFIG}"
            --num_requests "${requests}"
            --workload_type "${workload}"
            --warmup_steps "100"
            --output_file "${OUTPUT_DIR}/${run_name}_detailed.json"
        )

        # Add pyconfig arguments
        CMD+=("${COMMON_ARGS[@]}")
        CMD+=("base_output_directory=${OUTPUT_DIR}")
        CMD+=("run_name=${run_name}")
        CMD+=("attention=${attention}")
        CMD+=("per_device_batch_size=${batch}")

        # Add paged attention specific args
        if [ "$attention" = "paged" ]; then
            [ -n "$pages" ] && CMD+=("pagedattn_num_pages=${pages}")
            [ -n "$tpp" ] && CMD+=("pagedattn_tokens_per_page=${tpp}")
            [ -n "$ppb" ] && CMD+=("pagedattn_pages_per_compute_block=${ppb}")
        fi

        # Add prefill and target length arguments
        [ -n "$prefill" ] && CMD+=("max_prefill_predict_length=${prefill}")
        [ -n "$target" ] && CMD+=("max_target_length=${target}")

        output_log="${OUTPUT_DIR}/${run_name}.log"

        echo "======================================================================"
        echo "Running: ${run_name}"
        echo "Command: ${CMD[@]}"
        echo "Output: ${output_log}"
        echo "======================================================================"

        # Run the benchmark
        "${CMD[@]}" 2>&1 | tee "${output_log}"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Completed successfully"
            throughput=$(grep "Overall Throughput:" "${output_log}" | awk '{print $3}')
            p99_ttft=$(grep "Time to First Token (p99):" "${output_log}" | awk '{print $6}')

            echo "  Throughput: ${throughput} tokens/sec"
            echo "  P99 TTFT: ${p99_ttft} ms"
        else
            echo "✗ Failed with exit code ${PIPESTATUS[0]}"
        fi
        echo ""

        # Since we only process one line from the pipe, break the loop
        break
    done
done

# Final Summary Table
echo "======================================================================"
echo "SUMMARY OF ALL RUNS"
echo "======================================================================"
printf "%-90s | %-15s | %-15s | %-15s\n" "Configuration" "Throughput" "P99 TTFT" "Status"
echo "------------------------------------------------------------------------------------------------------------------------------"

for log in "${OUTPUT_DIR}"/controlled_*.log; do
    if [ -f "$log" ]; then
        config=$(basename "$log" .log)
        throughput=$(grep "Overall Throughput:" "$log" | awk '{print $3}')
        p99_ttft=$(grep "Time to First Token (p99):" "${output_log}" | awk '{print $6}')

        if [ -n "$throughput" ]; then
            status="Success"
        else
            status="Failed"
            throughput="N/A"
            p99_ttft="N/A"
        fi

        printf "%-90s | %-15s | %-15s | %-15s\n" "$config" "$throughput tok/s" "$p99_ttft ms" "$status"
    fi
done

echo ""
echo "Detailed results saved in: ${OUTPUT_DIR}"