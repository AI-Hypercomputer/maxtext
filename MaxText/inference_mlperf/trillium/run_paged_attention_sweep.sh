#!/usr/bin/env bash

# Launcher script: run_paged_attention_sweep.sh
# CWD when running this script: maxtext/MaxText/inference_mlperf/trillium/

# --- Configuration ---
# BENCHMARK_SCRIPT is in the same directory as this launcher.
BENCHMARK_SCRIPT="./benchmarks_llama2-70b-trillium_2x4.sh"

BENCHMARK_TYPE_TO_RUN="performance"
RUN_NAME_PREFIX="5k_sweep_compute_order"
OUTPUT_DIR="./benchmark_outputs_5k_compute_order" # Will be created in CWD (trillium/)
mkdir -p "${OUTPUT_DIR}"

# export USER_PREFILL_LEN_GLOBAL="2048" # Optional

# attention:num_pages:tokens_per_page:pages_per_compute_block:per_device_batch_size:prefill_length:run_type
declare -a SCENARIOS=(
    # "paged:16500:32:4:32:1024:test"
    # "paged:15000:32:4:4:2048:test"
    # "dot_product:15000:32:4:8:1024:test"
    # "paged:4500:32:4:8:1024:test"
    # "dot_product:15000:32:4:16:1024:test"
    # "paged:9000:32:4:16:1024:test"
    # "dot_product:15000:32:4:24:1024:test"
    # "paged:13500:32:4:24:1024:test"
    "dot_product:15000:32:4:32:1024:test"
    # "paged:18000:32:4:32:1024:test"
    # "paged:16500:32:4:32:1024:test"
    "paged:15000:32:4:32:1024:test"
    # "paged:16500:32:4:36:1024:test"
    #"paged:16500:32:4:38:1024:test"
    # "paged:16500:32:4:40:1024:test"
    #"paged:16500:32:4:42:1024:test"
    #"paged:16500:32:4:44:1024:test"
    #"paged:16500:32:4:46:1024:test"
    #"paged:16500:32:4:48:1024:test"
    #"paged:16500:32:4:50:1024:test"
    #"paged:16500:32:4:52:1024:test"
    # "paged:7000:64:2:5:2048"
    # "paged:7000:64:2:6:2048"
    # "paged:7000:64:2:7:2048"
    # "paged:7000:64:2:8:2048"
    # "paged:7000:64:2:9:2048"
    # "paged:7000:64:2:10:2048"
    # "paged:7000:64:2:11:2048"
    # "paged:7000:64:2:12:2048"
    # "paged:7000:64:2:13:2048"
    # "paged:7000:64:2:14:2048"
    # "paged:7000:64:2:15:2048"
    # "paged:7000:64:2:16:2048"
    # "paged:15000:32:8:10:2048"
    # "paged:7500:64:4:10:2048"
    # "paged:3750:128:1:10:2048"
    # "paged:3750:128:2:10:2048"
    # "paged:3750:128:4:10:2048"
)

if [ ! -x "${BENCHMARK_SCRIPT}" ]; then
    echo "Error: Benchmark script ${BENCHMARK_SCRIPT} not found or not executable from $(pwd)." >&2
    exit 1
fi

echo "Starting comprehensive benchmark sweep from $(pwd)..."
echo "Output will be saved in: ${OUTPUT_DIR}"
echo ""

for i in "${!SCENARIOS[@]}"; do
    scenario_params="${SCENARIOS[$i]}"
    IFS=':' read -r -a params_array <<< "$scenario_params"

    attn_type="${params_array[0]}"
    num_pages="${params_array[1]}"
    tokens_per_page="${params_array[2]}"
    pages_per_block="${params_array[3]}"
    batch_size="${params_array[4]}"
    prefill_len_spec="${params_array[5]:-default}" 
    test_mode_str="${params_array[6]}"              

    echo "======================================================================"
    echo "Running Scenario $((i+1))/${#SCENARIOS[@]}"
    echo "  Attention Type:        ${attn_type}"
    echo "  Num Pages:             ${num_pages}"
    echo "  Tokens per Page:       ${tokens_per_page}"
    echo "  Pages per Block:       ${pages_per_block}"
    echo "  Per-Device Batch Size: ${batch_size}"
    echo "  Test Mode specified:   ${test_mode_str:-<not_specified>}"

    export USER_ATTENTION_TYPE="${attn_type}"
    export USER_PAGEDATTN_NUM_PAGES="${num_pages}"
    export USER_PAGEDATTN_TOKENS_PER_PAGE="${tokens_per_page}"
    export USER_PAGEDATTN_PAGES_PER_COMPUTE_BLOCK="${pages_per_block}"
    export USER_BATCH_SIZE_PER_DEVICE="${batch_size}"

    effective_prefill_len_for_run_name="<default>" 
    unset USER_PREFILL_LEN 

    if [ "${prefill_len_spec}" != "default" ] && [ -n "${prefill_len_spec}" ]; then
        export USER_PREFILL_LEN="${prefill_len_spec}"
        effective_prefill_len_for_run_name="${USER_PREFILL_LEN}"
        echo "  Prefill Length (Scen): ${USER_PREFILL_LEN}"
    elif [ -n "${USER_PREFILL_LEN_GLOBAL:-}" ]; then
        export USER_PREFILL_LEN="${USER_PREFILL_LEN_GLOBAL}"
        effective_prefill_len_for_run_name="${USER_PREFILL_LEN}"
        echo "  Prefill Length (Glob): ${USER_PREFILL_LEN}"
    else
        effective_prefill_len_for_run_name="script_default_2048" 
        echo "  Prefill Length (Def): Benchmark script will use its default (e.g., 2048)"
    fi
    echo "======================================================================"

    script_args=()
    if [ "${attn_type}" == "paged" ]; then
        script_args+=("-g") 
    fi
    script_args+=("-b" "${BENCHMARK_TYPE_TO_RUN}")

    run_mode_suffix_for_name="full"
    if [ "${test_mode_str}" == "test" ]; then
        script_args+=("-t")
        run_mode_suffix_for_name="test"
    fi
    
    current_run_name="${RUN_NAME_PREFIX}_att-${attn_type}_np${num_pages}_tpp${tokens_per_page}_ppb${pages_per_block}_bs${batch_size}_pl${effective_prefill_len_for_run_name}_${run_mode_suffix_for_name}"
    script_args+=("-r" "${current_run_name}")

    output_file="${OUTPUT_DIR}/output_${current_run_name}.txt"

    echo "DEBUG: [Launcher] About to execute: bash ${BENCHMARK_SCRIPT} ${script_args[*]} for scenario $((i+1))" >&2
    echo "DEBUG: [Launcher] Output file will be: ${output_file}" >&2
    echo "  Effective Env Settings for ${BENCHMARK_SCRIPT} (Scenario $((i+1))):" >&2
    (env | grep USER_ATTENTION_TYPE) >&2 || echo "    USER_ATTENTION_TYPE not set or empty" >&2
    (env | grep USER_PAGEDATTN_NUM_PAGES) >&2 || echo "    USER_PAGEDATTN_NUM_PAGES not set or empty" >&2
    (env | grep USER_PAGEDATTN_TOKENS_PER_PAGE) >&2 || echo "    USER_PAGEDATTN_TOKENS_PER_PAGE not set or empty" >&2
    (env | grep USER_PAGEDATTN_PAGES_PER_COMPUTE_BLOCK) >&2 || echo "    USER_PAGEDATTN_PAGES_PER_COMPUTE_BLOCK not set or empty" >&2
    (env | grep USER_BATCH_SIZE_PER_DEVICE) >&2 || echo "    USER_BATCH_SIZE_PER_DEVICE not set or empty" >&2
    (env | grep USER_PREFILL_LEN) >&2 || echo "    USER_PREFILL_LEN is unset (benchmark script will use its default)" >&2

    echo "  Executing: bash ${BENCHMARK_SCRIPT} ${script_args[*]}"
    echo "  Output will be logged to: ${output_file}"
    echo ""

    bash "${BENCHMARK_SCRIPT}" "${script_args[@]}" > "${output_file}" 2>&1
    benchmark_exit_code=$?

    echo "DEBUG: [Launcher] bash ${BENCHMARK_SCRIPT} for scenario $((i+1)) exited with code: ${benchmark_exit_code}" >&2

    if [ ${benchmark_exit_code} -eq 0 ]; then
        echo "Scenario $((i+1)) completed successfully."
    else
        echo "Scenario $((i+1)) FAILED with exit code ${benchmark_exit_code}. Check ${output_file} for error details."
    fi
    echo "----------------------------------------------------------------------"
    echo ""
done

echo "All benchmark scenarios processed."
