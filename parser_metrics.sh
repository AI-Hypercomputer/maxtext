#!/bin/bash
mlperf_log='/tmp/large_scale_multislice_test_log'
# Calculate block_time for each first_epoch_num
grep "block_start" ${mlperf_log} | sort -u | while read -r line ; do
    echo "Processing $line"

    first_epoch_num=$(echo "$line" | grep -oP '(?<="first_epoch_num": )[0-9]+')
    block_start=$(echo "$line" | grep -oP '(?<="time_ms": )[0-9]+(?=.*block_start)')
    block_stop=$(grep -oP "(?<=\"time_ms\": )[0-9]+(?=.*block_stop.*first_epoch_num\": $first_epoch_num})" ${mlperf_log} | sort -u)

    echo "first_epoch_num: ${first_epoch_num}"
    echo "block_start: ${block_start}"
    echo "block_stop: ${block_stop}"

    if [[ ! -z "$block_start" ]] && [[ ! -z "$block_stop" ]]
    then
    echo ":::E2E summary::: block_time: $((block_stop - block_start))"
    fi
done

# Calculate e2e_time
run_start=$(grep -oP "(?<=\"time_ms\": )[0-9]+(?=.*run_start)" ${mlperf_log} | sort -u)
run_end=$(grep -oP "(?<=\"time_ms\": )[0-9]+(?=.*run_end)" ${mlperf_log} | sort -u)

echo "run_start: ${run_start}"
echo "run_end: ${run_end}"

if ! [[ -z "$run_start" ]] && ! [[ -z "$run_end" ]]
then
  echo ":::E2E summary::: e2e_time: $((run_end - run_start))"
fi