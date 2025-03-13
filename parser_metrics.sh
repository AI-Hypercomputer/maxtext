#!/bin/bash
mlperf_log='/tmp/large_scale_multislice_test_log'
# Calculate block_time for each first_epoch_num
total=0
num=0
avgblocktime=0
grep "block_start" ${mlperf_log} | sort -u | while read -r line ; do
  echo "Processing $line"

  first_epoch_num=$(echo "$line" | grep -oP '(?<="first_epoch_num": )[0-9]+')
  block_start=$(echo "$line" | grep -oP '(?<="time_ms": )[0-9]+(?=.*block_start)')
  block_stop=$(grep -oP "(?<=\"time_ms\": )[0-9]+(?=.*block_stop.*first_epoch_num\": $first_epoch_num})" ${mlperf_log} | sort -u)

  echo "first_epoch_num: ${first_epoch_num}"
  echo "block_start: ${block_start}"
  echo "block_stop: ${block_stop}"

  if ! [[ -z "$block_start" ]] && ! [[ -z "$block_stop" ]]
  then
    echo ":::E2E summary::: block_time: $((block_stop - block_start))"
    block_time=$((block_stop - block_start))

    total=$((total + block_time))
    num=$((num + 1))
  fi
  avgblocktime=$(echo $total / $num | bc -l)
  echo ":::E2E summary::: mov_avg block_time: ${avgblocktime}"
done

# Calculate e2e_time
run_start=$(grep -oP "(?<=\"time_ms\": )[0-9]+(?=.*run_start)" ${mlperf_log} | sort -u)
run_stop=$(grep -oP "(?<=\"time_ms\": )[0-9]+(?=.*run_stop)" ${mlperf_log} | sort -u)

echo "run_start: ${run_start}"
echo "run_stop: ${run_stop}"

if ! [[ -z "$run_start" ]] && ! [[ -z "$run_stop" ]]
then
  echo ":::E2E summary::: e2e_time: $((run_stop - run_start))"
fi