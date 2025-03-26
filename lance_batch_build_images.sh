#!/bin/bash

# start_date="2025-03-27"
# end_date="2025-04-01"
# skip_dates=("2025-03-31" "2025-03-30" "2025-03-28" "2025-03-26"  "2025-03-24" "2025-03-23" "2025-03-21")  # Add any dates you want to skip
start_date="2025-04-01"
end_date="2025-04-02"

build_mode="jsts_nightly"
# build_mode="jsts_stable"
output_dir="jax3p_nightly"
# output_dir="jax3p_stable"

current_date="$start_date"

while [ "$current_date" != "$(date -I -d "$end_date + 1 day")" ]; do
  # Check if current_date is in the skip list
  skip=false
  for skip_date in "${skip_dates[@]}"; do
    if [ "$current_date" == "$skip_date" ]; then
      skip=true
      break
    fi
  done

  if [ "$skip" == false ]; then
    export TAG="$current_date"
    echo "Running for TAG=$TAG"
    # Replace this with your actual command
    bash docker_build_dependency_image.sh DEVICE=gpu MODE=$build_mode JSTS_VERSION=$TAG; docker tag maxtext_base_image gcr.io/supercomputer-testing/$output_dir:$TAG; docker push gcr.io/supercomputer-testing/$output_dir:$TAG
  else
    echo "Skipping $current_date"
  fi

  # Move to next day
  current_date=$(date -I -d "$current_date + 1 day")
done
