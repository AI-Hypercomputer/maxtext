#!/bin/bash

start_date="2025-03-04"
end_date="2025-03-18"
skip_dates=()  # Add any dates you want to skip

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
    bash docker_build_dependency_image.sh DEVICE=gpu MODE=jsts_nightly JSTS_VERSION=$TAG; docker tag maxtext_base_image gcr.io/supercomputer-testing/jax3p_nightly:$TAG; docker push gcr.io/supercomputer-testing/jax3p_nightly:$TAG
  else
    echo "Skipping $current_date"
  fi

  # Move to next day
  current_date=$(date -I -d "$current_date + 1 day")
done
