#!/bin/bash

# Copies a baseline directory into a new directory so it can be loaded

baseline_to_copy=${1}
new_run_name=${2}

gsutil -m cp -r gs://maxtext-experiments-multipod/${baseline_to_copy} gs://maxtext-experiments-multipod/${new_run_name}