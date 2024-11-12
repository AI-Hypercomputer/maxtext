#!/bin/bash

echo "Running the jax.distributed.initialize test";
python -c "import jax; jax.distributed.initialize()";
if [["$?" -eq "0"]]
  echo "Test exit status 0, success!"
else
  echo "Non-zero exit status, test failed!"
