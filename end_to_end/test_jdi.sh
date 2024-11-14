#!/bin/bash

echo "Running the jax.distributed.initialize test";
python3 -c "import jax; jax.distributed.initialize()";
if [[ "$?" -eq "0" ]]; then
  echo "Test exit status 0, success!"
else
  echo "Non-zero exit status, test failed!"
fi
