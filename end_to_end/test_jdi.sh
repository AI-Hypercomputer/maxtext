#!/bin/sh

echo "Running the jax.distributed.initialize test";
if python3 -c "import jax; jax.distributed.initialize()"; then
  echo "Test exit status 0, success!"
else
  echo "Non-zero exit status, test failed!"
fi
