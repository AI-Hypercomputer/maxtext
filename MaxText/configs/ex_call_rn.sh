#!/bin/bash

echo "running qq"
# Call the hello.sh script and capture its output using command substitution
result=$(MaxText/configs/gen-7-25-run-names.sh)

echo "ran qq"

# Print the output
echo $result