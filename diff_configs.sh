# !/bin/bash

sweep="$1"
base="$2"

for n in $(ls MaxText/configs/*${sweep}*); do
  echo;
  echo;
  echo "DIFFING: $n";
  echo;
  diff -u --color $2 "$n";
done
