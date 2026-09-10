#!/bin/bash

# Exit on error
set -e

PYTHON=/home/liyinn_google_com/anaconda3/envs/maxtext/bin/python3
NOW=$(date +%Y%m%d_%H%M%S)

echo "Starting job at $NOW"

# Run training
echo "Starting training..."
PYTHONPATH=./src $PYTHON src/maxtext/experimental/cosmos_generator/trainers/train_vae.py > ./output_${NOW}.txt 2>&1

echo "Training finished."

# 3. Rename output images to include timestamp
if [ -f "./vae_reconstruction.png" ]; then
  mv ./vae_reconstruction.png ./vae_reconstruction_${NOW}.png
  echo "Saved reconstruction image as ./vae_reconstruction_${NOW}.png"
fi

if [ -f "./vae_generation.png" ]; then
  mv ./vae_generation.png ./vae_generation_${NOW}.png
  echo "Saved generation image as ./vae_generation_${NOW}.png"
fi

echo "Log saved to ./output_${NOW}.txt"
