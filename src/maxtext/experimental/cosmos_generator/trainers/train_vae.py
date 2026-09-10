# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for Vanilla VAE on MNIST."""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from maxtext.experimental.cosmos_generator.models.vae import VAE
from maxtext.experimental.cosmos_generator.losses import (
    kl_divergence_standard_normal,
    mse_reconstruction_loss,
)
from maxtext.utils import max_logging
from absl import app
import os

# Config
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_STEPS = 1000 # 20K steps took 12 min to run.
LATENT_DIM = 20
HIDDEN_DIM = 400
INPUT_DIM = 784
DATA_DIR = '/dev/shm/mnist_data'

def load_mnist():
  """Loads MNIST dataset using TFDS."""
  # Ensure dataset is downloaded to DATA_DIR
  ds = tfds.load('mnist', split='train', data_dir=DATA_DIR, download=True)
  ds = ds.map(lambda x: {
      'image': tf.cast(x['image'], tf.float32) / 255.0,
      'label': x['label']
  })
  ds = ds.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
  return ds

def loss_fn(model, x, rng_key):
  """Computes VAE loss (Reconstruction + KL)."""
  x_flat = x.reshape((x.shape[0], -1))
  x_hat, mu, logvar = model(x_flat, rng_key)
  
  recon_loss = mse_reconstruction_loss(x_hat, x_flat)
  kl_loss = kl_divergence_standard_normal(mu, logvar)
  
  total_loss = recon_loss + kl_loss
  return total_loss, (recon_loss, kl_loss)

def main(argv):
  del argv
  max_logging.log("Starting VAE training on MNIST")
  
  # Create data dir if not exists
  if not os.path.exists(DATA_DIR):
     os.makedirs(DATA_DIR, exist_ok=True)

  rngs = nnx.Rngs(0)
  model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, rngs=rngs)
  optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
  
  ds = load_mnist()
  ds_iter = ds.as_numpy_iterator()
  
  for step in range(NUM_STEPS):
    if step == 5:
      max_logging.log("Starting profiler trace...")
      jax.profiler.start_trace("./tensorboard/profile")

    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      try:
        batch = next(ds_iter)
      except StopIteration:
        ds_iter = ds.as_numpy_iterator()
        batch = next(ds_iter)
        
      x = batch['image']
      rng_key = jax.random.PRNGKey(step)
      
      def compute_loss(model):
        loss, (recon_loss, kl_loss) = loss_fn(model, x, rng_key)
        return loss, (recon_loss, kl_loss)

      grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
      (loss, (recon_loss, kl_loss)), grads = grad_fn(model)
      optimizer.update(model, grads)
      
    if step == 10:
      max_logging.log("Stopping profiler trace...")
      jax.profiler.stop_trace()

    if step % 100 == 0:
      max_logging.log(f"Step {step}, Loss: {loss:.4f}, Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}")

  max_logging.log("Training finished")

  # Inference verification
  try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Take a batch
    ds_iter = ds.as_numpy_iterator()
    batch = next(ds_iter)
    x = batch['image']
    x_flat = x.reshape((x.shape[0], -1))
    
    # Reconstruct
    rng_key = jax.random.PRNGKey(42)
    x_hat, mu, logvar = model(x_flat, rng_key)
    
    # Reshape back to image
    x_hat_img = x_hat.reshape((-1, 28, 28, 1))
    
    # Plot first 8 images
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(x[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        
        axes[1, i].imshow(x_hat_img[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Recon")
        
    plt.tight_layout()
    output_path = './vae_reconstruction.png'
    plt.savefig(output_path)
    max_logging.log(f"Reconstruction sample saved to {output_path}")
    
    # Also test generation from prior
    z_sample = jax.random.normal(jax.random.PRNGKey(0), (8, LATENT_DIM))
    gen_images = model.decoder(z_sample).reshape((-1, 28, 28, 1))
    
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        axes[i].imshow(gen_images[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
        
    plt.tight_layout()
    gen_output_path = './vae_generation.png'
    plt.savefig(gen_output_path)
    max_logging.log(f"Generated samples saved to {gen_output_path}")

  except Exception as e:
    max_logging.error(f"Failed to save images: {e}")

if __name__ == '__main__':
  # Prevent TF from using GPU/TPU memory if running on same pipeline
  tf.config.set_visible_devices([], 'GPU')
  app.run(main)
