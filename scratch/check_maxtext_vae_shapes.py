from maxtext.common import checkpointing
from maxtext.models.dit import DiT
from maxtext.configs import pyconfig
from flax import nnx
import jax
from jax.sharding import Mesh
import sys
import numpy as np

# Setup dummy config
config = pyconfig.initialize(["src/maxtext/configs/base.yml", "model_name=dit-xl-2-256", "tokenizer_path=", "tokenizer_type=huggingface"])

devices_array = np.array(jax.devices())
mesh = Mesh(devices_array, ['data'])

model = DiT(config, mesh, rngs=nnx.Rngs(0))

# Load parameters
checkpoint_path = sys.argv[1]
_, params, _ = nnx.split(model, nnx.Param, ...)

print("Loading parameters to check shapes...")
checkpointing.load_params_from_path(
    checkpoint_path,
    params,
    config.checkpoint_storage_concurrent_gb,
    use_ocdbt=config.checkpoint_storage_use_ocdbt,
    use_zarr3=config.checkpoint_storage_use_zarr3,
)

print("\nVAE Decoder conv_in kernel shape:", model.vae_decoder.conv_in.kernel.value.shape)
print("VAE Decoder conv_out kernel shape:", model.vae_decoder.conv_out.kernel.value.shape)
print("VAE Decoder post_quant_conv kernel shape:", model.vae_decoder.post_quant_conv.kernel.value.shape)
