import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from maxtext.nnx_exp.models import Llama, LlamaConfig
from maxtext.nnx_exp.sharding import LlamaSharding, create_mesh
from maxtext.nnx_exp.infra import (
    apply_remat,
    maybe_apply_remat,
    to_host,
    to_device,
    create_scanned_layers,
    create_scanned_remat_layers,
    scan_forward,
)

# Mock config for tests
CONFIG = LlamaConfig(
    vocab_size=1000,
    emb_dim=256,
    num_heads=4,
    num_kv_heads=2,
    num_layers=2,
    mlp_dim=512,
    head_dim=64,
)

def test_remat():
    mesh = create_mesh({"ici_dp_parallelism": 1, "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 1})
    with jax.set_mesh(mesh):
        sharding = LlamaSharding()
        rngs = nnx.Rngs(42)
        model = Llama(CONFIG, rngs=rngs, sharding=sharding)
        
        # Apply full remat
        apply_remat(model, "full")
        
        tokens = jnp.zeros((2, 32), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(32), (2, 32))
        
        # Forward pass
        out = model(tokens, positions)
        assert out.shape == (2, 32, CONFIG.vocab_size)

def test_remat_selective():
    mesh = create_mesh({"ici_dp_parallelism": 1, "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 1})
    with jax.set_mesh(mesh):
        sharding = LlamaSharding()
        rngs = nnx.Rngs(42)
        model = Llama(CONFIG, rngs=rngs, sharding=sharding)
        
        # Apply selective remat
        apply_remat(model, {"save": ["query_proj", "key_proj", "value_proj", "attention_out", "mlpwi"]})
        
        tokens = jnp.zeros((2, 32), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(32), (2, 32))
        
        out = model(tokens, positions)
        assert out.shape == (2, 32, CONFIG.vocab_size)

def test_offload():
    mesh = create_mesh({"ici_dp_parallelism": 1, "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 1})
    with jax.set_mesh(mesh):
        sharding = LlamaSharding()
        rngs = nnx.Rngs(42)
        model = Llama(CONFIG, rngs=rngs, sharding=sharding)
        
        gdef, params = nnx.split(model, nnx.Param)
        
        # Offload to host
        params_host = to_host(params)
        
        # Move back to device
        params_device = to_device(params_host)
        
        # Reconstruct model
        model = nnx.merge(gdef, params_device)
        
        tokens = jnp.zeros((2, 32), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(32), (2, 32))
        
        out = model(tokens, positions)
        assert out.shape == (2, 32, CONFIG.vocab_size)

def test_scan():
    mesh = create_mesh({"ici_dp_parallelism": 1, "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 1})
    with jax.set_mesh(mesh):
        sharding = LlamaSharding()
        rngs = nnx.Rngs(42)
        
        from maxtext.nnx_exp.models import DecoderLayer
        
        # Create scanned layers
        blocks = create_scanned_layers(DecoderLayer, CONFIG, CONFIG.num_layers, rngs=rngs, sharding=sharding)
        
        # Mock inputs
        x = jax.device_put(jnp.zeros((2, 32, CONFIG.emb_dim)), sharding.sequence_spec())
        positions = jnp.broadcast_to(jnp.arange(32), (2, 32))
        mask = None
        
        # Forward pass via scan
        out = scan_forward(x, blocks, positions, mask)
        assert out.shape == (2, 32, CONFIG.emb_dim)

def test_scan_remat():
    mesh = create_mesh({"ici_dp_parallelism": 1, "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 1})
    with jax.set_mesh(mesh):
        sharding = LlamaSharding()
        rngs = nnx.Rngs(42)
        
        from maxtext.nnx_exp.models import DecoderLayer
        
        # Create scanned remat layers
        blocks = create_scanned_remat_layers(DecoderLayer, CONFIG, CONFIG.num_layers, rngs=rngs, policy="full", sharding=sharding)
        
        x = jax.device_put(jnp.zeros((2, 32, CONFIG.emb_dim)), sharding.sequence_spec())
        positions = jnp.broadcast_to(jnp.arange(32), (2, 32))
        mask = None
        
        out = scan_forward(x, blocks, positions, mask)
        assert out.shape == (2, 32, CONFIG.emb_dim)

def test_quantization():
    mesh = create_mesh({"ici_dp_parallelism": 1, "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 1})
    with jax.set_mesh(mesh):
        sharding = LlamaSharding()
        rngs = nnx.Rngs(42)
        model = Llama(CONFIG, rngs=rngs, sharding=sharding)
        
        tokens = jnp.zeros((2, 32), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(32), (2, 32))
        
        from maxtext.nnx_exp.infra import quantize_model, int8_rules
        
        # Quantize model
        quantized_model = quantize_model(model, int8_rules(), tokens, positions)
        
        out = quantized_model(tokens, positions)
        assert out.shape == (2, 32, CONFIG.vocab_size)

