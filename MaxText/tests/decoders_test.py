# Copyright 2023–2025 Google LLC
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

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import os
import pytest

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.globals import PKG_DIR
from MaxText.common_types import Config, MODEL_MODE_TRAIN
from MaxText.layers import embeddings
from MaxText.layers.decoders import Decoder

class ModelHarness:
    """A wrapper providing a unified interface for testing Linen and NNX models."""
    def __init__(self, model, variables_or_state, framework):
        self._model = model
        self._variables_or_state = variables_or_state
        self._framework = framework

    def apply(self, *args, **kwargs):
        """Executes the forward pass for either Linen or NNX."""
        if self._framework == 'linen':
            rngs = {'dropout': jax.random.PRNGKey(1)} if 'deterministic' in kwargs and not kwargs['deterministic'] else None
            return self._model.apply(self._variables_or_state, *args, rngs=rngs, **kwargs)
        elif self._framework == 'nnx':
            return self._model(*args, **kwargs)
        raise TypeError(f"Unsupported model type: {type(self._model)}")

def get_mesh(config: Config) -> Mesh:
    """Provides a JAX device mesh for sharding."""
    devices_array = maxtext_utils.create_device_mesh(config)
    return Mesh(devices_array, config.mesh_axes)

def get_config_with_overrides(**overrides):
    argv = [None, os.path.join(PKG_DIR, "configs", "base.yml")]
    init_kwargs = {
        'run_name': 'test',
        'skip_jax_distributed_system': True,
    } | overrides 

    return pyconfig.initialize(
        argv,
        **init_kwargs
    )

@pytest.fixture(scope="module")
def detected_framework():
    """
    Inspects the imported Decoder class and returns its framework type.
    This runs only once per test session.
    """
    if issubclass(Decoder, nnx.Module):
        return 'nnx'
    # Check for Linen last as NNX modules might have nn.Module in their MRO
    if issubclass(Decoder, nn.Module):
        return 'linen'
    raise TypeError(
        "Imported 'Decoder' is not a recognized subclass of flax.linen.Module or flax.nnx.Module"
    )

@pytest.fixture
def harness_factory(detected_framework):
    """
    Returns a factory function that can create a model harness for the
    automatically detected framework.
    """
    def _create_harness(config, mesh):
        framework = detected_framework # The factory "remembers" the framework
        key = jax.random.PRNGKey(0)
        batch_size, seq_len = int(config.per_device_batch_size), 16
        decoder_input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        decoder_positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, -1)

        if framework == 'linen':
            shared_embedding = embeddings.embed_as_linen(
                num_embeddings=config.vocab_size,
                num_features=config.emb_dim,
                dtype=config.dtype,
                attend_dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,  # for logit training stability
                embedding_init=nn.initializers.normal(stddev=1.0),
                name="token_embedder",
                config=config,
            )
            model = Decoder(config=config, shared_embedding=shared_embedding, mesh=mesh)
            variables = model.init(
                {'params': key, 'dropout': key},
                decoder_input_tokens=decoder_input_tokens,
                decoder_positions=decoder_positions,
                deterministic=True,
                model_mode=MODEL_MODE_TRAIN,
            )
            return ModelHarness(model=model, variables_or_state=variables, framework='linen')

        elif framework == 'nnx':
            rngs = nnx.Rngs(params=0, dropout=1)
            shared_embedding = embeddings.Embed(
                num_embeddings=config.vocab_size,
                num_features=config.emb_dim,
                dtype=config.dtype,
                attend_dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,  # for logit training stability
                embedding_init=nn.initializers.normal(stddev=1.0),
                config=config,
                rngs=rngs,
            )
            model = Decoder(config=config, shared_embedding=shared_embedding, mesh=mesh, rngs=rngs)
            return ModelHarness(model=model, variables_or_state=model, framework='nnx')

    return _create_harness


@pytest.fixture
def base_config():
    """Provides a default, immutable config for tests."""
    return get_config_with_overrides(
        dropout_rate=0.5,
        enable_dropout=True,
        per_device_batch_size=4,
        ici_tensor_parallelism=1,
        scan_layers=False
    )

class TestUnifiedDecoder:
    """A single test class for both Linen and NNX Decoder implementations."""

    def test_forward_pass_shape(self, harness_factory, base_config):
        """Tests the forward pass shape and dtype."""
        mesh = get_mesh(config=base_config)
        harness = harness_factory(base_config, mesh)
        batch_size, seq_len = int(base_config.per_device_batch_size), 16
        decoder_input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        decoder_positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, -1)

        logits, hidden_state = harness.apply(
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            deterministic=True,
            model_mode=MODEL_MODE_TRAIN,
        )

        assert logits.shape == (batch_size, seq_len, base_config.vocab_size)
        assert hidden_state.shape == (batch_size, seq_len, base_config.emb_dim)
        assert hidden_state.dtype == base_config.dtype

    @pytest.mark.parametrize('use_untrainable', [True, False])
    @pytest.mark.parametrize('trainable_size', [0, 32])
    def test_embedding_logic(self, harness_factory, use_untrainable, trainable_size):
        """Tests that enabling positional embeddings changes the output."""
        config_base = get_config_with_overrides(
            use_untrainable_positional_embedding=False, trainable_position_size=0
        )
        harness_base = harness_factory(config_base, get_mesh(config_base))

        config_custom = get_config_with_overrides(
            use_untrainable_positional_embedding=use_untrainable, trainable_position_size=trainable_size
        )
        harness_custom = harness_factory(config_custom, get_mesh(config_custom))

        batch_size, seq_len = int(config_base.per_device_batch_size), 16
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, -1)
        apply_kwargs = dict(
            decoder_input_tokens=tokens, decoder_positions=positions,
            deterministic=True, model_mode=MODEL_MODE_TRAIN
        )

        _, hidden_state_base = harness_base.apply(**apply_kwargs)
        _, hidden_state_custom = harness_custom.apply(**apply_kwargs)

        if not use_untrainable and trainable_size == 0:
            np.testing.assert_allclose(hidden_state_base, hidden_state_custom, atol=1e-6)
        else:
            assert not np.allclose(hidden_state_base, hidden_state_custom)

    @pytest.mark.parametrize('logits_via_embedding', [True, False])
    def test_output_head_logic(self, harness_factory, logits_via_embedding):
        """Tests switching between tied and separate output logits layer."""
        config = get_config_with_overrides(logits_via_embedding=logits_via_embedding)
        harness = harness_factory(config, get_mesh(config))
        
        batch_size, seq_len = int(config.per_device_batch_size), 16
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, -1)
        
        logits, _ = harness.apply(
            decoder_input_tokens=tokens, decoder_positions=positions,
            deterministic=True, model_mode=MODEL_MODE_TRAIN
        )
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_deterministic_mode_for_dropout(self, harness_factory, base_config):
        """Ensures dropout is active only when deterministic is False."""
        config = base_config
        mesh = get_mesh(config=config)
        
        batch_size, seq_len = int(config.per_device_batch_size), 16
        tokens = jax.random.randint(jax.random.PRNGKey(10), (batch_size, seq_len), 0, config.vocab_size)
        positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, -1)

        harness1 = harness_factory(config, mesh)
        logits_det1, _ = harness1.apply(
            decoder_input_tokens=tokens, decoder_positions=positions,
            deterministic=True, model_mode=MODEL_MODE_TRAIN
        )

        harness2 = harness_factory(config, mesh)
        logits_det2, _ = harness2.apply(
            decoder_input_tokens=tokens, decoder_positions=positions,
            deterministic=True, model_mode=MODEL_MODE_TRAIN
        )
        np.testing.assert_allclose(logits_det1, logits_det2, atol=1e-7, rtol=1e-7)

        harness3 = harness_factory(config, mesh)
        logits_nondet, _ = harness3.apply(
            decoder_input_tokens=tokens, decoder_positions=positions,
            deterministic=False, model_mode=MODEL_MODE_TRAIN
        )
        assert not np.allclose(logits_det1, logits_nondet)

    @pytest.mark.parametrize('scan_layers', [True, False])
    def test_scan_layers_flag(self, harness_factory, scan_layers):
        """Tests that the model works with and without layer scanning."""
        config = get_config_with_overrides(scan_layers=scan_layers)
        harness = harness_factory(config, get_mesh(config))
        
        batch_size, seq_len = int(config.per_device_batch_size), 16
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, -1)
        
        logits, hidden_state = harness.apply(
            decoder_input_tokens=tokens, decoder_positions=positions,
            deterministic=True, model_mode=MODEL_MODE_TRAIN
        )
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert hidden_state.shape == (batch_size, seq_len, config.emb_dim)

    def test_input_shape_validation(self, harness_factory, base_config):
        """Tests that the model raises AssertionError for incorrect input dimensions."""
        mesh = get_mesh(config=base_config)
        harness = harness_factory(base_config, mesh)
        
        batch_size, seq_len = int(base_config.per_device_batch_size), 16
        bad_tokens = jnp.ones((batch_size, seq_len, 1), dtype=jnp.int32)
        positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, -1)
        
        with pytest.raises((AssertionError,ValueError)):
            harness.apply(
                decoder_input_tokens=bad_tokens,
                decoder_positions=positions,
                deterministic=True,
                model_mode=MODEL_MODE_TRAIN
            )