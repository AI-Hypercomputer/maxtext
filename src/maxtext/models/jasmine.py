import math
from typing import Dict, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import einops
from flax import nnx
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import DenseGeneral
from maxtext.common.common_types import AttentionType, MODEL_MODE_TRAIN

def _get_spatiotemporal_positional_encoding(d_model: int, max_len: int = 5000):
    pe = jnp.zeros((max_len, d_model))
    position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

    def _encode(x: jax.Array) -> jax.Array:
        assert x.ndim == 4, f"Input must be 4-dimensional, but got shape {x.shape}"
        num_timesteps = x.shape[1]
        num_spatial_patches = x.shape[2]
        temporal_pe = pe[jnp.newaxis, :num_timesteps, jnp.newaxis, :]
        x = x + temporal_pe
        spatial_pe = pe[jnp.newaxis, jnp.newaxis, :num_spatial_patches, :]
        x = x + spatial_pe
        return x

    return _encode

class AxialBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        use_flash_attention: bool,
        spatial_causal: bool,
        temporal_causal: bool,
        config: Any,
        mesh: Any,
        num_spatial_patches: int,
        temporal_seq_len: int,
        decode: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_flash_attention = use_flash_attention
        self.spatial_causal = spatial_causal
        self.temporal_causal = temporal_causal
        self.decode = decode
        self.config = config
        self.mesh = mesh

        self.spatial_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            rngs=rngs,
        )
        
        head_dim = self.dim // self.num_heads
        self.spatial_attention = Attention(
            config=self.config,
            num_query_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=head_dim,
            max_target_length=num_spatial_patches, # N
            mesh=self.mesh,
            attention_kernel="dot_product",
            inputs_q_shape=(1, 1, self.dim),
            inputs_kv_shape=(1, 1, self.dim),
            dtype=self.dtype,
            weight_dtype=self.param_dtype,
            is_nope_layer=True,  # Disable RoPE
            use_bias_in_projections=True,
            attention_type=AttentionType.FULL,  # Non-causal
            query_pre_attn_scalar=head_dim ** -0.5,
            rngs=rngs,
        )

        self.temporal_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            rngs=rngs,
        )
        
        self.temporal_attention = Attention(
            config=self.config,
            num_query_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=head_dim,
            max_target_length=temporal_seq_len, # T
            mesh=self.mesh,
            attention_kernel="dot_product",
            inputs_q_shape=(1, 1, self.dim),
            inputs_kv_shape=(1, 1, self.dim),
            dtype=self.dtype,
            weight_dtype=self.param_dtype,
            is_nope_layer=True,
            use_bias_in_projections=True,
            attention_type=AttentionType.GLOBAL,  # Causal
            query_pre_attn_scalar=head_dim ** -0.5,
            rngs=rngs,
        )

        self.ffn_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            rngs=rngs,
        )
        
        self.ffn_dense1 = DenseGeneral(
            in_features_shape=self.dim,
            out_features_shape=self.ffn_dim,
            dtype=self.dtype,
            weight_dtype=self.param_dtype,
            kernel_axes=("embed", "mlp"),
            use_bias=True,
            rngs=rngs,
        )
        self.ffn_dense2 = DenseGeneral(
            in_features_shape=self.ffn_dim,
            out_features_shape=self.dim,
            dtype=self.dtype,
            weight_dtype=self.param_dtype,
            kernel_axes=("mlp", "embed"),
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x_BTNM: jax.Array, deterministic: bool = True) -> jax.Array:
        # --- Spatial attention ---
        z_BTNM = self.spatial_norm(x_BTNM)
        B, T, N, M = z_BTNM.shape
        z_flat_spatial = z_BTNM.reshape(B * T, N, M)
        
        z_flat_spatial, _ = self.spatial_attention(
            z_flat_spatial,
            z_flat_spatial,
            model_mode=MODEL_MODE_TRAIN,
        )
        z_BTNM = z_flat_spatial.reshape(B, T, N, M)
        x_BTNM = x_BTNM + z_BTNM

        # --- Temporal attention ---
        x_BNTM = x_BTNM.swapaxes(1, 2)
        z_BNTM = self.temporal_norm(x_BNTM)
        B, N, T, M = z_BNTM.shape
        z_flat_temporal = z_BNTM.reshape(B * N, T, M)
        
        z_flat_temporal, _ = self.temporal_attention(
            z_flat_temporal,
            z_flat_temporal,
            model_mode=MODEL_MODE_TRAIN,
        )
        z_BNTM = z_flat_temporal.reshape(B, N, T, M)
        x_BNTM = x_BNTM + z_BNTM
        x_BTNM = x_BNTM.swapaxes(1, 2)

        # --- Feedforward ---
        z_BTNM = self.ffn_norm(x_BTNM)
        z_BTND = self.ffn_dense1(z_BTNM)
        z_BTND = jax.nn.gelu(z_BTND)
        z_BTNM = self.ffn_dense2(z_BTND)
        x_BTNM = x_BTNM + z_BTNM
        return x_BTNM

class AxialTransformer(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        ffn_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        use_flash_attention: bool,
        spatial_causal: bool,
        temporal_causal: bool,
        config: Any,
        mesh: Any,
        num_spatial_patches: int,
        temporal_seq_len: int,
        decode: bool = False,
        max_len: int = 5000,
        rngs: nnx.Rngs = None,
    ):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_flash_attention = use_flash_attention
        self.spatial_causal = spatial_causal
        self.temporal_causal = temporal_causal
        self.decode = decode
        self.config = config
        self.mesh = mesh

        self.input_norm1 = nnx.LayerNorm(
            num_features=self.input_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            rngs=rngs,
        )
        self.input_dense = DenseGeneral(
            in_features_shape=self.input_dim,
            out_features_shape=self.model_dim,
            dtype=self.dtype,
            weight_dtype=self.param_dtype,
            kernel_axes=("embed", "embed"),
            use_bias=True,
            rngs=rngs,
        )
        self.input_norm2 = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            rngs=rngs,
        )
        
        self.pos_enc = _get_spatiotemporal_positional_encoding(self.model_dim, max_len=max_len)
        
        self.blocks = nnx.List([
            AxialBlock(
                dim=self.model_dim,
                ffn_dim=self.ffn_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_flash_attention=self.use_flash_attention,
                spatial_causal=self.spatial_causal,
                temporal_causal=self.temporal_causal,
                config=self.config,
                mesh=self.mesh,
                num_spatial_patches=num_spatial_patches,
                temporal_seq_len=temporal_seq_len,
                decode=self.decode,
                rngs=rngs,
            )
            for _ in range(self.num_blocks)
        ])

        self.output_dense = DenseGeneral(
            in_features_shape=self.model_dim,
            out_features_shape=self.out_dim,
            dtype=self.dtype,
            weight_dtype=self.param_dtype,
            kernel_axes=("embed", "vocab"),
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x_BTNI: jax.Array, deterministic: bool = True) -> jax.Array:
        x_BTNI = self.input_norm1(x_BTNI)
        x_BTNM = self.input_dense(x_BTNI)
        x_BTNM = self.input_norm2(x_BTNM)
        x_BTNM = self.pos_enc(x_BTNM)
        
        for block in self.blocks:
            x_BTNM = block(x_BTNM, deterministic=deterministic)

        x_BTNV = self.output_dense(x_BTNM)
        return x_BTNV

class DynamicsMaskGIT(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_latents: int,
        latent_action_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        mask_limit: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        use_flash_attention: bool,
        config: Any,
        mesh: Any,
        num_spatial_patches: int,
        temporal_seq_len: int,
        decode: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.num_latents = num_latents
        self.latent_action_dim = latent_action_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask_limit = mask_limit
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_flash_attention = use_flash_attention
        self.decode = decode
        self.config = config
        self.mesh = mesh

        self.patch_embed = nnx.Embed(
            num_embeddings=self.num_latents,
            features=self.model_dim,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
        self.mask_token = nnx.Param(
            jax.random.uniform(rngs.params(), (1, 1, 1, self.model_dim)) * 0.02
        )
        self.action_up = DenseGeneral(
            in_features_shape=self.latent_action_dim,
            out_features_shape=self.model_dim,
            dtype=self.dtype,
            weight_dtype=self.param_dtype,
            kernel_axes=("mlp", "embed"),
            use_bias=True,
            rngs=rngs,
        )
        self.transformer = AxialTransformer(
            input_dim=self.model_dim,
            model_dim=self.model_dim,
            ffn_dim=self.ffn_dim,
            out_dim=self.num_latents,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
            config=self.config,
            mesh=self.mesh,
            num_spatial_patches=num_spatial_patches,
            temporal_seq_len=temporal_seq_len,
            decode=self.decode,
            rngs=rngs,
        )

    def __call__(
        self,
        video_tokens_BTN: jax.Array,
        latent_actions_BTm11L: jax.Array,
        mask_rng: jax.Array = None,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        vid_embed_BTNM = self.patch_embed(video_tokens_BTN)
        batch_size = vid_embed_BTNM.shape[0]
        if mask_rng is not None:
            _rng_prob, *_rngs_mask = jax.random.split(mask_rng, batch_size + 1)
            mask_prob = jax.random.uniform(
                _rng_prob, shape=(batch_size,), minval=self.mask_limit
            )
            per_sample_shape = vid_embed_BTNM.shape[1:-1]
            mask = jax.vmap(
                lambda rng, prob: jax.random.bernoulli(rng, prob, per_sample_shape),
                in_axes=(0, 0),
            )(jnp.asarray(_rngs_mask), mask_prob)
            mask = mask.at[:, 0].set(False)
            vid_embed_BTNM = jnp.where(
                jnp.expand_dims(mask, -1), self.mask_token[...], vid_embed_BTNM
            )
        else:
            mask = jnp.zeros(video_tokens_BTN.shape, dtype=jnp.bool_)

        act_embed_BTm11M = self.action_up(latent_actions_BTm11L)
        padded_act_embed_BT1M = jnp.pad(
            act_embed_BTm11M, ((0, 0), (1, 0), (0, 0), (0, 0))
        )
        vid_embed_BTNp1M = jnp.concatenate(
            [padded_act_embed_BT1M, vid_embed_BTNM], axis=2
        )
        logits_BTNp1V = self.transformer(vid_embed_BTNp1M, deterministic=deterministic)
        logits_BTNV = logits_BTNp1V[:, :, 1:]
        return logits_BTNV, mask

    def sample(
        self,
        token_idxs_BTN: jax.Array,
        action_tokens_EL: jax.Array,
        seq_len: int,
        steps: int = 25,
        temperature: float = 1.0,
        sample_argmax: bool = False,
        rng: jax.Array = None,
    ) -> Tuple[jax.Array, jax.Array]:
        B, T, N = token_idxs_BTN.shape
        pad_shape = (B, seq_len - T, N)
        pad = jnp.zeros(pad_shape, dtype=token_idxs_BTN.dtype)
        token_idxs_BSN = jnp.concatenate([token_idxs_BTN, pad], axis=1)
        init_logits_BSNV = jnp.zeros(shape=(*token_idxs_BSN.shape, self.num_latents))
        
        def maskgit_step_fn(carry, step):
            rng, token_idxs_BSN, logits_BSNV, mask_BSN = carry
            S, N = token_idxs_BSN.shape[1:]
            L = action_tokens_EL.shape[-1]
            
            vid_embed_BSNM = self.patch_embed(token_idxs_BSN)
            mask_expanded_BSN1 = mask_BSN[..., jnp.newaxis]
            vid_embed_BSNM = jnp.where(
                mask_expanded_BSN1, self.mask_token[...], vid_embed_BSNM
            )
            
            action_tokens_BSm1L = jnp.reshape(action_tokens_EL, (B, S - 1, L))
            act_embed_BSm1M = self.action_up(action_tokens_BSm1L)
            act_embed_BSM = jnp.pad(act_embed_BSm1M, ((0, 0), (1, 0), (0, 0)))
            act_embed_BS1M = jnp.reshape(
                act_embed_BSM, (B, S, 1, act_embed_BSM.shape[-1])
            )
            vid_embed_BSNp1M = jnp.concatenate([act_embed_BS1M, vid_embed_BSNM], axis=2)
            
            unmasked_ratio = jnp.cos(jnp.pi * (step + 1) / (steps * 2))
            step_temp = temperature * (1.0 - unmasked_ratio)
            final_logits_BSNp1V = (
                self.transformer(vid_embed_BSNp1M, deterministic=True) / step_temp
            )
            final_logits_BSNV = final_logits_BSNp1V[:, :, 1:]
            
            if sample_argmax:
                sampled_token_idxs_BSN = jnp.argmax(final_logits_BSNV, axis=-1)
            else:
                rng, _rng = jax.random.split(rng)
                sampled_token_idxs_BSN = jax.random.categorical(_rng, final_logits_BSNV)
                
            gather_fn = jax.vmap(jax.vmap(jax.vmap(lambda x, y: x[y])))
            final_token_probs_BSN = gather_fn(
                jax.nn.softmax(final_logits_BSNV), sampled_token_idxs_BSN
            )
            final_token_probs_BSN += ~mask_BSN
            
            token_idxs_BSN = jnp.where(mask_BSN, sampled_token_idxs_BSN, token_idxs_BSN)
            logits_BSNV = jnp.where(
                jnp.expand_dims(mask_BSN, -1), final_logits_BSNV, logits_BSNV
            )
            
            num_unmasked_tokens = jnp.round(N * (1.0 - unmasked_ratio)).astype(int)
            final_token_probs_flat_BP = einops.rearrange(
                final_token_probs_BSN, "b s n -> b (s n)"
            )
            idx_mask_P = (
                jnp.arange(final_token_probs_flat_BP.shape[-1])
                <= N - num_unmasked_tokens
            )
            sorted_idxs_BP = jnp.argsort(final_token_probs_flat_BP, axis=-1)
            mask_update_fn = jax.vmap(lambda msk, ids: msk.at[ids].set(idx_mask_P))
            mask_flat_BP = einops.rearrange(mask_BSN, "b s n -> b (s n)")
            new_mask_flat_BP = mask_update_fn(mask_flat_BP, sorted_idxs_BP)
            new_mask_BSN = einops.rearrange(new_mask_flat_BP, "b (s n) -> b s n", n=N)
            
            return (rng, token_idxs_BSN, logits_BSNV, new_mask_BSN), None
 
        def generation_step_fn(carry, step_t):
            rng, current_token_idxs_BSN, current_logits_BSNV = carry
            rng, step_rng = jax.random.split(rng)
            
            mask_S = jnp.arange(seq_len) == step_t
            mask_BSN = jnp.broadcast_to(
                mask_S[jnp.newaxis, :, jnp.newaxis], (B, seq_len, N)
            ).astype(bool)
            masked_token_idxs_BSN = current_token_idxs_BSN * ~mask_BSN
            masked_logits_BSNV = current_logits_BSNV * jnp.expand_dims(~mask_BSN, -1)
            
            init_carry_maskgit = (
                step_rng,
                masked_token_idxs_BSN,
                masked_logits_BSNV,
                mask_BSN,
            )
            (final_rng, updated_token_idxs_BSN, updated_logits_BSNV, _) , _ = jax.lax.scan(
                maskgit_step_fn,
                init_carry_maskgit,
                jnp.arange(steps)
            )
            
            return (rng, updated_token_idxs_BSN, updated_logits_BSNV), None
 
        initial_carry = (rng, token_idxs_BSN, init_logits_BSNV)
        timesteps_to_scan = jnp.arange(T, seq_len)
        (final_rng, final_token_idxs_BSN, final_logits_BSNV), _ = jax.lax.scan(
            generation_step_fn,
            initial_carry,
            timesteps_to_scan
        )
        return final_token_idxs_BSN, final_logits_BSNV
