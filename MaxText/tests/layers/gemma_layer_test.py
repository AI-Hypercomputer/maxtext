import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import Module
from MaxText.layers.gemma import GemmaDecoderLayer
from MaxText.pyconfig import initialize as pyconfig_initialize, string_to_bool
from jax.sharding import Mesh
from MaxText import common_types

class TestGemmaDecoderLayer(unittest.TestCase):
    def setUp(self):
        print("Attempting setUp for TestGemmaDecoderLayer...")
        argv = ['test_gemma_layer.py', 'MaxText/configs/base.yml']

        config_kwargs = {
            'run_name': 'gemma_layer_test_run',
            'model_name': 'default',
            'override_model_config': "False",
            'log_config': "False",
            'tokenizer_path': 'tokenizer_llama3.tiktoken',
            'tokenizer_type': 'tiktoken',
            'dataset_type': 'synthetic',
            'dataset_path': '',
            'enable_checkpointing': "False",
            'per_device_batch_size': 1.0,
            'steps': 10,
            'global_parameter_scale': 1,
            'learning_rate_schedule_steps': -1,
            'warmup_steps_fraction': 0.1,
            'mu_dtype': "float32",
            'ici_fsdp_parallelism': 1, 'ici_tensor_parallelism': 1, 'ici_pipeline_parallelism': 1, 'ici_sequence_parallelism':1,
            'dcn_fsdp_parallelism': 1, 'dcn_tensor_parallelism': 1, 'dcn_pipeline_parallelism': 1, 'dcn_sequence_parallelism':1,
            'max_target_length': 64, # Corrected: Ensure this is passed for pyconfig validation if needed
            'max_prefill_predict_length': 32,
        }

        print(f"Calling pyconfig_initialize with argv: {argv}, kwargs: {config_kwargs}")
        self.config = pyconfig_initialize(argv=argv, **config_kwargs)
        print("pyconfig_initialize call completed.")

        if hasattr(self.config, '_config') and hasattr(self.config._config, 'keys') and isinstance(self.config._config.keys, dict):
            config_dict_to_modify = self.config._config.keys
            print("Accessed self.config._config.keys")
        elif isinstance(self.config.keys, dict):
            config_dict_to_modify = self.config.keys
            print("Accessed self.config.keys")
        else:
            print("Could not find suitable config dictionary to modify. Will attempt direct attribute setting on self.config.")
            config_dict_to_modify = self.config

        print("Manually setting/overriding config parameters...")

        config_dict_to_modify["enable_checkpointing"] = False
        config_dict_to_modify.setdefault("async_checkpointing", False)
        config_dict_to_modify.setdefault("enable_profiler", False)
        config_dict_to_modify.setdefault("log_host_info", False)
        config_dict_to_modify["override_model_config"] = False
        config_dict_to_modify["log_config"] = False

        config_dict_to_modify.setdefault('coordinator_address', None)
        config_dict_to_modify.setdefault('num_processes', 1)
        config_dict_to_modify.setdefault('process_id', 0)
        config_dict_to_modify.setdefault('jax_distributed_initialization_timeout',300)

        config_dict_to_modify['dtype'] = jnp.float32
        config_dict_to_modify['weight_dtype'] = jnp.float32
        config_dict_to_modify.setdefault("mu_dtype", jnp.float32)

        config_dict_to_modify['num_query_heads'] = 2
        config_dict_to_modify['num_kv_heads'] = 2
        config_dict_to_modify['head_dim'] = 4
        config_dict_to_modify['embed_dim'] = config_dict_to_modify['num_query_heads'] * config_dict_to_modify['head_dim']
        config_dict_to_modify['mlp_dim'] = 16
        config_dict_to_modify['num_decoder_layers'] = 1

        config_dict_to_modify['base_num_query_heads'] = config_dict_to_modify['num_query_heads']
        config_dict_to_modify['base_num_kv_heads'] = config_dict_to_modify['num_kv_heads']
        config_dict_to_modify['base_emb_dim'] = config_dict_to_modify['embed_dim']
        config_dict_to_modify['base_mlp_dim'] = config_dict_to_modify['mlp_dim']
        config_dict_to_modify['base_num_decoder_layers'] = config_dict_to_modify['num_decoder_layers']
        config_dict_to_modify.setdefault("global_parameter_scale", 1)

        config_dict_to_modify['dropout_rate'] = 0.0
        config_dict_to_modify.setdefault('attention_type', "dot_product")
        config_dict_to_modify.setdefault('attention', "dot_product")
        config_dict_to_modify['mlp_activations'] = ("gelu", "linear")
        config_dict_to_modify['max_prefill_predict_length'] = 32 # Corrected Indentation
        config_dict_to_modify['max_target_length'] = 64    # Corrected Indentation & Value
        config_dict_to_modify['scan_layers'] = False
        config_dict_to_modify['record_internal_nn_metrics'] = False
        config_dict_to_modify['use_bias'] = True
        config_dict_to_modify.setdefault('activation_function', "gelu")
        config_dict_to_modify['decoder_block'] = common_types.DecoderBlockType.GEMMA

        config_dict_to_modify['quant'] = None
        config_dict_to_modify['kv_quant'] = None
        config_dict_to_modify['float32_qk_product'] = True
        config_dict_to_modify['float32_logits'] = True
        config_dict_to_modify['use_ragged_attention'] = False
        config_dict_to_modify['ragged_block_size'] = 0
        config_dict_to_modify.setdefault("attention_kernel", "autoselected")
        config_dict_to_modify.setdefault("attention_num_segments", 1)
        config_dict_to_modify.setdefault("rope_type", "default")

        config_dict_to_modify.setdefault("use_unpadded_segment_ids", False)
        config_dict_to_modify.setdefault("reset_transformer_max_length_for_eval", False)
        config_dict_to_modify.setdefault("max_reuse_decoding_length", -1)
        config_dict_to_modify.setdefault("moe_load_balancing_loss_weight", 0.01)
        config_dict_to_modify.setdefault("moe_router_aux_loss_factor", 0.01)
        config_dict_to_modify.setdefault("logical_axis_rules", [('activation_embed_and_logits_batch', ('data', 'fsdp'))])
        config_dict_to_modify.setdefault("data_sharding", [('data',)])
        config_dict_to_modify.setdefault("mlp_dim_scale_factor", 2.0)
        config_dict_to_modify.setdefault("per_device_batch_size", 1.0)
        config_dict_to_modify.setdefault("steps", 10)
        config_dict_to_modify.setdefault("learning_rate_schedule_steps", config_dict_to_modify["steps"])
        config_dict_to_modify.setdefault("warmup_steps_fraction", 0.1)
        config_dict_to_modify.setdefault("metrics_file", None)
        config_dict_to_modify.setdefault("command_line_str", "")
        config_dict_to_modify.setdefault("use_wandb", "False")
        config_dict_to_modify.setdefault("hf_access_token", None)
        config_dict_to_modify.setdefault("grain_train_files", "")
        config_dict_to_modify.setdefault("grain_worker_count", 0)
        config_dict_to_modify.setdefault("eval_interval", 0)
        config_dict_to_modify.setdefault("dataset_name", "")
        config_dict_to_modify.setdefault("eval_split", "")
        config_dict_to_modify.setdefault("hf_train_files", None)
        config_dict_to_modify.setdefault("hf_eval_files", None)
        config_dict_to_modify.setdefault("hf_eval_split", "")
        config_dict_to_modify.setdefault("hf_data_dir", "")
        config_dict_to_modify.setdefault("hf_path", "")
        config_dict_to_modify.setdefault("num_epoch",1)
        config_dict_to_modify.setdefault("c4_mlperf_path","")
        config_dict_to_modify.setdefault("remat_policy","none")
        config_dict_to_modify.setdefault("compute_axis_order", "0,1,2,3")
        config_dict_to_modify.setdefault("kv_quant_axis", "")
        config_dict_to_modify.setdefault("quantize_kvcache", False)
        config_dict_to_modify.setdefault("profiler", "")
        config_dict_to_modify.setdefault("profile_periodically_period", 0)
        config_dict_to_modify.setdefault("profiler_steps", 1)
        config_dict_to_modify.setdefault("model_call_mode", "")
        config_dict_to_modify.setdefault("load_parameters_path", "")
        config_dict_to_modify.setdefault("load_full_state_path", "")
        config_dict_to_modify.setdefault("enable_emergency_checkpoint", False)
        config_dict_to_modify.setdefault("dump_hlo_xla_flags", "")
        config_dict_to_modify.setdefault("dump_hlo_module_name", "")
        config_dict_to_modify.setdefault("dump_hlo_local_dir", "")
        config_dict_to_modify.setdefault("dump_hlo_gcs_dir", "")
        config_dict_to_modify.setdefault("pipeline_parallel_layers", -1)
        config_dict_to_modify.setdefault("num_pipeline_repeats", -1)
        config_dict_to_modify.setdefault("num_layers_per_pipeline_stage", -1)
        config_dict_to_modify.setdefault("num_pipeline_microbatches", -1)
        config_dict_to_modify.setdefault("pipeline_delay_activation_forwarding", False)
        config_dict_to_modify.setdefault("use_multimodal", False)
        config_dict_to_modify.setdefault("interleave_moe_layer_step",1)
        config_dict_to_modify.setdefault("sparse_matmul", False)
        config_dict_to_modify.setdefault("dump_hlo", False)
        config_dict_to_modify.setdefault("jax_cache_dir", "")
        config_dict_to_modify.setdefault("jax_debug_log_modules", "")
        config_dict_to_modify.setdefault("compile_topology", "")
        config_dict_to_modify.setdefault("compile_topology_num_slices", 1)
        config_dict_to_modify.setdefault("expansion_factor_real_data", -1)
        config_dict_to_modify.setdefault("eval_per_device_batch_size", 0.0)
        config_dict_to_modify.setdefault("pagedattn_max_pages_per_group", 0)
        config_dict_to_modify.setdefault("quantization_local_shard_count", -1)
        config_dict_to_modify.setdefault("checkpoint_period", 1)
        config_dict_to_modify.setdefault("attn_logits_soft_cap", 0.0)
        config_dict_to_modify.setdefault("final_logits_soft_cap", 0.0)
        config_dict_to_modify.setdefault("use_iota_embed", False)
        for key_remat in ["decoder_layer_input", "context", "mlpwi", "mlpwi_0", "mlpwi_1", "mlpwo",
                         "query_proj", "key_proj", "value_proj", "out_proj"]:
            config_dict_to_modify.setdefault(key_remat, "remat")
        config_dict_to_modify.setdefault("num_experts", 1)
        config_dict_to_modify.setdefault("num_experts_per_tok",1)
        config_dict_to_modify.setdefault("moe_capacity_factor",1.0)
        config_dict_to_modify.setdefault("moe_eval_capacity_factor",1.0)
        config_dict_to_modify.setdefault("n_shared_experts", -1)
        config_dict_to_modify.setdefault("n_gaus_experts", -1)
        config_dict_to_modify.setdefault("n_gaus_experts_per_tok", -1)
        config_dict_to_modify.setdefault("n_routing_groups", -1)
        config_dict_to_modify.setdefault("topk_routing_group", -1)
        config_dict_to_modify.setdefault("chunk_attn_window_size", 0)
        config_dict_to_modify.setdefault("sliding_window_size", 0)
        for i_dcn in ["ici", "dcn"]: # Ensure all parallelism options are defaulted
            config_dict_to_modify.setdefault(f"{i_dcn}_data_parallelism", 1)
            config_dict_to_modify.setdefault(f"{i_dcn}_fsdp_parallelism", 1)
            config_dict_to_modify.setdefault(f"{i_dcn}_fsdp_transpose_parallelism", 1)
            config_dict_to_modify.setdefault(f"{i_dcn}_context_parallelism", 1)
            config_dict_to_modify.setdefault(f"{i_dcn}_context_autoregressive_parallelism", 1)
            config_dict_to_modify.setdefault(f"{i_dcn}_tensor_parallelism", 1)
            config_dict_to_modify.setdefault(f"{i_dcn}_tensor_transpose_parallelism", 1)
            config_dict_to_modify.setdefault(f"{i_dcn}_tensor_sequence_parallelism", 1)
            config_dict_to_modify.setdefault(f"{i_dcn}_expert_parallelism", 1)
            config_dict_to_modify.setdefault(f"{i_dcn}_autoregressive_parallelism", 1)

        print("Manual config parameter setting completed.")

        self.mesh = Mesh(np.array(jax.devices()), ('data',))
        print("Mesh created.")
        print("setUp for TestGemmaDecoderLayer completed successfully.")

    def test_output_shape(self):
        print("Running test_output_shape (now with full logic)...")
        B, L = 2, 4

        current_embed_dim = self.config.embed_dim

        inputs = jnp.ones((B, L, current_embed_dim), dtype=self.config.dtype)
        decoder_segment_ids = jnp.zeros((B, L), dtype=jnp.int32)
        decoder_positions = jnp.arange(L, dtype=jnp.int32)[None, :] + jnp.zeros((B,1), dtype=jnp.int32)

        layer = GemmaDecoderLayer(config=self.config, mesh=self.mesh)
        rng = jax.random.PRNGKey(0)

        params = layer.init(rng, inputs, decoder_segment_ids, decoder_positions, deterministic=True, model_mode=common_types.MODEL_MODE_TRAIN)['params']

        output = layer.apply(
            {'params': params},
            inputs,
            decoder_segment_ids,
            decoder_positions,
            deterministic=True,
            model_mode=common_types.MODEL_MODE_TRAIN
        )

        self.assertEqual(output.shape, (B, L, current_embed_dim))
        print("test_output_shape successfully asserted output shape.")

    def test_model_modes(self):
        """Tests that the layer runs with different model_mode values."""
        print("Running test_model_modes...")
        B, L = 1, 2

        current_embed_dim = self.config.embed_dim
        inputs = jnp.ones((B, L, current_embed_dim), dtype=self.config.dtype)
        decoder_segment_ids = jnp.zeros((B, L), dtype=jnp.int32)
        decoder_positions = jnp.arange(L, dtype=jnp.int32)[None, :]

        layer = GemmaDecoderLayer(config=self.config, mesh=self.mesh)
        rng = jax.random.PRNGKey(0)

        print("Initializing layer for cache in test_model_modes...")
        init_vars = layer.init(
            rng,
            inputs,
            decoder_segment_ids,
            decoder_positions,
            deterministic=True,
            model_mode=common_types.MODEL_MODE_PREFILL,
            mutable=['params', 'cache']
        )
        params = init_vars['params']
        print("Layer initialized for cache in test_model_modes.")

        modes_to_test = [
            common_types.MODEL_MODE_TRAIN,
            common_types.MODEL_MODE_PREFILL,
            common_types.MODEL_MODE_AUTOREGRESSIVE
        ]

        for mode in modes_to_test:
            print(f"Testing model_mode: {mode}...")

            current_inputs_for_mode = inputs
            current_decoder_segment_ids_for_mode = decoder_segment_ids
            current_decoder_positions_for_mode = decoder_positions
            current_L_for_mode = L

            if mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
                print("Adjusting inputs for AUTOREGRESSIVE mode (L=1)...")
                current_L_for_mode = 1
                current_inputs_for_mode = jnp.ones((B, current_L_for_mode, current_embed_dim), dtype=self.config.dtype)
                current_decoder_segment_ids_for_mode = jnp.zeros((B, current_L_for_mode), dtype=jnp.int32)
                # For AR, positions are typically the current token's index.
                # Using a fixed position (e.g., original L-1, or 0 if L is dynamic) for shape test.
                # If the original L was 2, L-1 is 1. For a new L of 1, position should be 0.
                # Let's use a position relative to the *original* sequence length if it matters for KV cache slotting,
                # or simply 0 if it's a fresh sequence of length 1.
                # For this test, the exact position value for L=1 might not be critical beyond being valid.
                current_decoder_positions_for_mode = jnp.array([[L-1]], dtype=jnp.int32) if L > 1 else jnp.array([[0]], dtype=jnp.int32)


            apply_vars = {'params': params}
            if mode != common_types.MODEL_MODE_TRAIN:
                apply_vars['cache'] = init_vars['cache']

            print(f"Applying layer with mode: {mode}...")
            output_tuple = layer.apply(
                apply_vars,
                current_inputs_for_mode,
                current_decoder_segment_ids_for_mode,
                current_decoder_positions_for_mode,
                deterministic=True,
                model_mode=mode,
                mutable=['cache'] if mode != common_types.MODEL_MODE_TRAIN else False
            )

            if mode != common_types.MODEL_MODE_TRAIN:
                main_output = output_tuple[0]
            else:
                main_output = output_tuple

            self.assertEqual(main_output.shape, (B, current_L_for_mode, current_embed_dim), f"Output shape mismatch for mode {mode}")
        print("test_model_modes finished.")

    def test_deterministic_mode_dropout(self):
        print("Running test_deterministic_mode_dropout...")
        B, L = 2, 4
        self.config._config.keys['dropout_rate'] = 0.1

        current_embed_dim = self.config.embed_dim

        inputs = jnp.ones((B, L, current_embed_dim), dtype=self.config.dtype)
        decoder_segment_ids = jnp.zeros((B, L), dtype=jnp.int32)
        decoder_positions = jnp.arange(L, dtype=jnp.int32)[None, :] * jnp.ones((B,1), dtype=jnp.int32)

        layer = GemmaDecoderLayer(config=self.config, mesh=self.mesh)
        rng_init = jax.random.PRNGKey(0)

        params = layer.init(rng_init, inputs, decoder_segment_ids, decoder_positions, deterministic=True, model_mode=common_types.MODEL_MODE_TRAIN)['params']

        rng_dropout_1 = jax.random.PRNGKey(1)
        rng_dropout_2 = jax.random.PRNGKey(2)

        output_deterministic = layer.apply(
            {'params': params},
            inputs,
            decoder_segment_ids,
            decoder_positions,
            deterministic=True,
            model_mode=common_types.MODEL_MODE_TRAIN,
            rngs={'dropout': rng_dropout_1}
        )

        output_nondeterministic_rng1 = layer.apply(
            {'params': params},
            inputs,
            decoder_segment_ids,
            decoder_positions,
            deterministic=False,
            model_mode=common_types.MODEL_MODE_TRAIN,
            rngs={'dropout': rng_dropout_1}
        )

        output_nondeterministic_rng1_again = layer.apply(
            {'params': params},
            inputs,
            decoder_segment_ids,
            decoder_positions,
            deterministic=False,
            model_mode=common_types.MODEL_MODE_TRAIN,
            rngs={'dropout': rng_dropout_1}
        )

        output_nondeterministic_rng2 = layer.apply(
            {'params': params},
            inputs,
            decoder_segment_ids,
            decoder_positions,
            deterministic=False,
            model_mode=common_types.MODEL_MODE_TRAIN,
            rngs={'dropout': rng_dropout_2}
        )

        self.assertTrue(jnp.allclose(output_nondeterministic_rng1, output_nondeterministic_rng1_again), "Outputs with the same dropout RNG key (non-deterministic) should be identical.")

        if self.config.dropout_rate > 0:
          print("Testing dropout active assertions...")
          self.assertFalse(jnp.allclose(output_deterministic, output_nondeterministic_rng1, atol=1e-6, rtol=1e-6), "Deterministic output should differ from non-deterministic (dropout) output when rate > 0.")
          self.assertFalse(jnp.allclose(output_nondeterministic_rng1, output_nondeterministic_rng2, atol=1e-6, rtol=1e-6), "Outputs with different dropout RNGs should differ when rate > 0.")
        else:
          print("Testing dropout inactive assertions (dropout_rate is 0)...")
          self.assertTrue(jnp.allclose(output_deterministic, output_nondeterministic_rng1), "Outputs should be same if dropout_rate is 0.")
          self.assertTrue(jnp.allclose(output_nondeterministic_rng1, output_nondeterministic_rng2), "Outputs should be same if dropout_rate is 0.")
        print("test_deterministic_mode_dropout finished.")

if __name__ == '__main__':
    unittest.main()
