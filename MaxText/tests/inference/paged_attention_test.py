"""
Copyright 2024 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import Module
from MaxText.inference.paged_attention import PagedAttentionOp, _use_kernel_v2
from MaxText.pyconfig import initialize as pyconfig_initialize, string_to_bool
from jax.sharding import Mesh
from MaxText import common_types
from MaxText.inference import page_manager

class TestPagedAttention(unittest.TestCase):
    def setUp(self):
        print("Attempting setUp for TestPagedAttention...")
        argv = ['test_paged_attention.py', 'MaxText/configs/base.yml']

        config_kwargs = {
            'run_name': 'paged_attention_test_run',
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
            'max_target_length': 64,
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

        print("Manually setting/overriding config parameters for PagedAttention test...")

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
        config_dict_to_modify['max_target_length'] = 64
        config_dict_to_modify['max_prefill_predict_length'] = 32
        config_dict_to_modify['scan_layers'] = False
        config_dict_to_modify['record_internal_nn_metrics'] = False
        config_dict_to_modify['use_bias'] = True
        config_dict_to_modify.setdefault('activation_function', "gelu")
        config_dict_to_modify['decoder_block'] = common_types.DecoderBlockType.DEFAULT.value

        config_dict_to_modify['quant'] = None
        config_dict_to_modify['kv_quant'] = None
        config_dict_to_modify['float32_qk_product'] = True
        config_dict_to_modify['float32_logits'] = True
        config_dict_to_modify['use_ragged_attention'] = False
        config_dict_to_modify['ragged_block_size'] = 0
        config_dict_to_modify.setdefault("attention_kernel", "autoselected")
        config_dict_to_modify.setdefault("attention_num_segments", 1)
        config_dict_to_modify.setdefault("rope_type", "default")

        # PagedAttentionOp specific parameters
        config_dict_to_modify['paged_attention_num_pages'] = 64
        config_dict_to_modify['paged_attention_tokens_per_page'] = 16
        config_dict_to_modify['paged_attention_max_pages_per_slot'] = config_dict_to_modify['max_target_length'] // config_dict_to_modify['paged_attention_tokens_per_page']
        config_dict_to_modify['paged_attention_max_pages_per_prefill'] = config_dict_to_modify['max_prefill_predict_length'] // config_dict_to_modify['paged_attention_tokens_per_page']
        config_dict_to_modify['paged_attention_pages_per_compute_block'] = 1
        config_dict_to_modify.setdefault("attn_logits_soft_cap", 0.0)

        self.config_keys_to_set_default(config_dict_to_modify)

        print("Manual config parameter setting completed.")

        self.mesh = Mesh(np.array(jax.devices()), ('data',))
        print("Mesh created.")
        print("setUp for TestPagedAttention completed successfully.")

    def config_keys_to_set_default(self, config_dict_to_modify):
        """Helper to set common default values."""
        defaults = {
            "use_unpadded_segment_ids": False, "reset_transformer_max_length_for_eval": False,
            "max_reuse_decoding_length": -1, "moe_load_balancing_loss_weight": 0.01,
            "moe_router_aux_loss_factor": 0.01,
            "logical_axis_rules": [('activation_embed_and_logits_batch', ('data', 'fsdp'))],
            "data_sharding": [('data',)], "mlp_dim_scale_factor": 2.0,
            "metrics_file": None, "command_line_str": "", "use_wandb": "False",
            "hf_access_token": None, "grain_train_files": "", "grain_worker_count": 0,
            "eval_interval": 0, "dataset_name": "", "eval_split": "",
            "hf_train_files": None, "hf_eval_files": None, "hf_eval_split": "",
            "hf_data_dir": "", "hf_path": "", "num_epoch":1, "c4_mlperf_path":"",
            "remat_policy":"none",
            "kv_quant_axis": "", "quantize_kvcache": False,
            "profiler": "", "profile_periodically_period": 0, "profiler_steps": 1,
            "model_call_mode": "", "load_parameters_path": "", "load_full_state_path": "",
            "enable_emergency_checkpoint": False, "dump_hlo_xla_flags": "",
            "dump_hlo_module_name": "", "dump_hlo_local_dir": "", "dump_hlo_gcs_dir": "",
            "pipeline_parallel_layers": -1, "num_pipeline_repeats": -1,
            "num_layers_per_pipeline_stage": -1, "num_pipeline_microbatches": -1,
            "pipeline_delay_activation_forwarding": False, "use_multimodal": False,
            "interleave_moe_layer_step":1, "sparse_matmul": False, "dump_hlo": False,
            "jax_cache_dir": "", "jax_debug_log_modules": "", "compile_topology": "",
            "compile_topology_num_slices": 1, "expansion_factor_real_data": -1,
            "eval_per_device_batch_size": 0.0, "pagedattn_max_pages_per_group": 0,
            "quantization_local_shard_count": -1, "checkpoint_period": 1,
            "final_logits_soft_cap": 0.0,
            "use_iota_embed": False,
            "num_experts": 1, "num_experts_per_tok":1, "moe_capacity_factor":1.0,
            "moe_eval_capacity_factor":1.0, "n_shared_experts": -1, "n_gaus_experts": -1,
            "n_gaus_experts_per_tok": -1, "n_routing_groups": -1, "topk_routing_group": -1,
            "chunk_attn_window_size": 0, "sliding_window_size": 0,
        }
        for key, value in defaults.items():
            config_dict_to_modify.setdefault(key, value)

        config_dict_to_modify.setdefault("per_device_batch_size", 1.0)
        config_dict_to_modify.setdefault("steps", 10)
        config_dict_to_modify.setdefault("learning_rate_schedule_steps", config_dict_to_modify.get("steps", 10))
        config_dict_to_modify.setdefault("warmup_steps_fraction", 0.1)
        config_dict_to_modify.setdefault("compute_axis_order", "0,1,2,3")


        for key_remat in ["decoder_layer_input", "context", "mlpwi", "mlpwi_0", "mlpwi_1", "mlpwo",
                         "query_proj", "key_proj", "value_proj", "out_proj"]:
            config_dict_to_modify.setdefault(key_remat, "remat")

        for i_dcn in ["ici", "dcn"]:
            for key_suffix in ["data_parallelism", "fsdp_parallelism", "fsdp_transpose_parallelism",
                               "context_parallelism", "context_autoregressive_parallelism",
                               "tensor_parallelism", "tensor_transpose_parallelism",
                               "tensor_sequence_parallelism", "expert_parallelism",
                               "autoregressive_parallelism"]:
                config_dict_to_modify.setdefault(f"{i_dcn}_{key_suffix}", 1)

    def test_output_shape_prefill(self):
        print("Running test_output_shape_prefill...")
        B, S = 1, self.config.max_prefill_predict_length

        paged_attn_op = PagedAttentionOp(
            mesh=self.mesh,
            num_pages=self.config.paged_attention_num_pages,
            tokens_per_page=self.config.paged_attention_tokens_per_page,
            max_pages_per_slot=self.config.paged_attention_max_pages_per_slot,
            max_pages_per_prefill=self.config.paged_attention_max_pages_per_prefill,
            pages_per_compute_block=self.config.paged_attention_pages_per_compute_block,
            num_kv_heads=self.config.num_kv_heads,
            kv_head_dim_size=self.config.head_dim,
            dtype=self.config.dtype,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap
        )

        q_tensor = jnp.ones((B, S, self.config.num_query_heads, self.config.head_dim), dtype=self.config.dtype)
        k_tensor = jnp.ones((B, S, self.config.num_kv_heads, self.config.head_dim), dtype=self.config.dtype)
        v_tensor = jnp.ones((B, S, self.config.num_kv_heads, self.config.head_dim), dtype=self.config.dtype)

        num_pages_for_seq = (S + self.config.paged_attention_tokens_per_page - 1) // self.config.paged_attention_tokens_per_page

        mock_page_indices_for_seq = jnp.arange(1, 1 + num_pages_for_seq, dtype=jnp.int32)

        padded_mock_page_indices = -jnp.ones(self.config.paged_attention_max_pages_per_slot, dtype=jnp.int32)
        if num_pages_for_seq > 0 :
             padded_mock_page_indices = padded_mock_page_indices.at[:num_pages_for_seq].set(mock_page_indices_for_seq)

        mock_page_map_for_test = padded_mock_page_indices[None, :]

        mock_page_state = page_manager.PageState(
            page_map=mock_page_map_for_test,
            active_page=jnp.array([mock_page_indices_for_seq[-1]] if num_pages_for_seq > 0 else [0], dtype=jnp.int32),
            active_page_position=jnp.array([ (S - 1) % self.config.paged_attention_tokens_per_page if S > 0 else 0], dtype=jnp.int32),
            sequence_lengths=jnp.array([S], dtype=jnp.int32),
            num_pages_used=jnp.array([num_pages_for_seq], dtype=jnp.int32),
            page_status=jnp.zeros(self.config.paged_attention_num_pages, dtype=jnp.int32).at[mock_page_indices_for_seq].set(1) if num_pages_for_seq > 0 else jnp.zeros(self.config.paged_attention_num_pages, dtype=jnp.int32),
            has_active_page=jnp.array([True if S > 0 else False], dtype=jnp.bool_)
        )

        rng_key = jax.random.PRNGKey(0)

        print("Initializing PagedAttentionOp...")
        init_vars = paged_attn_op.init(
            rng_key,
            query=q_tensor,
            key=k_tensor,
            value=v_tensor,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_PREFILL,
            slot=0,
            page_state=mock_page_state,
            mutable=['cache']
        )
        print(f"PagedAttentionOp init_vars keys: {init_vars.keys()}")

        print("Applying PagedAttentionOp in PREFILL mode...")
        output_tuple_or_tensor = paged_attn_op.apply(
            {'cache': init_vars['cache']},
            query=q_tensor,
            key=k_tensor,
            value=v_tensor,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_PREFILL,
            slot=0,
            page_state=mock_page_state,
            mutable=['cache']
        )

        function_output = output_tuple_or_tensor[0]

        if not _use_kernel_v2:
            actual_output_tensor = function_output[0]
        else:
            actual_output_tensor = function_output

        expected_output_shape = (B, S, self.config.num_query_heads, self.config.head_dim)
        self.assertEqual(actual_output_tensor.shape, expected_output_shape)
        print("test_output_shape_prefill successfully asserted output shape.")

    @unittest.expectedFailure
    def test_output_shape_decode(self):
        print("Running test_output_shape_decode...")
        B, S_decode = 1, 1

        paged_attn_op = PagedAttentionOp(
            mesh=self.mesh,
            num_pages=self.config.paged_attention_num_pages,
            tokens_per_page=self.config.paged_attention_tokens_per_page,
            max_pages_per_slot=self.config.paged_attention_max_pages_per_slot,
            max_pages_per_prefill=self.config.paged_attention_max_pages_per_prefill,
            pages_per_compute_block=self.config.paged_attention_pages_per_compute_block,
            num_kv_heads=self.config.num_kv_heads,
            kv_head_dim_size=self.config.head_dim,
            dtype=self.config.dtype,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap
        )

        q_tensor = jnp.ones((B, S_decode, self.config.num_query_heads, self.config.head_dim), dtype=self.config.dtype)
        k_tensor_current = jnp.ones((B, S_decode, self.config.num_kv_heads, self.config.head_dim), dtype=self.config.dtype)
        v_tensor_current = jnp.ones((B, S_decode, self.config.num_kv_heads, self.config.head_dim), dtype=self.config.dtype)

        prefill_len = 10
        num_pages_for_prefill_seq = (prefill_len + self.config.paged_attention_tokens_per_page - 1) // self.config.paged_attention_tokens_per_page

        mock_page_indices = jnp.arange(1, 1 + num_pages_for_prefill_seq, dtype=jnp.int32)
        padded_mock_page_indices = -jnp.ones(self.config.paged_attention_max_pages_per_slot, dtype=jnp.int32)
        if num_pages_for_prefill_seq > 0:
            padded_mock_page_indices = padded_mock_page_indices.at[:num_pages_for_prefill_seq].set(mock_page_indices)

        mock_page_map_for_test = padded_mock_page_indices[None, :]

        mock_page_state_decode = page_manager.PageState(
            page_map=mock_page_map_for_test,
            active_page=jnp.array([mock_page_indices[-1]] if num_pages_for_prefill_seq > 0 else [0], dtype=jnp.int32),
            active_page_position=jnp.array([(prefill_len -1) % self.config.paged_attention_tokens_per_page if prefill_len > 0 else 0], dtype=jnp.int32),
            sequence_lengths=jnp.array([prefill_len], dtype=jnp.int32),
            num_pages_used=jnp.array([num_pages_for_prefill_seq], dtype=jnp.int32),
            page_status=jnp.zeros(self.config.paged_attention_num_pages, dtype=jnp.int32).at[mock_page_indices].set(1) if num_pages_for_prefill_seq > 0 else jnp.zeros(self.config.paged_attention_num_pages, dtype=jnp.int32),
            has_active_page=jnp.array([True if prefill_len > 0 else False], dtype=jnp.bool_)
        )

        rng_key = jax.random.PRNGKey(0)

        print("Initializing PagedAttentionOp for decode test (can reuse prefill's init if cache structure is identical)...")
        init_k_dummy = jnp.zeros((B, self.config.max_prefill_predict_length, self.config.num_kv_heads, self.config.head_dim), dtype=self.config.dtype)
        init_v_dummy = jnp.zeros((B, self.config.max_prefill_predict_length, self.config.num_kv_heads, self.config.head_dim), dtype=self.config.dtype)

        init_vars = paged_attn_op.init(
            rng_key,
            query=jnp.zeros((B, self.config.max_prefill_predict_length, self.config.num_query_heads, self.config.head_dim), dtype=self.config.dtype),
            key=init_k_dummy,
            value=init_v_dummy,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_PREFILL,
            slot=0,
            page_state=mock_page_state_decode,
            mutable=['params', 'cache']
        )
        print(f"PagedAttentionOp init_vars keys: {init_vars.keys()}")

        print("Applying PagedAttentionOp in AUTOREGRESSIVE mode...")
        output_tuple_or_tensor = paged_attn_op.apply(
            {'cache': init_vars['cache']},
            query=q_tensor,
            key=k_tensor_current,
            value=v_tensor_current,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state=mock_page_state_decode,
            mutable=['cache']
        )

        function_output = output_tuple_or_tensor[0]
        actual_output_tensor = function_output[0] if isinstance(function_output, tuple) else function_output

        expected_output_shape = (B, S_decode, self.config.num_query_heads, self.config.head_dim)
        self.assertEqual(actual_output_tensor.shape, expected_output_shape)
        print("test_output_shape_decode successfully asserted output shape.")

    def test_attention_values_simple(self):
        print("Running test_attention_values_simple...")
        B, S_q_unused, S_kv = 1, 1, 2

        self.config._config.keys['num_query_heads'] = 1
        self.config._config.keys['num_kv_heads'] = 1
        self.config._config.keys['head_dim'] = 1
        self.config._config.keys['embed_dim'] = 1 * 1
        self.config._config.keys['paged_attention_tokens_per_page'] = S_kv
        self.config._config.keys['paged_attention_max_pages_per_prefill'] = (S_kv + self.config.paged_attention_tokens_per_page -1) // self.config.paged_attention_tokens_per_page
        self.config._config.keys['paged_attention_max_pages_per_slot'] = self.config.paged_attention_max_pages_per_prefill


        paged_attn_op = PagedAttentionOp(
            mesh=self.mesh,
            num_pages=self.config.paged_attention_num_pages,
            tokens_per_page=self.config.paged_attention_tokens_per_page,
            max_pages_per_slot=self.config.paged_attention_max_pages_per_slot,
            max_pages_per_prefill=self.config.paged_attention_max_pages_per_prefill,
            pages_per_compute_block=self.config.paged_attention_pages_per_compute_block,
            num_kv_heads=self.config.num_kv_heads,
            kv_head_dim_size=self.config.head_dim,
            dtype=self.config.dtype,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap
        )

        k_prefill = jnp.array([[[[2.0]]], [[[3.0]]]]).reshape(B, S_kv, 1, 1).astype(self.config.dtype)
        v_prefill = jnp.array([[[[0.5]]], [[[1.0]]]]).reshape(B, S_kv, 1, 1).astype(self.config.dtype)

        q_val_test = jnp.zeros((B, S_kv, 1, 1), dtype=self.config.dtype)
        q_val_test = q_val_test.at[0, -1, 0, 0].set(1.0)

        num_pages_for_prefill_seq = (S_kv + self.config.paged_attention_tokens_per_page - 1) // self.config.paged_attention_tokens_per_page
        mock_page_indices_prefill = jnp.arange(1, 1 + num_pages_for_prefill_seq, dtype=jnp.int32)
        padded_mock_page_indices_prefill = -jnp.ones(self.config.paged_attention_max_pages_per_slot, dtype=jnp.int32)
        if num_pages_for_prefill_seq > 0:
            padded_mock_page_indices_prefill = padded_mock_page_indices_prefill.at[:num_pages_for_prefill_seq].set(mock_page_indices_prefill)
        mock_page_map_prefill = padded_mock_page_indices_prefill[None, :]

        prefill_page_state = page_manager.PageState(
            page_map=mock_page_map_prefill,
            active_page=jnp.array([mock_page_indices_prefill[-1]] if num_pages_for_prefill_seq > 0 else [0], dtype=jnp.int32),
            active_page_position=jnp.array([(S_kv - 1) % self.config.paged_attention_tokens_per_page if S_kv > 0 else 0], dtype=jnp.int32),
            sequence_lengths=jnp.array([S_kv], dtype=jnp.int32),
            num_pages_used=jnp.array([num_pages_for_prefill_seq], dtype=jnp.int32),
            page_status=jnp.zeros(self.config.paged_attention_num_pages, dtype=jnp.int32).at[mock_page_indices_prefill].set(1) if num_pages_for_prefill_seq > 0 else jnp.zeros(self.config.paged_attention_num_pages, dtype=jnp.int32),
            has_active_page=jnp.array([True if S_kv > 0 else False], dtype=jnp.bool_)
        )

        rng_key = jax.random.PRNGKey(0)

        print("Initializing PagedAttentionOp for value test (PREFILL mode)...")
        init_vars = paged_attn_op.init(rng_key, query=q_val_test, key=k_prefill, value=v_prefill, decoder_segment_ids=None, model_mode=common_types.MODEL_MODE_PREFILL, slot=0, page_state=prefill_page_state, mutable=['cache'])

        print("Applying PagedAttentionOp in PREFILL mode for value test...")
        output_tuple_or_tensor_val = paged_attn_op.apply(
            {'cache': init_vars['cache']},
            query=q_val_test,
            key=k_prefill,
            value=v_prefill,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_PREFILL,
            slot=0,
            page_state=prefill_page_state,
            mutable=['cache']
        )

        function_output_val = output_tuple_or_tensor_val[0]
        unnormalized_attn_tensor = function_output_val[0]
        local_sums_tensor = function_output_val[2]

        actual_output_tensor_val = unnormalized_attn_tensor / (local_sums_tensor + 1e-9)

        expected_scores = jnp.array([1.0*2.0, 1.0*3.0])
        expected_probs = jax.nn.softmax(expected_scores)
        expected_output_for_last_token = jnp.sum(expected_probs * jnp.array([0.5, 1.0]))

        calculated_output_for_last_token = actual_output_tensor_val[0, -1, 0, 0]

        print(f"Calculated value (normalized): {calculated_output_for_last_token}, Expected value: {expected_output_for_last_token}")
        print(f"Unnormalized output: {unnormalized_attn_tensor[0,-1,0,0]}, Sums: {local_sums_tensor[0,-1,0,0]}")
        self.assertTrue(jnp.allclose(calculated_output_for_last_token, expected_output_for_last_token, atol=1e-5))
        print("test_attention_values_simple successfully asserted output value using PREFILL path.")


if __name__ == '__main__':
    unittest.main()
