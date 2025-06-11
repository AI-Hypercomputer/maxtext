import sys
import unittest

sys.path.insert(0, '/workspace/transformers/src')

import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modular_qwen3_moe import Qwen3MoeSparseMoeBlock


class Qwen3MoeRoutingTest(unittest.TestCase):
    """Basic tests for Qwen3 MoE routing logic."""

    def test_topk_prob_normalization(self):
        config = Qwen3MoeConfig(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
        block = Qwen3MoeSparseMoeBlock(config)
        hidden_states = torch.randn(2, 3, 16)
        _output, router_logits = block(hidden_states)
        probs = torch.softmax(router_logits, dim=-1)
        topk, _ = torch.topk(probs, config.num_experts_per_tok, dim=-1)
        sums = topk.sum(dim=-1)
        ones = torch.ones_like(sums)
        self.assertTrue(torch.allclose(sums, ones, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
