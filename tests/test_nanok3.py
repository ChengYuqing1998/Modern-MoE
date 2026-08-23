import unittest

import torch

from nanok3 import NanoK3Config, NanoK3ForCausalLM, parameter_report
from nanok3.layers import SiTUAndMul


def tiny_config(**overrides):
    values = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        attention_pattern=("mla",),
        final_layer_is_mla=True,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        kda_head_dim=8,
        kda_num_heads=4,
        first_k_dense_replace=1,
        dense_intermediate_size=64,
        latent_moe_dim=24,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        attn_res_block_size=2,
        attn_res_checkpoint=True,
        mla_backend="eager",
    )
    values.update(overrides)
    return NanoK3Config(**values)


class NanoK3Test(unittest.TestCase):
    def test_k3_attention_pattern(self):
        config = NanoK3Config(num_hidden_layers=12)
        pattern = [
            config.attention_type(index)
            for index in range(config.num_hidden_layers)
        ]
        self.assertEqual(pattern.count("kda"), 9)
        self.assertEqual(pattern.count("mla"), 3)
        self.assertEqual(pattern[-1], "mla")

    def test_cpu_mla_forward_backward(self):
        torch.manual_seed(7)
        model = NanoK3ForCausalLM(tiny_config())
        model.train()
        input_ids = torch.randint(0, 128, (2, 8))
        output = model(input_ids, labels=input_ids)
        self.assertEqual(output.logits.shape, (2, 8, 128))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.layers[1].feed_forward.router.weight.grad)
        self.assertGreater(
            model.layers[1].feed_forward.router.weight.grad.abs().sum().item(),
            0,
        )

    def test_quantile_balancing_is_centered(self):
        torch.manual_seed(11)
        model = NanoK3ForCausalLM(tiny_config())
        router = model.layers[1].feed_forward.router
        before = router.expert_bias.clone()
        model(torch.randint(0, 128, (2, 8)))
        self.assertFalse(torch.equal(before, router.expert_bias))
        self.assertAlmostEqual(router.expert_bias.mean().item(), 0.0, places=5)

    def test_situ_is_bounded(self):
        activation = SiTUAndMul(beta=4.0, linear_beta=25.0)
        x = torch.tensor([1e4, -1e4])
        y = activation(x, x)
        self.assertLessEqual(y.abs().max().item(), 100.001)

    def test_parameter_report(self):
        model = NanoK3ForCausalLM(tiny_config())
        report = parameter_report(model)
        self.assertGreater(report["total_parameters"], 0)
        self.assertLessEqual(
            report["active_parameters_per_token"],
            report["total_parameters"],
        )

    def test_single_token_moe_fast_path_matches_generic_dispatch(self):
        torch.manual_seed(19)
        model = NanoK3ForCausalLM(tiny_config(quantile_balancing=False))
        moe = model.layers[1].feed_forward.eval()
        token = torch.randn(1, 1, model.config.hidden_size)
        fast = moe(token)
        # Two identical tokens force the original general dispatch path.
        generic = moe(token.expand(1, 2, -1))[:, :1]
        torch.testing.assert_close(fast, generic, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
