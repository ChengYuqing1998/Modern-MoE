import unittest
import tempfile
from pathlib import Path

import torch

from modern_moe import ModernMoEConfig, ModernMoEForCausalLM
from modern_moe.generation import GenerationConfig, generate
from modern_moe.jlens import (
    JacobianLens,
    capture_residuals,
    capture_residuals_and_routes,
    jacobian_for_tokens,
)
from modern_moe.jlens_visualization import write_jlens_html


def tiny_config() -> ModernMoEConfig:
    return ModernMoEConfig(
        vocab_size=101,
        tokenizer_path="unused-in-unit-test",
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=96,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        num_mtp_layers=1,
        mtp_loss_coef=0.1,
    )


class ModelTest(unittest.TestCase):
    def test_attention_pattern_and_pre_norm(self):
        model = ModernMoEForCausalLM(tiny_config())
        self.assertEqual(
            [layer.attention_type for layer in model.layers],
            ["linear", "linear", "linear", "full"],
        )
        self.assertTrue(all(hasattr(layer, "input_norm") for layer in model.layers))

    def test_dense_prefix_then_sparse_moe(self):
        config = tiny_config()
        config.first_k_dense_replace = 1
        config.dense_intermediate_size = 128
        config.attention_pattern = ("full",)
        config.full_attention_backend = "eager"
        model = ModernMoEForCausalLM(config)
        self.assertFalse(hasattr(model.layers[0].moe, "router"))
        self.assertEqual(
            model.layers[0].moe.ffn.gate_proj.out_features,
            config.dense_intermediate_size,
        )
        self.assertTrue(hasattr(model.layers[1].moe, "router"))
        tokens = torch.randint(0, config.vocab_size, (2, 7))
        output = model(tokens)
        self.assertTrue(torch.isfinite(output.router_aux_loss))
        self.assertTrue(torch.isfinite(output.router_z_loss))

    def test_forward_backward(self):
        torch.manual_seed(0)
        config = tiny_config()
        config.attention_pattern = ("full",)
        config.full_attention_backend = "eager"
        model = ModernMoEForCausalLM(config)
        tokens = torch.randint(0, model.config.vocab_size, (2, 9))
        output = model(tokens, attention_mask=torch.ones_like(tokens), labels=tokens)
        self.assertEqual(output.logits.shape, (2, 9, model.config.vocab_size))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.embed_tokens.weight.grad)

    def test_mtp_loss_and_logits(self):
        torch.manual_seed(1)
        config = tiny_config()
        config.attention_pattern = ("full",)
        config.full_attention_backend = "eager"
        model = ModernMoEForCausalLM(config)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 9))
        targets = torch.randint(0, model.config.vocab_size, (2, 9))
        output = model(
            input_ids,
            mtp_targets=targets,
            return_mtp_logits=True,
        )
        self.assertIsNotNone(output.mtp_loss)
        self.assertTrue(torch.isfinite(output.mtp_loss))
        self.assertEqual(len(output.mtp_logits), 1)
        self.assertEqual(
            output.mtp_logits[0].shape,
            (2, 8, model.config.vocab_size),
        )

    def test_mtp_checkpointed_backward(self):
        torch.manual_seed(2)
        config = tiny_config()
        config.attention_pattern = ("full",)
        config.full_attention_backend = "eager"
        model = ModernMoEForCausalLM(config)
        model.gradient_checkpointing_enable()
        model.train()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 9))
        targets = torch.randint(0, model.config.vocab_size, (2, 9))
        output = model(input_ids, mtp_targets=targets)
        self.assertIsNotNone(output.mtp_loss)
        total_loss = output.logits.float().mean() + output.mtp_loss
        total_loss.backward()
        self.assertIsNotNone(model.mtp_layers[0].fusion.weight.grad)

    def test_incremental_cache_matches_full_forward_on_cpu(self):
        torch.manual_seed(3)
        config = tiny_config()
        config.attention_pattern = ("full",)
        config.full_attention_backend = "sdpa"
        model = ModernMoEForCausalLM(config).eval()
        tokens = torch.randint(0, config.vocab_size, (1, 8))
        expected = model(tokens).logits
        cache = None
        pieces = []
        for position in range(tokens.size(1)):
            output = model.forward_inference(
                tokens[:, position : position + 1],
                cache=cache,
                max_cache_length=tokens.size(1),
            )
            cache = output.cache
            pieces.append(output.logits)
        actual = torch.cat(pieces, dim=1)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

    def test_greedy_generation_modes_match_on_cpu(self):
        torch.manual_seed(4)
        config = tiny_config()
        config.attention_pattern = ("full",)
        config.full_attention_backend = "sdpa"
        model = ModernMoEForCausalLM(config).eval()
        prompt = torch.randint(0, config.vocab_size, (1, 6))
        common = dict(max_new_tokens=5, temperature=0.0, top_p=1.0)
        baseline = generate(
            model,
            prompt,
            GenerationConfig(mode="no_cache", **common),
        )
        cached = generate(
            model,
            prompt,
            GenerationConfig(mode="cache", **common),
        )
        speculative = generate(
            model,
            prompt,
            GenerationConfig(mode="mtp", **common),
        )
        self.assertTrue(torch.equal(baseline.token_ids, cached.token_ids))
        self.assertTrue(torch.equal(baseline.token_ids, speculative.token_ids))
        self.assertEqual(speculative.mtp_proposed, 5)

    def test_jacobian_lens_fit_read_and_round_trip_on_cpu(self):
        torch.manual_seed(5)
        config = tiny_config()
        config.attention_pattern = ("full",)
        config.full_attention_backend = "sdpa"
        config.hidden_size = 16
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.intermediate_size = 24
        model = ModernMoEForCausalLM(config).eval()
        tokens = torch.randint(0, config.vocab_size, (1, 8))
        matrices = jacobian_for_tokens(
            model,
            tokens,
            source_layers=[0, 3],
            dim_batch=2,
            skip_first=1,
        )
        self.assertEqual(matrices[0].shape, (16, 16))
        self.assertTrue(torch.isfinite(matrices[3]).all())
        lens = JacobianLens(
            matrices=matrices,
            hidden_size=16,
            samples=1,
            skip_first=1,
        )
        activations, _ = capture_residuals(model, tokens)
        readout = lens.read(model, activations)
        self.assertEqual(readout[0].shape, (1, config.vocab_size))
        _, _, routes = capture_residuals_and_routes(model, tokens)
        self.assertEqual(routes[0].shape, (1, 8, config.num_experts_per_tok))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lens.pt"
            lens.save(path)
            restored = JacobianLens.load(path)
            torch.testing.assert_close(restored.matrices[3], matrices[3])

    def test_jlens_html_is_self_contained(self):
        payload = {
            "title": "test",
            "prompt": "</script><b>safe</b>",
            "samples": 1,
            "layers": [0],
            "tokens": ["A"],
            "jlens": [[{"top": ["B", "C"], "ids": [1, 2], "margin": 1.0}]],
            "logit": [[{"top": ["C", "B"], "ids": [2, 1], "margin": 0.5}]],
            "routes": [[[0, 1]]],
            "crystallization": [0],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "view.html"
            write_jlens_html(payload, path)
            document = path.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", document)
            self.assertNotIn("</script><b>safe</b>", document)


if __name__ == "__main__":
    unittest.main()
