"""Native Qwen2.5-VL connection tests using tiny random CPU models, no downloads.

The Qwen tests require a transformers installation supporting Qwen2.5-VL.
"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "open_r1"))
from modules.adsa import (
    ADSA, DEFAULT_DISEASE_CATEGORIES, Qwen2_5_VLADSAConnector,
    apa_alignment_loss, extract_disease_categories,
)

try:
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None


CATEGORIES = list(DEFAULT_DISEASE_CATEGORIES)
PROMPT = (
    "Please determine whether each of the following categories appears in the image, "
    "and provide the corresponding bounding box for the disease areas of the categories "
    "that appear; for the categories that do not appear, simply ignore them. " + repr(CATEGORIES)
)


class TinyTokenizer:
    """Test tokenizer with distinct special IDs and a small character vocabulary."""

    pad_token_id = 0
    eos_token_id = 127

    def __init__(self):
        self.query_batches = []

    def encode(self, text, add_special_tokens=False):
        return [20 + ord(char) % 90 for char in text]

    def __call__(self, names, **kwargs):
        self.query_batches.append(list(names))
        rows = [self.encode(name) for name in names]
        length = max(map(len, rows))
        return {
            "input_ids": torch.tensor([row + [0] * (length - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (length - len(row)) for row in rows]),
        }


class QueryParsingTests(unittest.TestCase):
    def test_full_user_instruction_includes_all_candidates(self):
        self.assertEqual(extract_disease_categories(PROMPT), CATEGORIES)
        for invalid in [PROMPT[:-1], "['Pneumonia', 'pneumonia']", "['Pneumonia'] ['Mass']", "[1, 2]"]:
            with self.assertRaises(ValueError):
                extract_disease_categories(invalid)


@unittest.skipIf(Qwen2_5_VLForConditionalGeneration is None, "transformers lacks Qwen2.5-VL")
class NativeQwenConnectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)
        config = Qwen2_5_VLConfig(
            text_config=dict(
                vocab_size=128, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=4096,
                rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
            ),
            vision_config=dict(
                depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                out_hidden_size=32, patch_size=2, temporal_patch_size=1,
                spatial_merge_size=2, window_size=8, fullatt_block_indexes=[0],
            ),
            image_token_id=3, video_token_id=5,
            vision_start_token_id=1, vision_end_token_id=2,
            pad_token_id=0, eos_token_id=127,
        )
        config._attn_implementation = "eager"
        self.model = Qwen2_5_VLForConditionalGeneration(config)
        self.tokenizer = TinyTokenizer()
        self.connector = Qwen2_5_VLADSAConnector(
            self.model, self.tokenizer,
            ADSA(embed_dim=32, text_dim=32, film_hidden=48),
        )
        self.ids = torch.tensor([[9, 1, 3, 3, 3, 3, 2, 10, 11]])
        self.grid = torch.tensor([[1, 4, 4]])
        self.pixels = torch.randn(16, 12)
        self.labels = torch.full_like(self.ids, -100)
        self.labels[0, -2:] = self.ids[0, -2:]

    def test_default_diseases_pack_and_forward_with_correct_labels_and_positions(self):
        vision_calls, text_calls = [], []
        vision_hook = self.model.model.visual.register_forward_hook(lambda *args: vision_calls.append(1))
        text_hook = self.model.model.language_model.register_forward_pre_hook(
            lambda module, args, kwargs: text_calls.append(kwargs["input_ids"].clone())
            if kwargs.get("input_ids") is not None else None,
            with_kwargs=True,
        )
        try:
            packed = self.connector.prepare_inputs(
                self.ids, self.pixels, self.grid, labels=self.labels,
            )
        finally:
            vision_hook.remove()
            text_hook.remove()
        self.assertEqual(len(vision_calls), 1)
        self.assertEqual(len(text_calls), 1)
        self.assertEqual(text_calls[0].shape[0], len(CATEGORIES))
        self.assertEqual(self.tokenizer.query_batches, [CATEGORIES])
        self.assertEqual(packed.candidate_categories, [CATEGORIES])
        self.assertEqual(packed.adsa_output.category_image_embeds[0].shape, (len(CATEGORIES), 4, 32))
        self.assertEqual(packed.adsa_output.apa_outputs[0].prototype_features.shape, (16, 32))
        self.assertEqual((packed.input_ids == 3).sum().item(), len(CATEGORIES) * 4)
        self.assertEqual(packed.image_grid_thw.shape, (len(CATEGORIES), 3))
        self.assertEqual(packed.labels[packed.labels != -100].tolist(), [10, 11])
        expected_positions, _ = self.model.model.get_rope_index(
            packed.input_ids, image_grid_thw=packed.image_grid_thw, attention_mask=packed.attention_mask,
        )
        torch.testing.assert_close(packed.position_ids, expected_positions)
        self.assertNotIn("pixel_values", packed.forward_kwargs())
        self.assertNotIn("image_grid_thw", packed.forward_kwargs())
        output = self.connector(**packed.forward_kwargs(), use_cache=False)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.logits.shape[:2], packed.input_ids.shape)
        alignment = apa_alignment_loss(
            packed.adsa_output.saliency, packed.adsa_output.grids,
            [torch.tensor([[0., 0., 0.5, 1.], [0.5, 0., 1., 1.]])], [torch.tensor([0, 4])],
        )
        (output.loss + 0.5 * alignment).backward()
        for parameter in [
            self.connector.adsa.apa.proto_center,
            self.connector.adsa.apa.query_proj.weight,
            self.connector.adsa.dse.film[-1].weight,
            self.model.model.language_model.layers[0].self_attn.q_proj.weight,
            self.model.model.visual.blocks[0].attn.qkv.weight,
        ]:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_variable_candidates_padding_and_annotation_alignment(self):
        ids = torch.tensor([
            [9, 1, 3, 3, 3, 3, 2, 10, 11],
            [0, 0, 0, 9, 1, 3, 2, 10, 11],
        ])
        mask = (ids != 0).long()
        grid = torch.tensor([[1, 4, 4], [1, 2, 2]])
        pixels = torch.randn(20, 12)
        labels = torch.full_like(ids, -100)
        labels[:, -2:] = ids[:, -2:]
        groups = [["Pneumonia", "Unknown"], ["Mass"]]
        for side in ["left", "right"]:
            packed = self.connector.prepare_inputs(
                ids, pixels, grid, groups, mask, labels, padding_side=side,
            )
            self.assertEqual(packed.image_grid_thw.tolist(), [[1, 4, 4], [1, 4, 4], [1, 2, 2]])
            self.assertEqual(packed.labels[packed.labels != -100].tolist(), [10, 11, 10, 11])
            self.assertTrue((packed.labels[packed.attention_mask == 0] == -100).all())
            edge = 0 if side == "left" else -1
            self.assertEqual(packed.attention_mask[1, edge].item(), 0)
            output = self.connector(**packed.forward_kwargs(), use_cache=False)
            self.assertTrue(torch.isfinite(output.loss))

    def test_two_images_in_one_sample_keep_feature_and_grid_order(self):
        ids = torch.tensor([[9, 1, 3, 3, 3, 3, 2, 10, 1, 3, 2, 11]])
        packed = self.connector.prepare_inputs(
            ids, torch.randn(20, 12), torch.tensor([[1, 4, 4], [1, 2, 2]]),
            [["Pneumonia"], ["Mass", "Unknown"]],
        )
        self.assertEqual(packed.image_grid_thw.tolist(), [[1, 4, 4], [1, 2, 2], [1, 2, 2]])
        output = self.connector(**packed.forward_kwargs(), use_cache=False)
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_generation_cache_and_no_cache_agree_and_hook_is_removed(self):
        self.connector.eval()
        initial_hooks = len(self.model.get_input_embeddings()._forward_hooks)
        kwargs = dict(
            input_ids=self.ids, pixel_values=self.pixels, image_grid_thw=self.grid,
            disease_queries=["Pneumonia", "Unknown"],
            max_new_tokens=3, min_new_tokens=3, do_sample=False,
            eos_token_id=127, pad_token_id=0,
        )
        with_cache = self.connector.generate(**kwargs, use_cache=True)
        without_cache = self.connector.generate(**kwargs, use_cache=False)
        self.assertEqual(with_cache.shape, (1, 3))
        torch.testing.assert_close(with_cache, without_cache)
        self.assertEqual(len(self.model.get_input_embeddings()._forward_hooks), initial_hooks)
        beam = self.connector.generate(**kwargs, num_beams=2)
        self.assertEqual(beam.shape, (1, 3))
        self.assertEqual(len(self.model.get_input_embeddings()._forward_hooks), initial_hooks)

    def test_generation_failure_always_removes_hook(self):
        self.connector.eval()
        initial_hooks = len(self.model.get_input_embeddings()._forward_hooks)
        original = self.model.generate

        def fail(**kwargs):
            raise RuntimeError("intentional test failure")

        self.model.generate = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                self.connector.generate(self.ids, self.pixels, self.grid, ["Pneumonia"])
        finally:
            self.model.generate = original
        self.assertEqual(len(self.model.get_input_embeddings()._forward_hooks), initial_hooks)

    def test_batched_generation_with_variable_grids_and_beams(self):
        self.connector.eval()
        self.model.generation_config.return_dict_in_generate = True
        ids = torch.tensor([
            [9, 1, 3, 3, 3, 3, 2, 10, 11],
            [0, 0, 0, 9, 1, 3, 2, 10, 11],
        ])
        output = self.connector.generate(
            ids, torch.randn(20, 12), torch.tensor([[1, 4, 4], [1, 2, 2]]),
            [["Pneumonia", "Unknown"], ["Mass"]], attention_mask=(ids != 0).long(),
            max_new_tokens=3, min_new_tokens=3, do_sample=False, num_beams=2,
            eos_token_id=127, pad_token_id=0,
        )
        self.assertEqual(output.shape, (2, 3))

    def test_invalid_placeholder_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "placeholder count"):
            self.connector.prepare_inputs(self.ids, self.pixels, torch.tensor([[1, 2, 2]]), CATEGORIES)


if __name__ == "__main__":
    unittest.main()
