

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "open_r1"))
from modules.adsa import DEFAULT_DISEASE_CATEGORIES
from modules.grounding_data import (
    GroundingDataCollator, bbox_targets_from_response, disease_queries_from_prompt,
)
import test_qwen_adsa as qwen_tests

try:
    from transformers import AutoConfig, TrainingArguments
    from modules.stage1 import Qwen2_5_VLForStageI, StageITrainer
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    Qwen2_5_VLForStageI = None


class TinyProcessor:
    image_token = "<image>"

    def __init__(self):
        self.tokenizer = qwen_tests.TinyTokenizer()
        self.tokenizer.convert_tokens_to_ids = lambda token: 3

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        user = "".join(
            "<image>" if item["type"] == "image" else item["text"]
            for item in messages[0]["content"]
        )
        return user + "\nASSISTANT\n" + (messages[1]["content"] if len(messages) > 1 else "")

    def __call__(self, text, images, **kwargs):
        rows = []
        for value in text:
            prefix, suffix = value.split("<image>")
            rows.append(self.tokenizer.encode(prefix) + [1, 3, 3, 3, 3, 2] + self.tokenizer.encode(suffix))
        length = max(map(len, rows))
        return {
            "input_ids": torch.tensor([row + [0] * (length - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (length - len(row)) for row in rows]),
            "pixel_values": torch.arange(len(rows) * 16 * 12).float().reshape(-1, 12) / 100,
            "image_grid_thw": torch.tensor([[1, 4, 4]] * len(rows)),
        }


class GroundingTargetsTests(unittest.TestCase):
    def test_multi_box_and_class_matching_use_original_pixel_size(self):
        categories = list(DEFAULT_DISEASE_CATEGORIES)
        boxes, ids = bbox_targets_from_response(
            "Hepatocellular Carcinoma<box>[[0, 0, 160, 120], [160, 120, 320, 240]]</box>\n"
            "Hepatic Cyst<box>[[80, 60, 160, 120]]</box>",
            categories, (320, 240),
        )
        torch.testing.assert_close(boxes, torch.tensor([[0., 0., .5, .5], [.5, .5, 1., 1.], [.25, .25, .5, .5]]))
        self.assertEqual(ids.tolist(), [0, 0, 6])
        for response in ["Mass<box>[[0,0,10,10]]</box>", "Unknown<box>[[0,0,999,999]]</box>", "Unknown"]:
            with self.assertRaises(ValueError):
                bbox_targets_from_response(response, categories, (320, 240))
        empty, ids = bbox_targets_from_response("[]", categories, (320, 240))
        self.assertEqual(empty.shape, (0, 4))
        self.assertEqual(ids.shape, (0,))

    def test_queries_are_read_from_user_prompt_and_other_coordinate_formats(self):
        self.assertEqual(disease_queries_from_prompt("Check ['Hepatic Cyst', 'Unknown']"), ["Hepatic Cyst", "Unknown"])
        self.assertEqual(disease_queries_from_prompt(
            "Please provide the bounding box coordinate of the region this sentence describes: Nodule, position: right."
        ), ["Nodule"])
        for coordinate_format, response in [
            ("normalized", "Unknown<box>[[0, 0, 0.5, 1]]</box>"),
            ("qwen_1000", "Unknown<box>[[0, 0, 500, 1000]]</box>"),
        ]:
            boxes, _ = bbox_targets_from_response(response, ["Unknown"], (320, 240), coordinate_format)
            torch.testing.assert_close(boxes, torch.tensor([[0., 0., .5, 1.]]))


@unittest.skipIf(Qwen2_5_VLForStageI is None, "Stage I requires transformers with Qwen2.5-VL and PEFT")
class StageITrainingTests(unittest.TestCase):
    def setUp(self):
        fixture = qwen_tests.NativeQwenConnectionTests()
        fixture.setUp()
        self.config = copy.deepcopy(fixture.model.config)
        self.config.stage1_adsa = {"lambda_apa": 0.5, "film_hidden": 48}
        self.model = Qwen2_5_VLForStageI(self.config)
        self.processor = TinyProcessor()
        self.model.configure_stage1(self.processor.tokenizer)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image_path = self.root / "image.png"
        Image.new("RGB", (8, 8), "white").save(image_path)
        self.example = {
            "messages": [
                {"role": "user", "content": "<image>Check ['Hepatocellular Carcinoma', 'Hepatic Cyst', 'Unknown']"},
                {"role": "assistant", "content": "Hepatic Cyst<box>[[0, 0, 4, 8]]</box>"},
            ],
            "images": [str(image_path)],
        }
        self.collator = GroundingDataCollator(self.processor)

    def test_joint_loss_and_collator_annotation_alignment(self):
        batch = self.collator([self.example])
        self.assertEqual(batch["disease_queries"], [["Hepatocellular Carcinoma", "Hepatic Cyst", "Unknown"]])
        self.assertEqual(batch["category_indices"][0].tolist(), [1])
        torch.testing.assert_close(batch["boxes_xyxy"][0], torch.tensor([[0., 0., .5, 1.]]))
        outputs = self.model(**batch)
        torch.testing.assert_close(outputs.loss, outputs.loss_gnd + 0.3 * outputs.loss_apa)
        self.assertGreater(outputs.loss_apa.item(), 0)
        outputs.loss.backward()
        for parameter in [self.model.adsa.apa.proto_center, self.model.adsa.dse.film[-1].weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "boxes_xyxy"):
            self.model(**{key: value for key, value in batch.items() if key != "boxes_xyxy"})

    def test_full_checkpoint_roundtrip_and_generation_preserve_prefix(self):
        batch = self.collator([self.example])
        self.model.eval()
        reference = self.model(**batch)
        saved = self.root / "full"
        self.model.save_pretrained(saved)
        restored = Qwen2_5_VLForStageI.from_pretrained(saved).configure_stage1(self.processor.tokenizer).eval()
        actual = restored(**batch)
        torch.testing.assert_close(actual.logits, reference.logits)
        torch.testing.assert_close(actual.loss, reference.loss)
        self.assertEqual(restored.config.stage1_adsa["lambda_apa"], 0.5)
        prompt = {key: batch[key] for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "disease_queries")}
        ids = restored.generate(**prompt, max_new_tokens=2, min_new_tokens=2, do_sample=False, pad_token_id=0, eos_token_id=127)
        torch.testing.assert_close(ids[:, :batch["input_ids"].shape[1]], batch["input_ids"])
        self.assertEqual(ids.shape[1], batch["input_ids"].shape[1] + 2)

    def test_loading_native_base_initializes_missing_gaussians_and_film(self):
        native = qwen_tests.Qwen2_5_VLForConditionalGeneration(copy.deepcopy(self.config))
        base = self.root / "native"
        native.save_pretrained(base)
        loaded = Qwen2_5_VLForStageI.from_pretrained(base)
        expected = torch.tensor([[(x + .5) / 4, (y + .5) / 4] for y in range(4) for x in range(4)])
        torch.testing.assert_close(loaded.adsa.apa.centers, expected)
        torch.testing.assert_close(loaded.adsa.apa.scales, torch.full((16, 2), .25))
        self.assertEqual(loaded.adsa.dse.film[-1].weight.abs().sum().item(), 0)
        self.assertEqual(loaded.adsa.dse.film[-1].bias.abs().sum().item(), 0)

    def make_peft_model(self):
        self.model.requires_grad_(False)
        model = get_peft_model(self.model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["adsa"], lora_dropout=0,
        ))
        model.enable_input_require_grads()
        return model

    def trainer(self, model, output_dir, max_steps):
        args = TrainingArguments(
            output_dir=str(output_dir), use_cpu=True, report_to="none",
            per_device_train_batch_size=1, gradient_accumulation_steps=2,
            max_steps=max_steps, learning_rate=1e-3, logging_steps=1,
            save_steps=1, save_strategy="steps", remove_unused_columns=False,
            label_names=["labels"], disable_tqdm=True,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        return StageITrainer(
            model=model, args=args, train_dataset=[self.example, self.example], data_collator=self.collator,
        )

    def test_real_optimizer_step_peft_save_reload_and_resume(self):
        # Use a real base checkpoint so the adapter can be reconstructed.
        base = self.root / "base"
        self.model.save_pretrained(base)
        model = self.make_peft_model()
        model.peft_config["default"].base_model_name_or_path = str(base)
        active_adsa = model.get_base_model().adsa.modules_to_save["default"]
        before = active_adsa.apa.proto_center.detach().clone()
        output_dir = self.root / "train"
        trainer = self.trainer(model, output_dir, max_steps=1)
        trainer.train()
        self.assertFalse(torch.equal(before, active_adsa.apa.proto_center))
        logs = next(row for row in trainer.state.log_history if "loss_gnd" in row)
        self.assertIn("loss_apa", logs)
        metrics = trainer.evaluate(eval_dataset=[self.example])
        self.assertTrue(torch.isfinite(torch.tensor(metrics["eval_loss"])))
        eval_logs = next(row for row in trainer.state.log_history if "eval_loss_gnd" in row)
        self.assertIn("eval_loss_apa", eval_logs)
        checkpoint = output_dir / "checkpoint-1"
        self.assertTrue((checkpoint / "adapter_model.safetensors").is_file())
        self.assertTrue((checkpoint / "config.json").is_file())
        config = AutoConfig.from_pretrained(checkpoint)
        restored_base = Qwen2_5_VLForStageI.from_pretrained(base, config=config)
        restored_base.configure_stage1(self.processor.tokenizer)
        restored = PeftModel.from_pretrained(restored_base, checkpoint, is_trainable=True)
        restored_adsa = restored.get_base_model().adsa.modules_to_save["default"]
        torch.testing.assert_close(restored_adsa.apa.proto_center, active_adsa.apa.proto_center)
        model.eval()
        restored.eval()
        batch = self.collator([self.example])
        torch.testing.assert_close(restored(**batch).loss, model(**batch).loss)
        restored.train()
        continued = self.trainer(restored, output_dir, max_steps=2)
        continued.train(resume_from_checkpoint=str(checkpoint))
        self.assertEqual(continued.state.global_step, 2)
        self.assertFalse(torch.equal(restored_adsa.apa.proto_center, before))
        self.assertTrue((output_dir / "checkpoint-2" / "adapter_model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
