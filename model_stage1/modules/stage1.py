"""Qwen2.5-VL Stage I: grounding NLL + lambda_APA * bbox-guided APA BCE.

The model owns APA/DSE so DDP/DeepSpeed see the complete forward graph. With
PEFT, use modules_to_save=['adsa'] to include APA/DSE in every adapter checkpoint.
The processor/tokenizer must be attached after from_pretrained(), before use.
"""

import math
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Trainer

from .adsa import ADSA, APA, DSE, Qwen2_5_VLADSAConnector, apa_alignment_loss


class Qwen2_5_VLForStageI(Qwen2_5_VLForConditionalGeneration):
    # Losses are microbatch means. Trainer must divide the entire joint loss
    # by gradient_accumulation_steps, not normalize only NLL by token count.
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.loss_type = "ForCausalLM"
        settings = {
            "lambda_apa": 0.5,
            "num_prototypes": 16,
            "proto_dim": None,
            "film_hidden": 512,
            "dropout": 0.0,
            "initial_sigma": 0.25,
            "identity_init": True,
        }
        settings.update(getattr(config, "stage1_adsa", {}))
        config.stage1_adsa = settings
        dim = getattr(config, "text_config", config).hidden_size
        self.adsa = ADSA(
            embed_dim=dim, text_dim=dim,
            **{key: value for key, value in settings.items() if key != "lambda_apa"},
        )
        # HF calls _init_weights for missing APA/DSE keys after materializing
        # meta tensors. Do not mark these modules initialized prematurely here.
        self._stage1_tokenizer = None
        self._stage1_generating = False
        config.keys_to_ignore_at_inference = list(dict.fromkeys([
            *getattr(config, "keys_to_ignore_at_inference", []), "loss_gnd", "loss_apa",
        ]))

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, APA):
            k = module.num_prototypes
            cols = math.ceil(math.sqrt(k))
            rows = math.ceil(k / cols)
            indices = torch.arange(k, device=module.proto_center.device)
            centers = torch.stack(
                ((indices % cols + 0.5) / cols, (indices // cols + 0.5) / rows), -1,
            )
            sigma = getattr(self.config, "stage1_adsa", {}).get("initial_sigma", 0.25) - module.eps
            raw_sigma = sigma + math.log(-math.expm1(-sigma))
            with torch.no_grad():
                module.proto_center.copy_(torch.logit(centers))
                module.proto_raw_sigma.fill_(raw_sigma)
        if isinstance(module, DSE) and getattr(self.config, "stage1_adsa", {}).get("identity_init", True):
            nn.init.zeros_(module.film[-1].weight)
            nn.init.zeros_(module.film[-1].bias)

    def configure_stage1(self, tokenizer, lambda_apa=None):
        self._stage1_tokenizer = tokenizer
        if lambda_apa is not None:
            if not math.isfinite(lambda_apa) or lambda_apa < 0:
                raise ValueError("lambda_apa must be finite and nonnegative.")
            self.config.stage1_adsa["lambda_apa"] = float(lambda_apa)
        return self

    def _connector(self):
        if self._stage1_tokenizer is None:
            raise ValueError("Call model.configure_stage1(processor.tokenizer) before Stage I forward/generation.")
        # Do not register a connector on self: it already owns this model.
        # Only the actual adsa parameters are registered in the saved model.
        connector = Qwen2_5_VLADSAConnector(self, self._stage1_tokenizer, self.adsa)
        connector.training = self.training
        return connector

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        disease_queries=None,
        boxes_xyxy=None,
        category_indices=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        # Native text-only/cached decoder passes, including generation after
        # connector feature injection, must not re-run APA or the vision encoder.
        if pixel_values is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                image_grid_thw=image_grid_thw, use_cache=use_cache,
                return_dict=return_dict, **kwargs,
            )
        if inputs_embeds is not None or past_key_values is not None:
            raise ValueError("Stage I image forward expects original processor inputs, without embeddings/cache.")
        if labels is not None and boxes_xyxy is None:
            raise ValueError("Stage I supervised forward requires boxes_xyxy (empty (0,4) tensors for negatives).")
        packed = self._connector().prepare_inputs(
            input_ids, pixel_values, image_grid_thw, disease_queries,
            attention_mask=attention_mask, labels=labels, padding_side="right",
        )
        if labels is not None and not (packed.labels != -100).any():
            raise ValueError("Stage I labels contain no supervised response tokens.")
        # Original sequence positions/cache indices are invalid after expansion.
        kwargs.pop("cache_position", None)
        kwargs.pop("num_items_in_batch", None)
        outputs = super().forward(
            **packed.forward_kwargs(), use_cache=False if labels is not None else use_cache,
            return_dict=True, **kwargs,
        )
        if labels is not None:
            loss_gnd = outputs.loss
            loss_apa = apa_alignment_loss(
                packed.adsa_output.saliency, packed.adsa_output.grids,
                boxes_xyxy, category_indices,
            )
            # Equation (12): L_Stage I = L_gnd + lambda_APA * L_APA.
            outputs.loss = loss_gnd + self.config.stage1_adsa["lambda_apa"] * loss_apa
            outputs["loss_gnd"] = loss_gnd
            outputs["loss_apa"] = loss_apa
        return outputs.to_tuple() if return_dict is False else outputs

    def generate(self, input_ids=None, **kwargs):
        if self._stage1_generating or kwargs.get("pixel_values") is None:
            return super().generate(input_ids=input_ids, **kwargs)
        pixel_values = kwargs.pop("pixel_values")
        grid = kwargs.pop("image_grid_thw")
        queries = kwargs.pop("disease_queries", None)
        mask = kwargs.pop("attention_mask", None)
        if any(key in kwargs for key in ("labels", "boxes_xyxy", "category_indices")):
            raise ValueError("Generation takes prompts/images and candidate names, without GT labels or boxes.")
        self._stage1_generating = True
        try:
            continuation = self._connector().generate(
                input_ids, pixel_values, grid, queries, attention_mask=mask, **kwargs,
            )
        finally:
            self._stage1_generating = False
        # Preserve HF's public generate contract: original prompt + continuation.
        repeats = continuation.shape[0] // input_ids.shape[0]
        return torch.cat((input_ids.repeat_interleave(repeats, dim=0).to(continuation.device), continuation), dim=1)


class StageITrainer(Trainer):
    """Use the joint model loss, logging L_gnd and L_APA separately.

    Plain Trainer avoids SFTTrainer's token-accuracy/entropy calculations that
    assume logits and the ORIGINAL, unexpanded input batch have equal lengths.
    The collator already supplies assistant-only labels, so no SFT preprocessing
    or independent optimizer is needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._stage1_metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        if self.args.gradient_checkpointing:
            checkpoint_kwargs = dict(self.args.gradient_checkpointing_kwargs or {})
            if checkpoint_kwargs.get("use_reentrant", False):
                raise ValueError("Stage I reuses the text backbone; set gradient_checkpointing_kwargs.use_reentrant=false.")
            checkpoint_kwargs["use_reentrant"] = False
            self.args.gradient_checkpointing_kwargs = checkpoint_kwargs
        if self.args.label_smoothing_factor:
            raise ValueError("Stage I uses the model's grounding NLL; label_smoothing_factor must be 0.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs, use_cache=False, return_dict=True)
        loss = outputs.loss
        mode = "train" if model.training else "eval"
        for name in ("loss_gnd", "loss_apa"):
            self._stage1_metrics[mode][name].append(outputs[name].detach())
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        mode = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        metrics = getattr(self, "_stage1_metrics", {}).get(mode, {})
        logs = dict(logs)
        for name, values in metrics.items():
            if values:
                mean = torch.stack(values).mean().reshape(1)
                logs[("eval_" if mode == "eval" else "") + name] = (
                    self.accelerator.gather(mean).mean().item()
                )
                values.clear()
        return super().log(logs, start_time)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        # PEFT saves adapter_config + modules_to_save, but not the base config.
        # Persist APA settings/lambda at intermediate checkpoints as well.
        self.model.config.save_pretrained(output_dir or self.args.output_dir)
