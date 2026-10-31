"""Candidate queries and bbox targets for Stage I grounding conversations."""

import ast
import re

import torch
from PIL import Image

from .adsa import DEFAULT_DISEASE_CATEGORIES, extract_disease_categories


BOX_PATTERN = re.compile(r"<box>(.*?)</box>", re.DOTALL | re.IGNORECASE)


def disease_queries_from_prompt(user_text):
    """Read candidate names from user text only, never from target labels."""
    if "[" in user_text:
        return extract_disease_categories(user_text)
    # Preserve the existing single-disease referring-expression dataset format.
    match = re.search(r"region this sentence describes:\s*(.+)", user_text, re.IGNORECASE)
    if match:
        name = re.split(r",\s*position\s*:", match.group(1), flags=re.IGNORECASE)[0].strip().rstrip(".")
        if not name:
            raise ValueError("The referring expression contains no disease name.")
        return [name]
    return list(DEFAULT_DISEASE_CATEGORIES)


def bbox_targets_from_response(response, categories, image_size, coordinate_format="pixel"):
    """Parse Disease<box>[[xyxy], ...]</box> into normalized boxes/query indices.

    Matching is exact and case-insensitive, allowing the existing ', position:'
    suffix. Missing boxes for negative candidates do not produce fake targets.
    Invalid/unmatched annotations fail rather than silently removing L_APA.
    """
    # Some legacy grounding JSON files store literal backslash-n separators.
    response = response.replace("\\n", "\n")
    if coordinate_format not in ("pixel", "normalized", "qwen_1000"):
        raise ValueError("bbox_coordinate_format must be pixel, normalized or qwen_1000.")
    lookup = {name.casefold(): i for i, name in enumerate(categories)}
    boxes, indices = [], []
    cursor = 0
    matches = list(BOX_PATTERN.finditer(response))
    if not matches:
        if response.strip().casefold().rstrip(".") not in (
            "", "[]", "none", "normal", "no abnormalities", "no abnormality", "no findings",
        ):
            raise ValueError("Grounding answer must contain Disease<box>[[x1,y1,x2,y2]]</box> or an explicit negative.")
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(0, dtype=torch.long)
    for match in matches:
        label = response[cursor:match.start()].strip().strip(";, ")
        label = re.split(r",\s*position\s*:", label, flags=re.IGNORECASE)[0].strip()
        if label.casefold() not in lookup:
            raise ValueError(f"Annotated disease {label!r} is not a candidate in the user prompt.")
        try:
            parsed = ast.literal_eval(match.group(1).strip())
            tensor = torch.tensor(parsed, dtype=torch.float32)
        except (ValueError, SyntaxError, TypeError) as error:
            raise ValueError(f"Invalid box annotation for {label!r}.") from error
        if tensor.shape == (4,):
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != 4 or len(tensor) == 0:
            raise ValueError(f"Box annotation for {label!r} must have shape (N,4), N > 0.")
        boxes.append(tensor)
        indices.extend([lookup[label.casefold()]] * len(tensor))
        cursor = match.end()
    if response[cursor:].strip().strip(";, "):
        raise ValueError("Unexpected text after the last grounding box.")
    boxes = torch.cat(boxes)
    if coordinate_format == "pixel":
        width, height = image_size
        if min(width, height) <= 0:
            raise ValueError("Image dimensions must be positive.")
        boxes = boxes / boxes.new_tensor([width, height, width, height])
    elif coordinate_format == "qwen_1000":
        boxes = boxes / 1000
    if (not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any()
            or (boxes[:, 2:] < boxes[:, :2]).any()):
        raise ValueError("Invalid xyxy box or wrong bbox_coordinate_format for this image.")
    return boxes, torch.tensor(indices, dtype=torch.long)


class GroundingDataCollator:
    """Assistant-only language labels plus per-image APA bbox supervision.

    Currently requires one image and one user/assistant turn per supervised
    sample, matching the repository's grounding JSON datasets. Image errors
    raise immediately; blank replacements would attach real boxes to fake data.
    The processor and settings live on this object, so dataloader workers do
    not depend on a process-global processor.
    """

    def __init__(self, processor, use_adsa=True, bbox_coordinate_format="pixel"):
        self.processor = processor
        self.use_adsa = use_adsa
        self.bbox_coordinate_format = bbox_coordinate_format

    def __call__(self, examples):
        texts, prompts, images = [], [], []
        query_lists, boxes, indices = [], [], []
        for example in examples:
            messages = example["messages"]
            if (len(example["images"]) != 1 or len(messages) != 2
                    or messages[0]["role"] != "user" or messages[1]["role"] != "assistant"):
                raise ValueError("Grounding SFT expects one image and one user/assistant pair per sample.")
            user, answer = messages[0]["content"], messages[1]["content"]
            if user.count("<image>") != 1:
                raise ValueError("User content must contain exactly one <image> marker.")
            with Image.open(example["images"][0]) as source:
                image = source.convert("RGB")
            parts = []
            for segment in re.split(r"(<image>)", user):
                if segment == "<image>":
                    parts.append({"type": "image"})
                elif segment:
                    parts.append({"type": "text", "text": segment})
            conversation = [
                {"role": "user", "content": parts},
                {"role": "assistant", "content": answer},
            ]
            texts.append(self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False))
            prompts.append(self.processor.apply_chat_template(conversation[:1], tokenize=False, add_generation_prompt=True))
            images.append([image])
            if self.use_adsa:
                names = disease_queries_from_prompt(user)
                target_boxes, target_indices = bbox_targets_from_response(
                    answer, names, image.size, self.bbox_coordinate_format,
                )
                query_lists.append(names)
                boxes.append(target_boxes)
                indices.append(target_indices)
        kwargs = dict(images=images, return_tensors="pt", padding=True, padding_side="right")
        batch = self.processor(text=texts, **kwargs)
        prompt_batch = self.processor(text=prompts, **kwargs)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        image_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        labels[labels == image_id] = -100
        for i, length in enumerate(prompt_batch["attention_mask"].sum(-1).tolist()):
            # Refuse a template/tokenization mismatch instead of training on prompt text.
            if not torch.equal(batch["input_ids"][i, :length], prompt_batch["input_ids"][i, :length]):
                raise ValueError("Full conversation does not start with the tokenized generation prompt.")
            labels[i, :length] = -100
        batch["labels"] = labels
        if self.use_adsa:
            batch["disease_queries"] = query_lists
            batch["boxes_xyxy"] = boxes
            batch["category_indices"] = indices
        return batch
