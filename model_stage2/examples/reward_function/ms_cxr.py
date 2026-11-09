"""
Medical Imaging Multi-Task Reward Function for MS-CXR Dataset

Supports original dataset format:
{
    "dicom_id": "...",
    "label_text": ["answer1", "answer2"],  # list of possible answers
    "bbox": [["x1", "y1", "x2", "y2"], ...],  # string coordinates
    "path": "/path/to/image.jpg",
    "questions": ["question1", "question2"]  # list of possible questions
}

Evaluates:
1. Bounding Box Accuracy (IoU-based)
2. Answer Accuracy (semantic similarity)
3. Format Compliance (structured output)
4. Reasoning Quality (optional)
"""

import re
import json
import sys as _sys
# 直接 `python ms_cxr.py` 运行时，脚本所在目录会被加到 sys.path[0]，其中的本地 math.py
# 会遮蔽标准库 math，导致 scipy/numpy 导入失败。训练时用 importlib 加载不受影响，
# 这里仅为让 __main__ 自测可直接运行：移除脚本目录优先项。
if _sys.path and _sys.path[0] in ("", __file__.rsplit("/", 1)[0]):
    _sys.path.pop(0)
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import linear_sum_assignment
import numpy as np
import math

img_w = img_h =256

# ============================================================================
# bbox-reasoning 对齐 reward 所需的视觉 verifier V(crop, text) ∈ [0, 1]
# ----------------------------------------------------------------------------
# 三个新增 reward（local_support / contrastive_align / counterfactual）都依赖一个
# 能判断「图像 crop 是否支持一段 reasoning 文本」的视觉打分器。这里用 BiomedCLIP
# （胸片/医学图文对比模型）做 image-text 相似度。
#
# 部署方式：进程内 lazy singleton（reward worker 第一次用到时加载，常驻 GPU）。
# 性能优化（独立服务 / batch 推理）留作后续。
# 若模型无法加载（离线、缺依赖），verifier 退化为返回 0.5 的占位实现并打 warning，
# 保证训练流程不崩。
# ============================================================================

# bbox 坐标所基于的画布尺寸。gt bbox 与模型输出 bbox 都按这个尺寸标注（见文件顶部 img_w/img_h）。
# crop 时若真实图像尺寸不同，会按比例重映射到真实像素。
_BBOX_CANVAS = 256

_VERIFIER = None          # (model, preprocess, tokenizer) 或 "unavailable"
_VERIFIER_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def get_verifier():
    """Lazy-load BiomedCLIP 单例。失败时返回 None（调用方退化为占位分数）。"""
    global _VERIFIER
    if _VERIFIER is None:
        try:
            import torch
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(_VERIFIER_MODEL_NAME)
            tokenizer = open_clip.get_tokenizer(_VERIFIER_MODEL_NAME)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.eval().to(device)
            _VERIFIER = (model, preprocess, tokenizer, device)
            print(f"[ms_cxr reward] Loaded BiomedCLIP verifier on {device}.")
        except Exception as e:  # 离线 / 缺依赖 / 下载失败
            print(f"[ms_cxr reward] WARNING: failed to load BiomedCLIP verifier ({e}). "
                  f"Falling back to placeholder score 0.5 — alignment rewards will be uninformative.")
            _VERIFIER = "unavailable"

    return None if _VERIFIER == "unavailable" else _VERIFIER


@lru_cache(maxsize=64)
def _load_image(path: str):
    """按路径加载整图（RGB），LRU 缓存避免同一 batch 内重复 IO。失败返回 None。"""
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception as e:
        print(f"[ms_cxr reward] WARNING: failed to load image {path}: {e}")
        return None


def _remap_bbox_to_image(bbox, img_w_real, img_h_real):
    """把基于 _BBOX_CANVAS(256) 标注的 bbox 映射到真实图像像素坐标，并做边界裁剪。"""
    sx = img_w_real / _BBOX_CANVAS
    sy = img_h_real / _BBOX_CANVAS
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((x1 * sx, x2 * sx))
    y1, y2 = sorted((y1 * sy, y2 * sy))
    x1 = int(max(0, min(x1, img_w_real - 1)))
    y1 = int(max(0, min(y1, img_h_real - 1)))
    x2 = int(max(x1 + 1, min(x2, img_w_real)))
    y2 = int(max(y1 + 1, min(y2, img_h_real)))
    return x1, y1, x2, y2


def _crop(image, bbox):
    """从整图按 bbox（256 画布坐标）裁剪出 crop。image 为 PIL.Image。"""
    x1, y1, x2, y2 = _remap_bbox_to_image(bbox, image.width, image.height)
    return image.crop((x1, y1, x2, y2))


# def _mask_crop(crop):
#     """对 crop 做反事实遮挡：高斯模糊掉病灶信息，使其不再支持原 reasoning。"""
#     try:
#         from PIL import ImageFilter

#         # 半径取 crop 短边的一半，足以抹掉局部纹理/病灶
#         radius = max(8, min(crop.width, crop.height) // 2)
#         return crop.filter(ImageFilter.GaussianBlur(radius=radius))
#     except Exception:
#         return crop


def verify(crop, text: str) -> float:
    """V(crop, text) ∈ [0, 1]：crop 是否支持 reasoning text。

    用 BiomedCLIP 算图文 cosine 相似度，再线性映射到 [0,1]：(cos + 1) / 2。
    注意：故意【不】用陡峭 sigmoid 压缩——CLIP 类模型的图文 cosine 落在较窄正区间，
    陡峭 sigmoid 会把所有分数推到接近 1 而饱和，导致 contrastive reward 失去区分度。
    线性映射保留分数间差异，对 contrastive（看相对大小）和 RL 组内归一化都更友好。
    verifier 不可用时返回 0.5（中性占位）。
    """
    verifier = get_verifier()
    if verifier is None or crop is None or not text:
        return 0.5

    try:
        import torch

        model, preprocess, tokenizer, device = verifier
        image_input = preprocess(crop).unsqueeze(0).to(device)
        text_input = tokenizer([text], context_length=256).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            cos_sim = (image_features * text_features).sum(dim=-1).item()  # [-1, 1]

        # cosine -> [0,1] 线性映射，保留差异（不饱和）
        return float(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)))
    except Exception as e:
        print(f"[ms_cxr reward] WARNING: verify() failed ({e}); returning 0.5.")
        return 0.5

# def quantize_bbox(box, W, H, grid_size=32):   # 离散化坐标, 每个坐标会变成0~31中的数字
#     x1, y1, x2, y2 = box
    
#     return [
#         int(x1 / W * grid_size),
#         int(y1 / H * grid_size),
#         int(x2 / W * grid_size),
#         int(y2 / H * grid_size)
#     ]

# def dequantize_bbox(box, W, H, grid_size=32):  # 反量化，用于reward，会变成原来的pixels度量
#     x1, y1, x2, y2 = box
    
#     return [
#         x1 / grid_size * W,
#         y1 / grid_size * H,
#         x2 / grid_size * W,
#         y2 / grid_size * H
#     ]

def box_area(box):
    w = max(0, box[2] - box[0])
    h = max(0, box[3] - box[1])
    return w * h
def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)

def cal_center_score(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2

    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    return math.exp(-dist / 50)


# def size_score(box1, box2):
#     w1 = max(0, box1[2] - box1[0])
#     h1 = max(0, box1[3] - box1[1])

#     w2 = max(0, box2[2] - box2[0])
#     h2 = max(0, box2[3] - box2[1])

#     w_ratio = min(w1, w2) / (max(w1, w2) + 1e-6)
#     h_ratio = min(h1, h2) / (max(h1, h2) + 1e-6)

#     return (w_ratio + h_ratio) / 2

def cal_size_score(box1, box2):  # 改进后的size_score
    w1 = max(1e-6, box1[2] - box1[0])
    h1 = max(1e-6, box1[3] - box1[1])

    w2 = max(1e-6, box2[2] - box2[0])
    h2 = max(1e-6, box2[3] - box2[1])

    return np.sqrt(
        (min(w1, w2) / max(w1, w2)) *
        (min(h1, h2) / max(h1, h2))
    )


def compute_matching_score(pred_bboxes, gt_bboxes, img_diag):  # 这里用离散化的bbox坐标
    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return 0.0

    cost_matrix = np.zeros((len(pred_bboxes), len(gt_bboxes)))

    for i, p in enumerate(pred_bboxes):
        for j, g in enumerate(gt_bboxes):
            # print("@@@@@ p:", p, "g:", g)
            # 添加的代码
            # p_degrid = dequantize_bbox(p, img_w, img_h)
            # g_grid = quantize_bbox(g, img_w, img_h)

            iou = calculate_iou(p, g)  # 原代码
            # iou = calculate_iou(p_degrid, g)  # 用原始的计算iou

            # dist = center_distance(p, g) # 原代码
            # dist = center_distance(p_degrid, g) # 用原始的计算距离
            # center_score = 1 - dist / (img_diag + 1e-6) 
            # center_score = max(center_score, 0)
            # center_score = cal_center_score(p_degrid, g) # 用原始的计算距离
            center_score = cal_center_score(p, g) # 用原始的计算距离,计算中心点的距离


            size = cal_size_score(p, g) # 原代码
            # size = cal_size_score(p_degrid, g) # 用原始的计算size


            score = 0.5 * iou + 0.3 * center_score + 0.2 * size # 原代码
            # 先用iou计算，
            # score = iou
            cost_matrix[i, j] = 1 - score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    scores = []
    for i, j in zip(row_ind, col_ind):
        s = 1 - cost_matrix[i, j]
        scores.append(s)

    return np.mean(scores) if scores else 0.0

# def compute_count_score(pred_bboxes, gt_bboxes):  # 这里不影响
#     n_pred = len(pred_bboxes)
#     n_gt = len(gt_bboxes)

#     if n_gt == 0:
#         return 1.0 if n_pred == 0 else 0.0

#     return 1 - abs(n_pred - n_gt) / n_gt

# 改进的count score
def compute_count_score(pred_bboxes, gt_bboxes):
    diff = abs(len(pred_bboxes) - len(gt_bboxes))
    return math.exp(-diff)

def compute_coverage_score(pred_bboxes, gt_bboxes):  # 这里用原始的bbox坐标
    if len(gt_bboxes) == 0:
        return 1.0

    # cover_count = 0
    total = 0
    for gt in gt_bboxes:
        best = 0
        for pred in pred_bboxes:
            # gt_grid = quantize_bbox(gt, img_w, img_h)
            # pred_degrid = dequantize_bbox(pred, img_w, img_h)
            # iou = calculate_iou(pred, gt) # 原代码
            # iou = calculate_iou(pred_degrid, gt) # 原代码
                #         if iou > best_iou:
    #             best_iou = iou

    #     if best_iou > 0.3:
    #         cover_count += 1

    # return cover_count / len(gt_bboxes)
            # inter = intersection_area(pred_degrid, gt)
            inter = intersection_area(pred, gt) # 原代码
            best = max(best, inter / (box_area(gt) + 1e-6))
            
        total += best

    return total / len(gt_bboxes)

# 计算bbox的函数太复杂，现在只计算iou
def compute_bbox_set_score(pred_bboxes, gt_bboxes, img_w, img_h):
    img_diag = np.sqrt(img_w**2 + img_h**2)

    matching = compute_matching_score(pred_bboxes, gt_bboxes, img_diag) # 计算了iou, center, size的综合score
    # count = compute_count_score(pred_bboxes, gt_bboxes)
    # coverage = compute_coverage_score(pred_bboxes, gt_bboxes)

    # score = (
    #     0.5 * matching +
    #     0.3 * count +
    #     0.2 * coverage
    # )

    # return max(score, 0.0)
    return matching  # 先只用matching score作为bbox的reward


# ============================================================================
# bbox–reasoning 对齐：解析 per-bbox 配对输出 + 三个对齐 reward
# ============================================================================

def _normalise_region(bbox, reasoning, diagnosis="", **extra) -> Optional[Dict]:
    """Validate one region without executing generated text."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        coords = [float(v) for v in bbox]
    except (ValueError, TypeError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in coords):
        return None
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1 or not isinstance(reasoning, str) or not reasoning.strip():
        return None
    return {"bbox": coords, "reasoning": reasoning.strip(),
            "diagnosis": str(diagnosis).strip(), **extra}


def _natural_fields(block: str) -> Dict[str, str]:
    """Read labelled fields, including multiline reasoning and Markdown labels."""
    pattern = re.compile(
        r"(?im)^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?"
        r"(Bounding[ \t]*Box|BBox|Diagnosis|Suspected[ \t]+diagnosis|Reasoning|"
        r"Priority[ \t]+Level|Region[ \t]+ID)"
        r"(?:\*\*)?[ \t]*[:：][ \t]*(?:\*\*)?"
    )
    matches = list(pattern.finditer(block))
    fields = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        name = re.sub(r"\s+", " ", match.group(1).lower()).strip()
        fields[name] = block[match.end():end].strip()
    return fields


def _field_bbox(value: str):
    match = re.search(r"\[([^\[\]]+)\]", value)
    if not match:
        return None
    parts = match.group(1).split(",")
    if len(parts) != 4:
        return None
    try:
        return [float(v.strip()) for v in parts]
    except (ValueError, OverflowError):
        return None


def _parse_region_analyses(solution: str) -> List[Dict]:
    pattern = re.compile(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:-{2,}[ \t]*)?"
        r"Region[ \t]+(\d+)[ \t]+Analysis\b[^\n]*"
    )
    headings = list(pattern.finditer(solution))
    regions = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(solution)
        block = solution[heading.end():end]
        final = re.search(r"(?im)^[ \t]*(?:-{2,}[ \t]*)?(?:#{1,6}[ \t]*)?Final Summary\b", block)
        if final:
            block = block[:final.start()]
        fields = _natural_fields(block)
        step = re.search(r"(?im)^[ \t]*(?:\*\*)?Step[ \t]*1[.:：]", block)
        reasoning = block[step.start():].strip() if step else fields.get("reasoning", "")
        region = _normalise_region(
            _field_bbox(fields.get("bounding box", fields.get("bbox", ""))),
            reasoning, fields.get("diagnosis", fields.get("suspected diagnosis", "")),
            region_id=f"BBox {heading.group(1)}",
        )
        if region:
            regions.append(region)
    return regions


def _parse_priority_output(solution: str) -> List[Dict]:
    pattern = re.compile(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?"
        r"Priority[ \t]*\[?(\d+)\]?[ \t]*(?:\*\*)?[ \t]*[:：][ \t]*(?:\*\*)?"
    )
    headings = list(pattern.finditer(solution))
    regions = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(solution)
        fields = _natural_fields(solution[heading.end():end])
        extra = {"priority": int(heading.group(1))}
        if fields.get("region id"):
            extra["region_id"] = fields["region id"]
        region = _normalise_region(
            _field_bbox(fields.get("bounding box", fields.get("bbox", ""))),
            fields.get("reasoning", ""), fields.get("diagnosis", ""), **extra,
        )
        if region:
            regions.append(region)
    return regions


def parse_pairwise_output(solution: str) -> List[Dict]:
    """Parse regions JSON or Priority/Bounding Box/Diagnosis/Reasoning text.

    Detailed Region analyses and their priority summary are one set of regions,
    not two. When both occur, keep the summary order and final diagnosis while
    recovering the full reasoning by matching bbox coordinates. Priority rank
    is never treated as a stable region identifier.
    """
    if not isinstance(solution, str):
        return []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", solution):
        try:
            obj, _ = decoder.raw_decode(solution[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("regions"), list):
            continue
        regions = []
        for item in obj["regions"]:
            if not isinstance(item, dict):
                continue
            extra = {}
            if "region_id" in item:
                extra["region_id"] = str(item["region_id"])
            region = _normalise_region(item.get("bbox"), item.get("reasoning", ""),
                                       item.get("diagnosis", ""), **extra)
            if region:
                regions.append(region)
        if regions:
            return regions
    analyses = _parse_region_analyses(solution)
    priorities = _parse_priority_output(solution)
    if priorities:
        for region in priorities:
            for detail in analyses:
                if np.allclose(region["bbox"], detail["bbox"], rtol=0, atol=1e-6):
                    region["reasoning"] = detail["reasoning"]
                    region.setdefault("region_id", detail["region_id"])
                    break
        return priorities
    return analyses


def _verify_matrix(regions: List[Dict], image) -> np.ndarray:
    """计算 V[i, j] = V(crop_i, reasoning_j)，供 local_support / contrastive 复用。

    对角线 V[i, i] 即「bbox_i 是否支持 reasoning_i」。
    """
    n = len(regions)
    crops = [_crop(image, r["bbox"]) for r in regions]
    texts = [r["reasoning"] for r in regions]
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            mat[i, j] = verify(crops[i], texts[j])
    return mat


def compute_local_support(regions: List[Dict], image, verify_mat: Optional[np.ndarray] = None) -> float:
    """R_local_support = mean_i V(crop_i, reasoning_i)。

    保证每段 reasoning 至少被它绑定的 bbox 支持。无 region / 无图返回 0。
    """
    if not regions or image is None:
        return 0.0
    if verify_mat is None:
        verify_mat = _verify_matrix(regions, image)
    return float(np.mean(np.diag(verify_mat)))


def compute_contrastive_align(
    regions: List[Dict],
    image,
    margin: float = 0.1,
    verify_mat: Optional[np.ndarray] = None,
    mode: str = "symmetric_log",
    tau: float = 0.07,
) -> float:
    """Paper Eq. (17): mean symmetric log-probability of assigned pairs.

    verify_mat[crop, text] contains (cosine + 1) / 2. Recover raw cosine
    before applying tau. Row and column matching are averaged equally.
    N=1 returns zero. Values are non-positive and are NOT mapped to [0,1].
    The old name mode="softmax" is accepted as an alias for this corrected
    objective, not for the former probability-only reward. Explicit "margin"
    remains a legacy alternative, not the paper objective.
    """
    if not regions or image is None:
        return 0.0
    if mode not in ("symmetric_log", "softmax", "margin"):
        raise ValueError(f"Unsupported cross-instance reward mode: {mode}")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Cross-instance temperature must be finite and positive.")
    if verify_mat is None:
        verify_mat = _verify_matrix(regions, image)
    n = len(regions)
    mat = np.asarray(verify_mat, dtype=np.float64)
    if mat.shape != (n, n) or not np.all(np.isfinite(mat)):
        raise ValueError("verify_mat must be a finite N x N matrix.")
    if mode == "margin":
        if n == 1:
            return float(mat[0, 0])
        return float(np.mean([
            max(0.0, mat[i, i] - max(mat[j, i] for j in range(n) if j != i) - margin)
            for i in range(n)
        ]))
    if n == 1:
        return 0.0
    logits = (2.0 * mat - 1.0) / tau
    def logsumexp(values, axis):
        maximum = np.max(values, axis=axis, keepdims=True)
        return np.squeeze(maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)), axis=axis)
    diagonal = np.diag(logits)
    text_to_region = diagonal - logsumexp(logits, axis=0)
    region_to_text = diagonal - logsumexp(logits, axis=1)
    return float(np.mean(text_to_region + region_to_text) / 2.0)


# def compute_counterfactual(regions: List[Dict], image) -> float:
#     """R_counterfactual（仅 mask 项）= mean_i (1 - V(mask(crop_i), reasoning_i))。

#     遮挡掉 bbox_i 的病灶信息后，原 reasoning 不应再被支持；若仍被支持说明模型在 hallucinate。

#     注意：原设计的 swap / replace 两项需要对编辑后的图像【重跑策略模型】得到 ŷ_i^swap / ŷ_i^replace，
#     而本 reward 接口只拿到解码后的文本、无法调用策略模型，因此这两项无法在此实现，
#     需作为离线数据增强 pipeline 单独构造。这里只实现 verifier-only 的 mask 项。
#     """
#     if not regions or image is None:
#         return 0.0
#     scores = []
#     for r in regions:
#         crop = _crop(image, r["bbox"])
#         masked = _mask_crop(crop)
#         scores.append(1.0 - verify(masked, r["reasoning"]))
#     return float(np.mean(scores))


def compute_score(
    solution: str,
    ground_truth: Union[str, List[str]],  # Can be list or single string
    data_format: str = "original",  # "original" for your dataset format
    gt_bounding_boxes: Optional[Union[List[List[str]], List[List[int]]]] = None,
    image=None,  # 图像路径(str) 或 PIL.Image，用于 bbox-reasoning 对齐 reward 的 crop
    weights: Optional[Dict[str, float]] = None,  # 各分项权重，None 用默认
    contrastive_margin: float = 0.1,
    contrastive_mode: str = "symmetric_log",  # Paper Eq. (17); softmax is a compatibility alias.
    contrastive_tau: float = 0.07,  # Temperature applied to RAW cosine similarity.
    # img_w: int = 256,
    # img_h: int = 256,
    **kwargs
) -> Dict[str, float]:
    """
    Compute multi-task reward for medical imaging

    Args:
        solution: Model's complete output (including bboxes, thinking, answer)
        ground_truth: Ground truth label (can be string or list of strings)
        data_format: "original" for your dataset, "standard" for standard format
        gt_bounding_boxes: List of ground truth bounding boxes [["x1","y1","x2","y2"], ...] or [[x1,y1,x2,y2], ...]
        image: 图像路径或 PIL.Image；缺失时三个对齐 reward 记 0
        weights: 各分项权重 dict（answer/format/local_support/contrastive/counterfactual）
        contrastive_margin: contrastive reward 的 margin
        **kwargs: Other arguments

    Returns:
        dict，含 overall 与各分项分数（便于训练侧聚合 metrics）
    """

    # Handle ground truth format
    if isinstance(ground_truth, list):
        if len(ground_truth) > 0:
            gt_text = ground_truth[0]  # Take first answer if multiple provided
        else:
            gt_text = ""
    else:
        gt_text = ground_truth

    parsed = parse_model_output(solution)

    # 1. Bounding Box scoring (IoU)
    bbox_score = 0.0
    if gt_bounding_boxes is not None and len(gt_bounding_boxes) > 0:
        # Convert string coordinates to integers if needed
        gt_boxes_converted = []
        for bbox in gt_bounding_boxes:
            if isinstance(bbox[0], str):  # String coordinates
                converted_bbox = [int(float(coord)) for coord in bbox]
            else:  # Integer coordinates
                converted_bbox = [int(coord) for coord in bbox]
            gt_boxes_converted.append(converted_bbox)

        bbox_score = compute_bbox_set_score(parsed['bounding_boxes'], gt_boxes_converted, img_w=img_w, img_h=img_h)

    # 2. Answer accuracy scoring
    answer_score = compute_answer_score(parsed['answer'], gt_text)

    # 3. Format compliance scoring
    format_score = compute_format_score(solution)

    # 4. bbox-reasoning 对齐 reward（依赖图像 crop + BiomedCLIP verifier）
    #    解析 per-bbox 配对输出；无图或解析失败时全部记 0。
    pil_image = _load_image(image) if isinstance(image, str) else image
    regions = parse_pairwise_output(solution)

    local_support = 0.0
    contrastive = 0.0
    counterfactual = 0.0
    if regions and pil_image is not None:
        # V[i,j] = V(crop_i, reasoning_j)，local_support 与 contrastive 共用，避免重复推理
        verify_mat = _verify_matrix(regions, pil_image)
        local_support = compute_local_support(regions, pil_image, verify_mat=verify_mat)
        contrastive = compute_contrastive_align(
            regions, pil_image, margin=contrastive_margin,
            mode=contrastive_mode, tau=contrastive_tau, verify_mat=verify_mat,
        )
        # counterfactual = compute_counterfactual(regions, pil_image)

    # Combine scores with weights
    default_weights = {
        # "answer": 0.15,
        # "format": 0.25,
        "local_support": 0.50,
        "contrastive": 0.50,   # 一一对应最核心，给最高权重
        # "counterfactual": 0.20,
    }
    w = {**default_weights, **(weights or {})}

    final_score = (
        # w["answer"] * answer_score +
        # w["format"] * format_score +
        w["local_support"] * local_support +
        w["contrastive"] * contrastive 
        # w["counterfactual"] * counterfactual
    )

    return {
        "overall": float(min(final_score, 1.0)),
        "answer": float(answer_score),
        # alias：online_filtering 用 filter_key="accuracy"，必须存在此 key（见 ray_trainer.py）
        "accuracy": float(answer_score),
        "format": float(format_score),
        "bbox": float(bbox_score),
        "local_support": float(local_support),
        "contrastive": float(contrastive),
        "counterfactual": float(counterfactual),
    }


def parse_model_output(solution: str) -> Dict:
    """
    Parse structured output from model

    Expected JSON format:
    {
      "consensus_region": [x_min, y_min, x_max, y_max],
      "reasoning_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ...", "Step 4: ..."],
      "diagnosis": "Disease name",
      "evidence_confidence": "Confidence score"
    }

    Also tries to extract the 5 candidate bounding boxes from Task 1 output.
    """

    result = {
        'bounding_boxes': [],
        'confidences': [],
        'thinking': '',
        'answer': ''
    }

    # Try to extract JSON block from the output
    json_obj = None
    json_match = re.search(r'\{[^{}]*"consensus_region"[^{}]*\}', solution, re.DOTALL)
    if json_match:
        try:
            json_obj = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # If direct match failed, try to find any JSON object with relevant keys
    if json_obj is None:
        json_blocks = re.findall(r'\{[^{}]+\}', solution)
        for block in json_blocks:
            try:
                obj = json.loads(block)
                if any(k in obj for k in ['consensus_region', 'diagnosis', 'reasoning_steps']):
                    json_obj = obj
                    break
            except json.JSONDecodeError:
                continue

    # Extract consensus_region as the primary bounding box
    if json_obj and 'consensus_region' in json_obj:
        bbox = json_obj['consensus_region']
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [int(float(c)) for c in bbox]
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                    result['bounding_boxes'].append([x1, y1, x2, y2])
            except (ValueError, TypeError):
                pass

    # Also try to extract the 5 candidate bounding boxes from Task 1
    candidates_match = re.search(r'\[\s*\[(\d+)', solution)
    if candidates_match:
        bbox_array_match = re.search(r'\[\s*\[\d+.*?\]\s*\]', solution, re.DOTALL)
        if bbox_array_match:
            try:
                candidates = json.loads(bbox_array_match.group())
                if isinstance(candidates, list) and len(candidates) > 0:
                    for bbox in candidates:
                        if isinstance(bbox, list) and len(bbox) == 4:
                            x1, y1, x2, y2 = [int(float(c)) for c in bbox]
                            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                                if [x1, y1, x2, y2] not in result['bounding_boxes']:
                                    result['bounding_boxes'].append([x1, y1, x2, y2])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    # Extract confidence
    if json_obj and 'evidence_confidence' in json_obj:
        try:
            conf = float(json_obj['evidence_confidence'])
            result['confidences'] = [min(max(conf, 0.0), 1.0)] * len(result['bounding_boxes'])
        except (ValueError, TypeError):
            result['confidences'] = [0.5] * len(result['bounding_boxes'])
    else:
        result['confidences'] = [0.5] * len(result['bounding_boxes'])

    # Extract reasoning_steps as thinking
    if json_obj and 'reasoning_steps' in json_obj:
        steps = json_obj['reasoning_steps']
        if isinstance(steps, list):
            result['thinking'] = '\n'.join(steps)
        elif isinstance(steps, str):
            result['thinking'] = steps

    # Extract diagnosis as answer
    if json_obj and 'diagnosis' in json_obj:
        result['answer'] = str(json_obj['diagnosis'])

    return result


def compute_bbox_score(
    pred_boxes: List[List[int]],
    gt_boxes: List[List[int]],
    iou_threshold: float = 0.3  # Lower threshold for medical imaging
) -> float:
    """
    Calculate IoU score for bounding boxes
    
    Args:
        pred_boxes: Predicted bboxes [[x1,y1,x2,y2], ...]
        gt_boxes: Ground truth bboxes
        iou_threshold: IoU threshold for correct detection
    
    Returns:
        bbox detection accuracy (0.0-1.0)
    """
    
    if not gt_boxes:
        # If no ground truth, model shouldn't predict anything
        return 1.0 if not pred_boxes else 0.0
    
    if not pred_boxes:
        # Model predicted no boxes but there are ground truth
        return 0.0
    
    # Calculate max IoU for each predicted box to ground truth boxes
    true_positives = 0
    for pred_box in pred_boxes:
        max_iou = 0.0
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)
        
        if max_iou >= iou_threshold:
            true_positives += 1
    
    # Consider precision and recall
    precision = true_positives / len(pred_boxes) if pred_boxes else 0.0
    recall = true_positives / len(gt_boxes) if gt_boxes else 0.0
    
    # F1 Score
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU (Intersection over Union) between two bboxes"""
    
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_answer_score(pred_answer: str, gt_answer: str) -> float:
    """
    Calculate semantic similarity between predicted and ground truth answers
    
    Uses medical term overlap + basic text similarity
    """
    
    # ---- FIX: convert numpy/list to string ----
    import numpy as np
    
    if isinstance(gt_answer, (list, tuple)):
        gt_answer = gt_answer[0]

    if isinstance(gt_answer, np.ndarray):
        gt_answer = gt_answer.flatten()[0]

    if isinstance(pred_answer, (list, tuple)):
        pred_answer = pred_answer[0]

    if isinstance(pred_answer, np.ndarray):
        pred_answer = pred_answer.flatten()[0]
    # ------------------------------------------

    if not pred_answer or not gt_answer:
        return 0.0
    
    # Normalize text
    pred_answer = str(pred_answer).lower().strip()
    gt_answer = str(gt_answer).lower().strip()

    # print("GT:", gt_answer)
    # print("PRED:", pred_answer)
    # Extract medical terms (simple version, can be enhanced with UMLS)
    def extract_medical_terms(text: str) -> set:
        # Common medical patterns
        medical_patterns = [
            r'\b(pneumonia|effusion|pneumothorax|consolidation|infiltrate)\b',
            r'\b(cardiomegaly|enlarged|hypertrophy)\b',
            r'\b(fracture|break|lesion|mass|nodule|calcification)\b',
            r'\b(normal|clear|unremarkable)\b',
            r'\b(left|right|upper|lower|lobe|apical|basal)\b',
            r'\b(tiny|small|moderate|large)\b',
            r'\b(finding|observed|seen|present|absent)\b',
        ]
        
        terms = set()
        for pattern in medical_patterns:
            matches = re.findall(pattern, text)
            terms.update(matches)
        
        # Also add general terms (remove stopwords)
        words = re.findall(r'\b[a-z]+\b', text)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'this', 'that', 'these', 'those', 'in', 'on', 'at', 'to', 'for',
                     'with', 'by', 'from', 'of', 'and', 'or', 'but', 'so', 'if', 'then'}
        terms.update([w for w in words if w not in stopwords and len(w) > 2])
        
        return terms
    
    pred_terms = extract_medical_terms(pred_answer)
    gt_terms = extract_medical_terms(gt_answer)
    
    if not gt_terms:
        return 0.0
    
    # Calculate term overlap
    overlap = len(pred_terms & gt_terms)
    recall = overlap / len(gt_terms)
    precision = overlap / len(pred_terms) if pred_terms else 0.0
    
    # F1 Score
    if recall + precision == 0:
        # Fall back to simple substring match
        if gt_answer in pred_answer or pred_answer in gt_answer:
            return 1.0
        return 0.0
    
    f1 = 2 * (recall * precision) / (recall + precision)
    return min(f1, 1.0)


def compute_format_score(solution: str) -> float:
    """
    Evaluate output format compliance for JSON output format.

    Expected Task 2 JSON:
    {
      "consensus_region": [x_min, y_min, x_max, y_max],
      "reasoning_steps": ["Step 1: ...", ...],
      "diagnosis": "...",
      "evidence_confidence": "..."
    }
    Also checks that Task 1 candidate bboxes are present.
    """

    score = 0.0
    max_score = 5.0

    # Check Task 1: candidate bounding boxes array (e.g. [[x,y,x,y], ...])
    if re.search(r'\[\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]', solution):
        score += 1.0

    # Try to parse the Task 2 JSON block
    json_obj = None
    json_match = re.search(r'\{[^{}]*"consensus_region"[^{}]*\}', solution, re.DOTALL)
    if json_match:
        try:
            json_obj = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Check consensus_region: valid [x, y, x, y] list
    if json_obj and 'consensus_region' in json_obj:
        bbox = json_obj['consensus_region']
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [float(c) for c in bbox]
                if x2 > x1 and y2 > y1:
                    score += 1.0
            except (ValueError, TypeError):
                pass

    # Check reasoning_steps: non-empty list
    if json_obj and 'reasoning_steps' in json_obj:
        steps = json_obj['reasoning_steps']
        if isinstance(steps, list) and len(steps) > 0:
            score += 1.0

    # Check diagnosis: non-empty string
    if json_obj and 'diagnosis' in json_obj:
        if isinstance(json_obj['diagnosis'], str) and json_obj['diagnosis'].strip():
            score += 1.0

    # Check evidence_confidence: parseable as float in [0, 1]
    if json_obj and 'evidence_confidence' in json_obj:
        try:
            conf = float(json_obj['evidence_confidence'])
            if 0.0 <= conf <= 1.0:
                score += 1.0
        except (ValueError, TypeError):
            pass

    return score / max_score


# Batch evaluation function
def compute_batch_score(
    solutions: List[str],
    ground_truths: List[Union[str, List[str]]],
    gt_bounding_boxes_list: Optional[List[Union[List[List[str]], List[List[int]]]]] = None,
    **kwargs
) -> List[Dict[str, float]]:
    """Batch compute scores (每个元素是 compute_score 返回的分项 dict)"""

    scores = []
    for i, (sol, gt) in enumerate(zip(solutions, ground_truths)):
        gt_boxes = gt_bounding_boxes_list[i] if gt_bounding_boxes_list else None

        score = compute_score(
            sol,
            gt,
            gt_bounding_boxes=gt_boxes,
            **kwargs
        )
        scores.append(score)

    return scores


# Framework-compatible reward function
# 从训练代码跳进来，首先进入这个函数
def reward_function(reward_inputs: List[Dict], **kwargs) -> List[Dict]:
    """
    Framework-compatible reward function

    Args:
        reward_inputs: List of dicts with keys: response, response_length, ground_truth
        **kwargs: Additional arguments from config (e.g., data_format)

    Returns:
        List of dicts with keys: overall, format (optional), accuracy (optional)
    """
    results = []
    # img_w = kwargs.get('img_w', 256)  # 默认值从配置文件获取
    # img_h = kwargs.get('img_h', 256)  # 默认值从配置文件获取
    # reward_inputs[0] example: 
    # 'response' = "So, let's analyze the chest X-ray. First, look at the right lung base. The image shows patchy ground-glass opacities there. Need to find the bounding box. Let's check the coordinates. The right lung base is the lower part of the right lung. From the image, the area is between the diaphragm and the lower lung fields. So the bounding box would be around that region. Let's estimate. The right lung base has patchy opacities, so the coordinates might be [294, 629, 520, 739] or similar. Wait, need to be precise. Let's see: the right lung base is the lower part of the right lung. The opacities are in that area. So the bounding box for the right lung base patchy ground-glass opacities would be [294, 629, 520, 739] maybe. Confidence is high, so 0.9. Then, check if there are other findings, but the question is specifically about the right lung base. So first, identify the finding and its coordinates.\n</think>\n\n<bbox>[294, 629, 520, 739]</bbox> <confidence>0.90</confidence>\n<bbox>[349, 747, 474, 842]</bbox> <confid...
    # 'response_length' = 517
    # 'ground_truth' =  ['patchy ground-glass opacities at the right lung base']
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]

        # Parse ground truth to extract bounding boxes if available
        gt_bboxes = None
        gt_text = ground_truth
        image = None  # 图像路径，用于 bbox-reasoning 对齐 reward

        # If ground_truth is a dict with bbox field
        if isinstance(ground_truth, dict):
            gt_bboxes = ground_truth.get("bbox", None)
            gt_text = ground_truth.get("label_text", "")
            image = ground_truth.get("image", None)

        # Compute scores (返回 dict，含 overall 与各分项)
        score = compute_score(
            solution=response,
            ground_truth=gt_text,
            gt_bounding_boxes=gt_bboxes,
            image=image,
            **kwargs,
        )

        # 回传所有分项，function.py 会自动把每个 key 聚合进 reward_metrics
        results.append(score)

    return results


# 在 ms_cxr.py 文件末尾添加
# def reward_function_ray_ppo(
#     responses: List[str], 
#     ground_truths: List[Union[str, Dict]], 
#     contexts: Optional[List[Dict]] = None,
#     **kwargs
# ) -> List[float]:
#     """
#     Ray PPO compatible reward function that maintains all original functionality
    
#     Args:
#         responses: List of model responses from PPO rollout
#         ground_truths: List of ground truth values (may include bbox info in dict format)
#         contexts: Optional additional context containing bbox information
#         **kwargs: Additional parameters from PPO trainer
    
#     Returns:
#         List of scalar rewards (one per response)
#     """
#     rewards = []
    
#     for i, (response, gt) in enumerate(zip(responses, ground_truths)):
#         # Parse ground truth to extract bounding boxes if available
#         gt_bboxes = None
#         gt_text = gt
        
#         # If ground_truth is a dict with bbox field (MS-CXR format)
#         if isinstance(gt, dict):
#             gt_bboxes = gt.get("bbox", None)
#             gt_text = gt.get("label_text", "")
#             # Handle case where label_text is a list
#             if isinstance(gt_text, list) and len(gt_text) > 0:
#                 gt_text = gt_text[0]
        
#         # Also check contexts for bbox information if not found in ground_truth
#         if gt_bboxes is None and contexts and i < len(contexts):
#             context = contexts[i]
#             if isinstance(context, dict):
#                 gt_bboxes = context.get("bbox", None) or context.get("gt_bounding_boxes", None)
#                 if gt_text == gt and isinstance(context.get("label_text"), (str, list)):  # Only override if unchanged
#                     label_text = context.get("label_text", "")
#                     if isinstance(label_text, list) and len(label_text) > 0:
#                         gt_text = label_text[0]
#                     elif isinstance(label_text, str):
#                         gt_text = label_text
        
#         # Compute score using original function
#         score = compute_score(
#             solution=response,
#             ground_truth=gt_text,
#             gt_bounding_boxes=gt_bboxes
#         )
        
#         rewards.append(float(score))
    
#     return rewards


# def ray_ppo_reward_adapter(batch, tokenizer=None):
#     """
#     适配RayPPOTrainer的奖励函数接口
#     """
#     import torch
    
#     # 如果tokenizer未提供，需要从外部获取或创建
#     if tokenizer is None:
#         # 这里需要根据实际情况导入tokenizer
#         # 可能需要从训练配置中获取
#         print("Warning: tokenizer is None, please provide a valid tokenizer")
#         return torch.zeros((len(batch.batch["responses"]), 1)), {"error": ["tokenizer_missing"]}
    
#     responses = [tokenizer.decode(resp, skip_special_tokens=True) for resp in batch.batch["responses"]]
#     ground_truths = batch.non_tensor_batch["ground_truth"].tolist()
    
#     # 使用您的原始奖励函数计算分数
#     scores = reward_function_ray_ppo(responses, ground_truths)
    
#     # 转换为token-level scores（适用于每个response的token）
#     token_level_scores = []
#     for score, response_ids in zip(scores, batch.batch["responses"]):
#         # 计算有效token数量（排除padding）
#         valid_mask = (response_ids != tokenizer.pad_token_id)
#         token_len = valid_mask.sum().item()
        
#         if token_len > 0:
#             # 将总分均匀分配到每个有效token上
#             token_score_value = score / token_len
#             token_scores = torch.full((token_len,), token_score_value)
            
#             # 创建完整长度的张量，填充无效位置为0
#             full_scores = torch.zeros_like(response_ids, dtype=torch.float)
#             full_scores[valid_mask] = token_scores
#         else:
#             # 如果没有有效token，返回全零张量
#             full_scores = torch.zeros_like(response_ids, dtype=torch.float)
        
#         token_level_scores.append(full_scores)
    
#     # 堆叠成批次张量
#     token_level_tensor = torch.stack(token_level_scores)
    
#     # 返回token-level scores张量和指标字典
#     return token_level_tensor, {"overall_scores": scores, "num_responses": len(responses)}


# ============================================================================
# 自测：python ms_cxr.py [可选的真实图像路径]
#   - 不传图像路径时用合成图，验证解析 + 三个 reward 的代码路径与数值区间。
#   - verifier 若无法加载会退化为 0.5，分数仍应在 [0,1] 且不抛异常。
# ============================================================================
if __name__ == "__main__":
    import sys
    from PIL import Image

    # 1) 准备一张测试图（真实路径优先，否则合成 256x256）
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"Using real image: {test_image}")
    else:
        synthetic = Image.new("RGB", (256, 256), color=(40, 40, 40))
        # 左右两块不同灰度，模拟两个区域内容不同
        for x in range(0, 128):
            for y in range(128, 256):
                synthetic.putpixel((x, y), (200, 200, 200))
        test_image = synthetic
        print("Using synthetic 256x256 image.")

    # 2) 配对正确的输出：每个 region 的 reasoning 描述自己的框
    good_solution = json.dumps({
        "regions": [
            {"bbox": [10, 140, 120, 250], "reasoning": "Within this lower-left region there is a focal consolidation with increased opacity supporting pneumonia.", "diagnosis": "pneumonia"},
            {"bbox": [140, 10, 250, 120], "reasoning": "Within this upper-right region the lung field is clear with normal aeration.", "diagnosis": "normal"},
        ],
        "final_diagnosis": "pneumonia",
        "confidence": "0.80",
    })

    # 3) 故意串错：把两段 reasoning 交换，contrastive 应明显下降
    swapped_solution = json.dumps({
        "regions": [
            {"bbox": [10, 140, 120, 250], "reasoning": "Within this upper-right region the lung field is clear with normal aeration.", "diagnosis": "normal"},
            {"bbox": [140, 10, 250, 120], "reasoning": "Within this lower-left region there is a focal consolidation with increased opacity supporting pneumonia.", "diagnosis": "pneumonia"},
        ],
        "final_diagnosis": "pneumonia",
        "confidence": "0.80",
    })

    for name, sol in [("aligned", good_solution), ("swapped", swapped_solution)]:
        regions = parse_pairwise_output(sol)
        print(f"\n[{name}] parsed {len(regions)} regions")
        pil = _load_image(test_image) if isinstance(test_image, str) else test_image
        mat = _verify_matrix(regions, pil)
        ls = compute_local_support(regions, pil, verify_mat=mat)
        ca = compute_contrastive_align(regions, pil, verify_mat=mat)
        cf = compute_counterfactual(regions, pil)
        print(f"  local_support = {ls:.4f}")
        print(f"  contrastive   = {ca:.4f}")
        print(f"  counterfact   = {cf:.4f}")
        for v in (ls, ca, cf):
            assert 0.0 <= v <= 1.0, f"score out of range: {v}"

    # 4) 端到端 reward_function 接线自测
    print("\n[reward_function] end-to-end")
    out = reward_function([
        {
            "response": good_solution,
            "response_length": 100,
            "ground_truth": {
                "label_text": ["pneumonia"],
                "bbox": [["10", "140", "120", "250"]],
                "image": test_image if isinstance(test_image, str) else None,
            },
        }
    ])
    print("  result keys:", sorted(out[0].keys()))
    assert "overall" in out[0]
    for k in ("local_support", "contrastive", "counterfactual"):
        assert k in out[0], f"missing metric {k}"
    print("  overall =", round(out[0]["overall"], 4))
    print("\nAll self-tests passed.")







