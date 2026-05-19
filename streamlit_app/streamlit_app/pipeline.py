"""
Inference pipeline driven by **user prompts**, hỗ trợ 4 variant:

  - 'old'      : UNet+ASPP binary sigmoid (`unet_aspp_best.pth`).
  - 'v2'       : UNet+ASPP softmax 5ch    (`unet_aspp_new_v2_best.pth`).
  - 'v3'       : UNet+ASPP softmax + DINOv2 (`unet_aspp_new_v3_best.pth`).
  - 'finetune' : SAM3 LoRA fine-tuned (PEFT directory).

Logic route theo `class` của từng prompt:
  - class ∈ {diningtable, sofa, chair, bicycle}
        → variant ∈ {old,v2,v3}: SAM3 base coarse + co-occurrence + UNet+ASPP refine
          variant == 'finetune': finetuned SAM3 trực tiếp với user prompt.
  - class == 'pottedplant'
        → SAM3 base (user prompt + dual prompts `PLANT_PROMPTS`).
  - class khác (kể cả custom name)
        → SAM3 base với user prompt @ thr 0.3.
"""
from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from models import (
    ResNet34UNetASPP,
    ResNet34UNetASPPMulti,
    ResNet34UNetASPPMultiV3,
)

# ──────────────────── Constants ────────────────────────────────────────────

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_IDX = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}
INDEX_TO_CLASS = {0: "background", **{i + 1: n for i, n in enumerate(VOC_CLASSES)}}

TARGET_CLASSES = ["diningtable", "sofa", "chair", "bicycle"]   # 4 weak class
TARGET_TO_LOCAL_1 = {n: i + 1 for i, n in enumerate(TARGET_CLASSES)}  # 1..4
TARGET_TO_VOC = {n: VOC_CLASS_TO_IDX[n] for n in TARGET_CLASSES}
PLANT_VOC_IDX = VOC_CLASS_TO_IDX["pottedplant"]

CO_OCC_CLASSES = ["person", "cat", "dog", "bottle", "pottedplant"]
CO_OCC_PROMPTS = {
    "person": "a person",
    "cat": "a cat",
    "dog": "a dog",
    "bottle": "a bottle",
    "pottedplant": "potted plant with its pot or container",
}

# PLANT_PROMPTS — Dual prompts cố định cho method "dual prompt" ở pottedplant.
# Phải khớp với training pipeline (build_hybrid20*).
# User có thể chỉnh nếu muốn thấy gain rõ hơn (vd thêm 'flower pot', 'vase').
PLANT_PROMPTS = ["potted plant with its pot or container", "flowerpot"]

SAM3_CONFIDENCE = 0.3
CO_OCC_CONFIDENCE = 0.5
PLANT_CONFIDENCE = 0.3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 512
NUM_TARGET = len(TARGET_CLASSES)

WEAK_CLASS_SET = set(TARGET_CLASSES) | {"pottedplant"}


def assign_prompt_index(idx_in_list: int, cls: str) -> int:
    """Map (index, class) -> uint8 idx cho composite label map.
    Class ∈ VOC -> dùng VOC index (giữ đúng màu palette).
    Custom class -> 50 + i (tránh trùng 1..20).
    """
    if cls in VOC_CLASS_TO_IDX:
        return VOC_CLASS_TO_IDX[cls]
    return 50 + idx_in_list


def route_for(cls: str, variant: str) -> str:
    if cls in TARGET_CLASSES:
        return "finetune SAM3" if variant == "finetune" else f"UNet+ASPP refine ({variant})"
    if cls == "pottedplant":
        return "SAM3 dual-prompt"
    return "SAM3 baseline"


# ──────────────────── SAM3 helpers ────────────────────────────────────────

def _sam3_inference_context(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _extract_masks_scores(state: dict) -> tuple[list[np.ndarray], list[float]]:
    if "masks" not in state or state["masks"] is None or len(state["masks"]) == 0:
        return [], []
    masks, scores = [], []
    for i in range(len(state["masks"])):
        masks.append(state["masks"][i].squeeze(0).cpu().numpy().astype(bool))
        scores.append(float(state["scores"][i].item()))
    return masks, scores


def _union_masks(masks: list[np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    out = np.zeros((h, w), dtype=bool)
    for m in masks:
        out |= m.astype(bool)
    return out


def _run_single_prompt(
    processor: Any,
    state: dict,
    prompt: str,
    threshold: float,
    device: str,
) -> tuple[dict, list[np.ndarray], list[float]]:
    """Chạy 1 prompt ở threshold tùy ý (tạm thay confidence_threshold).

    Quan trọng: KHÔNG bọc try/except quanh `reset_all_prompts` — nếu reset fail,
    state cũ sẽ leak qua các prompt sau làm pred_sam3 sai. Để bug nổi lên ngay.
    """
    old_thr = processor.confidence_threshold
    processor.confidence_threshold = float(threshold)
    with _sam3_inference_context(device):
        if hasattr(processor, "reset_all_prompts"):
            processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
    masks, scores = _extract_masks_scores(state)
    processor.confidence_threshold = old_thr
    return state, masks, scores


def _run_baseline_fresh(
    processor: Any,
    image: Image.Image,
    prompt: str,
    threshold: float,
    device: str,
) -> tuple[list[np.ndarray], list[float]]:
    """Chạy SAM3 với 1 prompt từ STATE TRỐNG hoàn toàn — tái encode image.

    Tương đương standalone:
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        → masks, scores

    Dùng cho mask_sam3 (nguồn của pred_sam3 baseline) để đảm bảo KHÔNG có
    leak từ prompt trước đó. Chậm hơn nhưng kết quả khớp standalone test.
    """
    old_thr = processor.confidence_threshold
    processor.confidence_threshold = float(threshold)
    with _sam3_inference_context(device):
        state = processor.set_image(image)
        if hasattr(processor, "reset_all_prompts"):
            processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
    masks, scores = _extract_masks_scores(state)
    processor.confidence_threshold = old_thr
    return masks, scores


def _build_co_occ_mask(
    processor: Any,
    state: dict,
    image_hw: tuple[int, int],
    device: str,
) -> tuple[dict, np.ndarray]:
    h, w = image_hw
    co_occ = np.zeros((h, w), dtype=bool)
    for cls_name in CO_OCC_CLASSES:
        prompt = CO_OCC_PROMPTS[cls_name]
        thr = PLANT_CONFIDENCE if cls_name == "pottedplant" else CO_OCC_CONFIDENCE
        state, masks, _ = _run_single_prompt(processor, state, prompt, thr, device)
        co_occ |= _union_masks(masks, (h, w))
    return state, co_occ


# ──────────────────── UNet forward (3 variant) ─────────────────────────────

def _prepare_rgb(image_pil: Image.Image) -> torch.Tensor:
    img_r = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return TF.normalize(TF.to_tensor(img_r), IMAGENET_MEAN, IMAGENET_STD)


def _resize_mask_to_tensor(mask_hw: np.ndarray) -> torch.Tensor:
    m_r = np.array(
        Image.fromarray(mask_hw.astype(np.uint8)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST
        ),
        dtype=np.float32,
    )
    return torch.from_numpy((m_r > 0).astype(np.float32)).unsqueeze(0)


@torch.no_grad()
def run_unet_old(
    model: Any,
    device: str,
    image_pil: Image.Image,
    target_coarse: dict[str, np.ndarray],
    co_occ_hw: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """4 forward, binary sigmoid, argmax + threshold gate."""
    h, w = co_occ_hw.shape
    rgb = _prepare_rgb(image_pil)
    co_occ_t = _resize_mask_to_tensor(co_occ_hw)

    probs = np.zeros((NUM_TARGET, h, w), dtype=np.float32)
    for local_id, cls_name in enumerate(TARGET_CLASSES):
        coarse = target_coarse.get(cls_name, np.zeros((h, w), dtype=bool))
        if not coarse.any():
            continue

        coarse_t = _resize_mask_to_tensor(coarse)
        one_hot = torch.zeros(NUM_TARGET, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
        one_hot[local_id] = 1.0

        inp = torch.cat([rgb, coarse_t, co_occ_t, one_hot], dim=0).unsqueeze(0).to(device)
        logits = model(inp)
        prob_512 = torch.sigmoid(logits).squeeze().float().cpu().numpy()

        prob_pil = Image.fromarray((prob_512 * 255.0).clip(0, 255).astype(np.uint8))
        prob_pil = prob_pil.resize((w, h), Image.BILINEAR)
        probs[local_id] = np.asarray(prob_pil, dtype=np.float32) / 255.0

    best_local = probs.argmax(axis=0)
    best_prob = probs.max(axis=0)
    fire = best_prob >= threshold

    label_hw = np.zeros((h, w), dtype=np.uint8)
    for local_id in range(NUM_TARGET):
        sel = fire & (best_local == local_id)
        label_hw[sel] = local_id + 1
    return label_hw


@torch.no_grad()
def run_unet_v2(
    model: Any,
    device: str,
    image_pil: Image.Image,
    target_coarse: dict[str, np.ndarray],
    co_occ_hw: np.ndarray,
) -> np.ndarray:
    """1 forward, softmax 5ch."""
    h, w = co_occ_hw.shape
    rgb = _prepare_rgb(image_pil)
    co_occ_t = _resize_mask_to_tensor(co_occ_hw)

    coarse_chs = []
    for cls_name in TARGET_CLASSES:
        coarse = target_coarse.get(cls_name, np.zeros((h, w), dtype=bool))
        coarse_chs.append(_resize_mask_to_tensor(coarse))
    coarse_t = torch.cat(coarse_chs, dim=0)  # [4, 512, 512]

    inp = torch.cat([rgb, coarse_t, co_occ_t], dim=0).unsqueeze(0).to(device)
    logits = model(inp)
    pred_512 = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_pil = Image.fromarray(pred_512).resize((w, h), Image.NEAREST)
    return np.asarray(pred_pil, dtype=np.uint8)


@torch.no_grad()
def run_unet_v3(
    model: Any,
    dinov2_model: Any,
    dinov2_input_size: int,
    dinov2_grid_size: int,
    device: str,
    image_pil: Image.Image,
    target_coarse: dict[str, np.ndarray],
    co_occ_hw: np.ndarray,
) -> np.ndarray:
    """1 forward, softmax 5ch + DINOv2 patch feature."""
    h, w = co_occ_hw.shape
    rgb = _prepare_rgb(image_pil)
    co_occ_t = _resize_mask_to_tensor(co_occ_hw)

    coarse_chs = []
    for cls_name in TARGET_CLASSES:
        coarse = target_coarse.get(cls_name, np.zeros((h, w), dtype=bool))
        coarse_chs.append(_resize_mask_to_tensor(coarse))
    coarse_t = torch.cat(coarse_chs, dim=0)
    inp_8 = torch.cat([rgb, coarse_t, co_occ_t], dim=0).unsqueeze(0).to(device)

    # DINOv2 features
    dinov2_resized = image_pil.resize((dinov2_input_size, dinov2_input_size), Image.BILINEAR)
    dinov2_t = TF.normalize(
        TF.to_tensor(dinov2_resized),
        IMAGENET_MEAN, IMAGENET_STD,
    ).unsqueeze(0).to(device)

    feat_dict = dinov2_model.forward_features(dinov2_t)
    patch_tokens = feat_dict["x_norm_patchtokens"]  # [1, num_patch, 384]
    feat = patch_tokens.reshape(1, dinov2_grid_size, dinov2_grid_size, -1)
    feat = feat.permute(0, 3, 1, 2).contiguous()  # [1, 384, gh, gw]

    logits = model(inp_8, feat)
    pred_512 = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_pil = Image.fromarray(pred_512).resize((w, h), Image.NEAREST)
    return np.asarray(pred_pil, dtype=np.uint8)


# ──────────────────── Predict driven by user prompts ───────────────────────

def predict_user_prompts(
    image_pil: Image.Image,
    user_prompts: list[dict],          # each: {'class', 'prompt'}
    variant: str,                       # 'old' | 'v2' | 'v3' | 'finetune'
    *,
    processor_base: Any,                # base SAM3 (xử lý non-weak + pottedplant + coarse + co_occ)
    processor_ft: Any | None = None,    # finetuned SAM3 (chỉ dùng khi variant=='finetune')
    unet_model: Any | None = None,      # UNet model (chỉ dùng khi variant ∈ {old,v2,v3})
    old_threshold: float = 0.5,
    dinov2_model: Any | None = None,
    dinov2_input_size: int = 518,
    dinov2_grid_size: int = 37,
    device: str = "cuda",
    progress_cb=None,
) -> dict:
    """Trả về dict với:
      'per_prompt'  : list[dict] — chi tiết mỗi prompt (mask_sam3, mask_final, method, idx).
      'pred_sam3'   : (H, W) uint8 — composite SAM3-only.
      'pred_hybrid' : (H, W) uint8 — composite có refinement ở 5 weak class.
      'co_occ_mask' : bool (H, W) — None nếu không cần.
      'unet_label'  : (H, W) uint8 — chỉ có khi variant ∈ {old,v2,v3} và có target.
      'plant_extra' : bool (H, W) — plant union dual prompts (None nếu không cần).
      'has_target'  : bool.
      'has_plant'   : bool.
    """
    image = image_pil.convert("RGB")
    w, h = image.size

    def _cb(stage: str, frac: float) -> None:
        if progress_cb is not None:
            progress_cb(stage, frac)

    if variant == "finetune" and processor_ft is None:
        raise ValueError("variant='finetune' cần processor_ft (load finetuned SAM3 trước).")

    # ───────── B1: chạy SAM3 cho từng user prompt ─────────
    # MỖI user prompt được chạy với state TRỐNG hoàn toàn (re-encode image).
    # → mask_sam3 = đúng kết quả em sẽ thấy nếu chạy SAM3 standalone với
    #   processor.set_image(image) → reset_all_prompts → set_text_prompt(prompt).
    # KHÔNG share state giữa các prompt → KHÔNG có leak.
    per_prompt: list[dict] = []
    target_coarse: dict[str, np.ndarray] = {cls: np.zeros((h, w), dtype=bool) for cls in TARGET_CLASSES}
    target_score_ft: dict[str, float] = {cls: 0.0 for cls in TARGET_CLASSES}
    target_user_prompt: dict[str, str] = {}
    has_target = False
    has_plant = False

    n = len(user_prompts)
    for i, entry in enumerate(user_prompts):
        cls = entry["class"]
        prompt_text = entry["prompt"]
        _cb(f"SAM3 base prompt [{i+1}/{n}]: {cls}", 0.05 + 0.30 * (i + 1) / max(n, 1))

        # SAM3 BASE — fresh state mỗi prompt → mask_sam3 sạch, không leak.
        masks_b, scores_b = _run_baseline_fresh(
            processor_base, image, prompt_text, SAM3_CONFIDENCE, device
        )
        m_sam3_base = _union_masks(masks_b, (h, w))
        s_best_base = max(scores_b) if scores_b else 0.0

        per_prompt.append({
            "i": i,
            "class": cls,
            "prompt": prompt_text,
            "mask_sam3": m_sam3_base,
            "score_sam3": s_best_base,
            "raw_masks": [np.asarray(m, dtype=bool) for m in masks_b],   # debug: từng instance
            "raw_scores": [float(s) for s in scores_b],
            "idx": assign_prompt_index(i, cls),
            "method": route_for(cls, variant),
        })

        if cls in TARGET_CLASSES:
            has_target = True
            target_user_prompt[cls] = prompt_text
            if variant in {"old", "v2", "v3"}:
                target_coarse[cls] = m_sam3_base
            else:
                # variant == 'finetune' → chạy thêm finetune SAM3 (fresh state).
                masks_f, scores_f = _run_baseline_fresh(
                    processor_ft, image, prompt_text, SAM3_CONFIDENCE, device
                )
                target_coarse[cls] = _union_masks(masks_f, (h, w))
                target_score_ft[cls] = max(scores_f) if scores_f else 0.0
        if cls == "pottedplant":
            has_plant = True

    # Sau B1, encode lại image 1 lần cho processor_base để B2/B4 share state.
    with _sam3_inference_context(device):
        state_base = processor_base.set_image(image)

    # ───────── B2: co-occurrence (chỉ khi cần UNet) ─────────
    if has_target and variant in {"old", "v2", "v3"}:
        _cb("Co-occurrence (5 class che)", 0.45)
        state_base, co_occ_mask = _build_co_occ_mask(processor_base, state_base, (h, w), device)
    else:
        co_occ_mask = np.zeros((h, w), dtype=bool)

    # ───────── B3: UNet refine (chỉ khi variant ∈ {old,v2,v3}) ─────────
    unet_label = np.zeros((h, w), dtype=np.uint8)
    if has_target and variant in {"old", "v2", "v3"}:
        _cb(f"UNet+ASPP ({variant.upper()}) forward", 0.65)
        if variant == "old":
            unet_label = run_unet_old(
                unet_model, device, image, target_coarse, co_occ_mask,
                threshold=old_threshold,
            )
        elif variant == "v2":
            unet_label = run_unet_v2(unet_model, device, image, target_coarse, co_occ_mask)
        elif variant == "v3":
            if dinov2_model is None:
                raise ValueError("v3 cần dinov2_model.")
            unet_label = run_unet_v3(
                unet_model, dinov2_model, dinov2_input_size, dinov2_grid_size,
                device, image, target_coarse, co_occ_mask,
            )

    # ───────── B4: plant dual (chỉ khi user hỏi pottedplant) ─────────
    # Mỗi PLANT_PROMPT chạy fresh state (re-encode image) → KHÔNG leak từ B2/B3.
    plant_dual_per_prompt: list[dict] = []
    if has_plant:
        plant_extra = np.zeros((h, w), dtype=bool)
        for j, p_prompt in enumerate(PLANT_PROMPTS):
            _cb(f"Plant dual-prompt [{j+1}/{len(PLANT_PROMPTS)}]: {p_prompt}",
                0.80 + 0.10 * (j + 1) / len(PLANT_PROMPTS))
            p_masks, p_scores = _run_baseline_fresh(
                processor_base, image, p_prompt, PLANT_CONFIDENCE, device
            )
            m_p = _union_masks(p_masks, (h, w))
            plant_extra |= m_p
            plant_dual_per_prompt.append({
                "prompt": p_prompt,
                "mask": m_p,
                "raw_masks": [np.asarray(m, dtype=bool) for m in p_masks],
                "raw_scores": [float(s) for s in p_scores],
                "n_pixels": int(m_p.sum()),
            })
    else:
        plant_extra = np.zeros((h, w), dtype=bool)

    # ───────── B5: fill mask_final per prompt ─────────
    # Quy ước:
    #   - pred_sam3 (baseline) cho MỌI class luôn dùng entry["mask_sam3"]
    #     = SAM3 base với CHÍNH user prompt → đây là "single-prompt baseline".
    #   - pred_hybrid:
    #       * 4 weak: UNet refine / finetune SAM3.
    #       * pottedplant: chỉ dùng `plant_extra` (= union 2 dual prompts CỐ ĐỊNH,
    #         KHÔNG bao gồm user prompt) → đây mới là "dual prompt method"
    #         được so sánh với baseline. Nếu user prompt của em trùng đúng 1
    #         trong PLANT_PROMPTS thì baseline = single-prompt subset của dual.
    #       * còn lại: y nguyên SAM3 user prompt (không refine gì).
    for entry in per_prompt:
        cls = entry["class"]
        if cls in TARGET_CLASSES:
            if variant == "finetune":
                # mask = output từ finetuned SAM3 (đã lưu vào target_coarse[cls])
                entry["mask_final"] = target_coarse[cls].copy()
            else:
                local_id = TARGET_TO_LOCAL_1[cls]
                entry["mask_final"] = (unet_label == local_id)
        elif cls == "pottedplant":
            # ↓ CHỈ dual fixed, KHÔNG union với user prompt → đảm bảo khác baseline
            entry["mask_final"] = plant_extra.copy()
        else:
            entry["mask_final"] = entry["mask_sam3"].copy()

    # ───────── B6: build composite label maps ─────────
    pred_sam3 = np.zeros((h, w), dtype=np.uint8)
    best_score_sam3 = np.zeros((h, w), dtype=np.float32)
    pred_hybrid = np.zeros((h, w), dtype=np.uint8)

    for entry in per_prompt:
        m = entry["mask_sam3"]
        upd = m & (entry["score_sam3"] > best_score_sam3)
        pred_sam3[upd] = entry["idx"]
        best_score_sam3[upd] = entry["score_sam3"]

    # Hybrid: ưu tiên theo thứ tự user nhập (prompt sau đè prompt trước).
    for entry in per_prompt:
        pred_hybrid[entry["mask_final"]] = entry["idx"]

    _cb("Done", 1.0)

    return {
        "per_prompt": per_prompt,
        "pred_sam3": pred_sam3,
        "pred_hybrid": pred_hybrid,
        "co_occ_mask": co_occ_mask,
        "unet_label": unet_label,
        "plant_extra": plant_extra,
        "plant_dual_per_prompt": plant_dual_per_prompt,
        "has_target": has_target,
        "has_plant": has_plant,
        "target_coarse_or_ft": target_coarse,   # dict cls -> mask
        "target_user_prompt": target_user_prompt,
        "variant": variant,
    }


# ──────────────────── Model loaders ────────────────────────────────────────

def load_unet_variant(
    ckpt_path: str | Path,
    variant: str,
    device: str = "cuda",
) -> tuple[Any, dict]:
    """Load 1 trong 3 UNet variant từ checkpoint .pth."""
    ckpt = torch.load(ckpt_path, map_location=device)

    if variant == "old":
        in_ch = int(ckpt.get("in_channels", 9))
        model = ResNet34UNetASPP(in_channels=in_ch, out_channels=1).to(device)
        state_dict = ckpt["model_state_dict"].copy()
        load_info = model.load_state_dict(state_dict, strict=False)
        if load_info.unexpected_keys:
            raise RuntimeError(f"Unexpected keys: {load_info.unexpected_keys}")
        model.eval()
        meta = {
            "variant": "old",
            "in_channels": in_ch,
            "refine_threshold": float(ckpt.get("refine_threshold", 0.5)),
            "target_classes": ckpt.get("target_classes", TARGET_CLASSES),
            "epoch": ckpt.get("epoch", "?"),
            "miou": ckpt.get("miou", None),
            "missing_keys": load_info.missing_keys,
        }
        return model, meta

    if variant == "v2":
        in_ch = int(ckpt.get("in_channels", 8))
        out_ch = int(ckpt.get("num_out_classes", 5))
        model = ResNet34UNetASPPMulti(in_channels=in_ch, out_channels=out_ch).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        meta = {
            "variant": "v2",
            "in_channels": in_ch,
            "num_out_classes": out_ch,
            "arch_version": ckpt.get("arch_version", "unknown"),
            "target_classes": ckpt.get("target_classes", TARGET_CLASSES),
            "epoch": ckpt.get("epoch", "?"),
            "miou": ckpt.get("miou", None),
        }
        return model, meta

    if variant == "v3":
        in_ch_8 = int(ckpt.get("in_channels_8", 8))
        dinov2_dim = int(ckpt.get("dinov2_embed_dim", 384))
        dinov2_compress = int(ckpt.get("dinov2_compress", 32))
        out_ch = int(ckpt.get("num_out_classes", 5))
        model = ResNet34UNetASPPMultiV3(
            in_channels_8=in_ch_8,
            dinov2_dim=dinov2_dim,
            dinov2_compress=dinov2_compress,
            out_channels=out_ch,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        meta = {
            "variant": "v3",
            "in_channels_8": in_ch_8,
            "dinov2_embed_dim": dinov2_dim,
            "dinov2_compress": dinov2_compress,
            "dinov2_input_size": int(ckpt.get("dinov2_input_size", 518)),
            "dinov2_grid_size": int(ckpt.get("dinov2_grid_size", 37)),
            "dinov2_model_name": str(ckpt.get("dinov2_model_name", "dinov2_vits14")),
            "num_out_classes": out_ch,
            "arch_version": ckpt.get("arch_version", "unknown"),
            "target_classes": ckpt.get("target_classes", TARGET_CLASSES),
            "epoch": ckpt.get("epoch", "?"),
            "miou": ckpt.get("miou", None),
        }
        return model, meta

    raise ValueError(f"Unknown variant: {variant}")


def load_dinov2(model_name: str = "dinov2_vits14", device: str = "cuda") -> Any:
    """Load DINOv2 frozen backbone cho v3."""
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    except Exception:
        candidates = [
            Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main",
            Path("/kaggle/input/dinov2-repo"),
        ]
        repo = next((p for p in candidates if p.exists()), None)
        if repo is None:
            raise RuntimeError(
                "Không load được DINOv2. Thử: "
                "git clone https://github.com/facebookresearch/dinov2 "
                "~/.cache/torch/hub/facebookresearch_dinov2_main"
            )
        sys.path.insert(0, str(repo))
        from dinov2.hub.backbones import dinov2_vits14 as _build  # type: ignore

        model = _build(pretrained=True)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _build_base_sam3(sam3_ckpt: str | Path, bpe_path: str | Path | None, device: str):
    """Build raw SAM3 image model (chưa gắn LoRA)."""
    import sam3
    from sam3 import build_sam3_image_model

    if bpe_path is None:
        bpe_path = Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    # Patch precompute_resolution (giống finetune notebook để tránh OOM)
    import sam3.model.position_encoding as _pe
    _orig = _pe.PositionEmbeddingSine
    if not getattr(_orig, "_streamlit_patched", False):
        class _PatchedPE(_orig):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("precompute_resolution", 800)
                super().__init__(*args, **kwargs)
        _PatchedPE._streamlit_patched = True
        _pe.PositionEmbeddingSine = _PatchedPE

    sam3_model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=str(sam3_ckpt),
        load_from_HF=False,
    )
    return sam3_model.to(device).eval()


def load_sam3(
    sam3_ckpt: str | Path,
    bpe_path: str | Path | None = None,
    confidence_threshold: float = SAM3_CONFIDENCE,
    device: str = "cuda",
) -> Any:
    """Load SAM3 base processor."""
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_model = _build_base_sam3(sam3_ckpt, bpe_path, device)
    return Sam3Processor(sam3_model, confidence_threshold=confidence_threshold)


def load_finetune_sam3(
    sam3_ckpt: str | Path,
    lora_dir: str | Path,
    bpe_path: str | Path | None = None,
    confidence_threshold: float = SAM3_CONFIDENCE,
    device: str = "cuda",
) -> Any:
    """Load SAM3 base + apply LoRA adapter từ PEFT directory.

    `lora_dir` là directory được tạo bởi `model.save_pretrained(...)` trong
    finetunesam3+7.ipynb (chứa `adapter_config.json` + `adapter_model.safetensors`).
    """
    from sam3.model.sam3_image_processor import Sam3Processor

    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            "Cần cài `peft` để load finetuned SAM3. Chạy: pip install peft"
        ) from e

    base_model = _build_base_sam3(sam3_ckpt, bpe_path, device)
    ft_model = PeftModel.from_pretrained(base_model, str(lora_dir))
    ft_model = ft_model.to(device).eval()

    return Sam3Processor(ft_model, confidence_threshold=confidence_threshold)
