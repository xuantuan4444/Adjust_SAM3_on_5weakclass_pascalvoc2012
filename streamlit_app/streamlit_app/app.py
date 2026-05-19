"""
Streamlit UI: Inference 1 ảnh **theo user prompt** với 4 weight options.

User chọn 1 trong 4 checkpoint:
  - OLD     : unet_aspp_best.pth          (UNet+ASPP binary sigmoid)
  - v2      : unet_aspp_new_v2_best.pth   (UNet+ASPP softmax 5ch)
  - v3      : unet_aspp_new_v3_best.pth   (UNet+ASPP softmax + DINOv2)
  - finetune: SAM3 LoRA fine-tuned        (PEFT directory)

Mỗi prompt user nhập gồm `class` (canonical) + `prompt` (text gửi SAM3).
Auto-route:
  - class ∈ {diningtable, sofa, chair, bicycle}
        → variant ∈ {OLD,v2,v3}: UNet+ASPP refine
          variant == finetune: SAM3 LoRA
  - class == 'pottedplant'      → SAM3 dual-prompt (base)
  - class khác (kể cả custom)   → SAM3 base với prompt user

Chạy: `streamlit run app.py`
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

import pipeline as P
import viz as V

# ──────────────────── Config paths ────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent

# 2. Lùi ra thư mục cha 1 cấp, rồi trỏ vào thư mục "weights"
# (BASE_DIR chính là thư mục chứa cả streamlit_app và weights)
BASE_DIR = CURRENT_DIR.parent
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", BASE_DIR / "weights"))

DEFAULT_PATHS = {
    "SAM3":     os.environ.get("SAM3_CKPT",      str(WEIGHTS_DIR / "sam3.pt")),
    "OLD":      os.environ.get("UNET_CKPT_OLD",  str(WEIGHTS_DIR / "unet_aspp_best.pth")),
    "v2":       os.environ.get("UNET_CKPT_V2",   str(WEIGHTS_DIR / "unet_aspp_new_v2_best.pth")),
    "v3":       os.environ.get("UNET_CKPT_V3",   str(WEIGHTS_DIR / "unet_aspp_new_v3_best.pth")),
    "finetune": os.environ.get("SAM3_LORA_DIR",  str(WEIGHTS_DIR / "sam3_voc_lora_v16_best")),
}

VARIANT_LABEL = {
    "OLD":      "OLD — UNet+ASPP binary sigmoid (4 forward/ảnh, 9ch)",
    "v2":       "v2  — UNet+ASPP softmax 5ch (1 forward, 8ch)",
    "v3":       "v3  — UNet+ASPP + DINOv2 (1 forward + DINOv2)",
    "finetune": "finetune — SAM3 LoRA fine-tuned (PEFT adapter)",
}
VARIANT_KEY = {"OLD": "old", "v2": "v2", "v3": "v3", "finetune": "finetune"}


# ──────────────────── Cached loaders ──────────────────────────────────────

@st.cache_resource(show_spinner="Loading SAM3 base (1 lần)...")
def get_sam3_base(ckpt_path: str, device: str):
    return P.load_sam3(ckpt_path, device=device)


@st.cache_resource(show_spinner="Loading UNet variant...")
def get_unet(variant_key: str, ckpt_path: str, device: str):
    model, meta = P.load_unet_variant(ckpt_path, variant_key, device)
    return model, meta


@st.cache_resource(show_spinner="Loading DINOv2 (chỉ khi chọn v3)...")
def get_dinov2(model_name: str, device: str):
    return P.load_dinov2(model_name, device)


@st.cache_resource(show_spinner="Loading finetuned SAM3 (LoRA)...")
def get_sam3_finetune(sam3_ckpt: str, lora_dir: str, device: str):
    return P.load_finetune_sam3(sam3_ckpt, lora_dir, device=device)


# ──────────────────── UI ──────────────────────────────────────────────────

st.set_page_config(
    page_title="SAM3 + UNet+ASPP / Finetune — Inference",
    layout="wide",
)

st.title("SAM3 — Inference với 4 weight + user prompt")
st.caption(
    "Chọn 1 trong 4 weight, upload 1 ảnh bất kỳ, nhập danh sách prompt. "
    "App tự route mỗi prompt theo class:  4 weak class → UNet/finetune refine, "
    "pottedplant → dual prompt, các class khác → SAM3 baseline."
)

with st.sidebar:
    st.header("1. Checkpoints")

    sam3_path = st.text_input(
        "SAM3 base checkpoint (`sam3.pt`)",
        value=DEFAULT_PATHS["SAM3"],
    )

    st.divider()
    st.header("2. Chọn variant")
    variant_name = st.radio(
        "Weight để refine 4 weak class",
        options=list(VARIANT_LABEL.keys()),
        format_func=lambda k: VARIANT_LABEL[k],
        index=2,
    )

    if variant_name == "finetune":
        weight_label = "PEFT directory (chứa adapter_config.json + adapter_model.safetensors)"
    else:
        weight_label = f"UNet {variant_name} checkpoint (.pth)"

    weight_path = st.text_input(weight_label, value=DEFAULT_PATHS[variant_name])

    st.divider()
    

    show_intermediate = st.checkbox(
        "Hiện bước trung gian (co_occ + UNet/FT mask)",
        value=False,
    )
    alpha = st.slider("Độ đậm overlay", 0.2, 0.95, 0.55, 0.05)

    st.divider()
    st.header("3. Device")
    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    device = st.selectbox(
        "Device", options=["cuda", "cpu"],
        index=0 if device_default == "cuda" else 1,
        disabled=not torch.cuda.is_available(),
    )
    st.caption(f"CUDA available: **{torch.cuda.is_available()}**")


# ──────────────────── Main area ──────────────────────────────────────────

st.subheader("Upload ảnh")
uploaded = st.file_uploader(
    "Chọn 1 ảnh (JPEG/PNG)", type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

st.subheader("User prompts")
st.caption(
    "Nhập mỗi prompt 1 dòng theo định dạng `class | prompt`. "
    "VD `chair | a wooden chair`. "
    "`class` là tên định tuyến (canonical); prompt là text gửi SAM3. "
    "Nếu prompt không có dấu `|` thì lấy luôn làm cả class lẫn prompt. "
    f"Class kích hoạt UNet/finetune: `{', '.join(P.TARGET_CLASSES)}`. "
    "Class kích hoạt dual prompt: `pottedplant`."
)

DEFAULT_PROMPTS_TEXT = (
    "chair | a wooden chair\n"
    "sofa | a sofa\n"
    "pottedplant | potted plant with its pot or container\n"
    "person | a person\n"
)

prompts_text = st.text_area(
    "Prompt list",
    value=DEFAULT_PROMPTS_TEXT,
    height=160,
    help="Mỗi dòng 1 prompt. Empty line bỏ qua. "
         "Format: `class | prompt`.",
)


def _parse_prompts(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            cls, prompt = (s.strip() for s in line.split("|", 1))
        else:
            cls = prompt = line
        if not cls or not prompt:
            continue
        out.append({"class": cls, "prompt": prompt})
    return out


user_prompts = _parse_prompts(prompts_text)

if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    col_in_1, col_in_2 = st.columns([1, 1])
    with col_in_1:
        st.image(image_pil, caption=f"Input ({image_pil.size[0]}×{image_pil.size[1]})",
                 use_container_width=True)
    with col_in_2:
        st.markdown(f"**Variant**: `{variant_name}`")
        st.markdown(f"**Weight**: `{Path(weight_path).name}`")
        st.markdown(f"**Device**: `{device}`")
        st.markdown(f"**Số prompt**: `{len(user_prompts)}`")
        if user_prompts:
            rows = []
            for i, e in enumerate(user_prompts):
                rows.append({
                    "i": i,
                    "class": e["class"],
                    "prompt": e["prompt"],
                    "method": P.route_for(e["class"], VARIANT_KEY[variant_name]),
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Chưa có prompt hợp lệ.")
else:
    st.info("Chưa upload ảnh.")

run_btn = st.button(
    "Run inference",
    type="primary",
    disabled=(uploaded is None or not user_prompts),
)


# ──────────────────── Run pipeline ──────────────────────────────────────

if run_btn and uploaded is not None and user_prompts:
    variant_key = VARIANT_KEY[variant_name]

    # Validate paths
    if not Path(sam3_path).exists():
        st.error(f"Không tìm thấy SAM3 base: `{sam3_path}`")
        st.stop()
    if not Path(weight_path).exists():
        st.error(f"Không tìm thấy weight: `{weight_path}`")
        st.stop()

    # Load base SAM3 (always needed)
    try:
        processor_base = get_sam3_base(sam3_path, device)
    except Exception as e:
        st.error(f"Load SAM3 base thất bại: {e}")
        st.stop()

    processor_ft = None
    unet_model = None
    unet_meta: dict = {}
    dinov2_model = None
    dinov2_input_size = 518
    dinov2_grid_size = 37

    if variant_key in {"old", "v2", "v3"}:
        try:
            unet_model, unet_meta = get_unet(variant_key, weight_path, device)
        except Exception as e:
            st.error(f"Load UNet {variant_name} thất bại: {e}")
            st.stop()
        if variant_key == "v3":
            try:
                dinov2_model = get_dinov2(unet_meta["dinov2_model_name"], device)
                dinov2_input_size = unet_meta["dinov2_input_size"]
                dinov2_grid_size = unet_meta["dinov2_grid_size"]
            except Exception as e:
                st.error(f"Load DINOv2 thất bại: {e}")
                st.stop()
    elif variant_key == "finetune":
        try:
            processor_ft = get_sam3_finetune(sam3_path, weight_path, device)
        except Exception as e:
            st.error(f"Load finetuned SAM3 thất bại: {e}")
            st.exception(e)
            st.stop()

    # Metadata box
    if unet_meta:
        with st.expander("UNet checkpoint metadata", expanded=False):
            st.json(
                {k: (str(v) if not isinstance(v, (int, float, str, list, dict)) else v)
                 for k, v in unet_meta.items()}
            )

    progress = st.progress(0.0, text="Starting...")

    def _cb(stage: str, frac: float) -> None:
        progress.progress(min(max(frac, 0.0), 1.0), text=stage)

    try:
        with torch.inference_mode():
            result = P.predict_user_prompts(
                image_pil=image_pil,
                user_prompts=user_prompts,
                variant=variant_key,
                processor_base=processor_base,
                processor_ft=processor_ft,
                unet_model=unet_model,
                dinov2_model=dinov2_model,
                dinov2_input_size=dinov2_input_size,
                dinov2_grid_size=dinov2_grid_size,
                device=device,
                progress_cb=_cb,
            )
    except Exception as e:
        st.error(f"Inference thất bại: {e}")
        st.exception(e)
        st.stop()
    finally:
        progress.empty()

    # ──────────────────── Display results ────────────────────────────────

    pred_sam3 = result["pred_sam3"]
    pred_hybrid = result["pred_hybrid"]
    image_rgb = np.array(image_pil)

    st.success("Done!")

    # Color cho baseline vs method overlay
    BASELINE_COLOR = (255, 80, 0)     # cam — SAM3 baseline
    METHOD_COLOR = (30, 180, 60)      # xanh lá — phương pháp cải tiến

    # ============================================================
    # 1. PER-PROMPT DETAIL — ảnh gốc | SAM3 baseline | phương pháp
    # ============================================================
    st.subheader("Kết quả từng class (Original | SAM3 baseline | Phương pháp)")

    def _method_description(cls: str, variant_key: str) -> tuple[str, str]:
        """Trả về (method_short, method_long) để hiển thị panel 3."""
        if cls in P.TARGET_CLASSES:
            if variant_key == "finetune":
                return "Finetuned SAM3", "SAM3 LoRA fine-tuned trên VOC2012 (PEFT adapter)"
            label_map = {
                "old": "UNet+ASPP (OLD)",
                "v2":  "UNet+ASPP (v2 softmax)",
                "v3":  "UNet+ASPP + DINOv2 (v3)",
            }
            return label_map[variant_key], f"{label_map[variant_key]} refine từ SAM3 coarse + co-occurrence"
        if cls == "pottedplant":
            joined = ", ".join(f"'{p}'" for p in P.PLANT_PROMPTS)
            return "Dual prompt", f"Union của {len(P.PLANT_PROMPTS)} prompt cố định: {joined}"
        return "(no improvement)", "Class này không có phương pháp riêng → cùng kết quả với SAM3 baseline"

    for entry in result["per_prompt"]:
        cls = entry["class"]
        method_short, method_long = _method_description(cls, variant_key)
        baseline_mask = entry["mask_sam3"]
        method_mask = entry["mask_final"]

        st.markdown(
            f"### `[{entry['i']}]` class = `{cls}`  •  user prompt = `'{entry['prompt']}'`"
        )
        st.caption(
            f"→ Phương pháp cải tiến: **{method_short}** — _{method_long}_"
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("**Ảnh gốc**")
            st.image(image_rgb, use_container_width=True)
        with c2:
            st.caption(
                f"**SAM3 baseline** — SAM3 với CHỈ user prompt `'{entry['prompt']}'`  "
            )
            st.image(
                V.overlay_mask_on_image(image_rgb, baseline_mask, BASELINE_COLOR, alpha=alpha),
                use_container_width=True,
            )
        with c3:
            st.caption(
                f"**{method_short}**  (`{int(method_mask.sum())} px`)"
            )
            st.image(
                V.overlay_mask_on_image(image_rgb, method_mask, METHOD_COLOR, alpha=alpha),
                use_container_width=True,
            )

        # Pottedplant: thêm row diff Baseline vs Dual để thấy gain rõ
        if cls == "pottedplant":
            only_in_dual = method_mask & ~baseline_mask
            only_in_base = baseline_mask & ~method_mask
            inter = baseline_mask & method_mask
            with st.expander(
                f"DEBUG pottedplant — Baseline vs Dual diff "
                f"(Dual gain: {int(only_in_dual.sum())} px)",
                expanded=False,
            ):
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.caption(f"Chỉ ở Dual (gain) — `{int(only_in_dual.sum())} px`")
                    st.image(
                        V.overlay_mask_on_image(image_rgb, only_in_dual, (255, 0, 200), alpha=alpha),
                        use_container_width=True,
                    )
                with d2:
                    st.caption(f"Chỉ ở Baseline — `{int(only_in_base.sum())} px`")
                    st.image(
                        V.overlay_mask_on_image(image_rgb, only_in_base, (200, 200, 0), alpha=alpha),
                        use_container_width=True,
                    )
                with d3:
                    st.caption(f"Chung — `{int(inter.sum())} px`")
                    st.image(
                        V.overlay_mask_on_image(image_rgb, inter, (100, 100, 255), alpha=alpha),
                        use_container_width=True,
                    )

                st.markdown("**Từng PLANT_PROMPT riêng lẻ:**")
                pp_dual = result.get("plant_dual_per_prompt", [])
                if pp_dual:
                    cols_d = st.columns(min(len(pp_dual), 4))
                    for k, item in enumerate(pp_dual):
                        col = cols_d[k % len(cols_d)]
                        with col:
                            st.caption(
                                f"`'{item['prompt']}'` → `{item['n_pixels']} px`"
                            )
                            st.image(
                                V.overlay_mask_on_image(image_rgb, item["mask"],
                                                        METHOD_COLOR, alpha=alpha),
                                use_container_width=True,
                            )

        # Debug panel — raw SAM3 instances của user prompt (giúp khớp standalone)
        raw_masks = entry.get("raw_masks", [])
        raw_scores = entry.get("raw_scores", [])
        with st.expander(
            f"DEBUG — Raw SAM3 instances cho user prompt `'{entry['prompt']}'` "
            f"(`{len(raw_masks)}` instance)",
            expanded=False,
        ):
            if not raw_masks:
                st.warning(
                    "SAM3 không trả về instance nào @ confidence_threshold=0.3. "
                    "Đây phải khớp với standalone test."
                )
            else:
                order = sorted(range(len(raw_masks)), key=lambda k: -raw_scores[k])
                cols_r = st.columns(min(len(raw_masks), 4))
                for j, idx in enumerate(order):
                    with cols_r[j % len(cols_r)]:
                        m = raw_masks[idx]
                        st.caption(
                            f"#{idx} score=`{raw_scores[idx]:.3f}` `{int(m.sum())} px`"
                        )
                        st.image(
                            V.overlay_mask_on_image(image_rgb, m, BASELINE_COLOR, alpha=alpha),
                            use_container_width=True,
                        )

        st.markdown("---")

    # ============================================================
    # 2. COMPOSITE OVERVIEW — gộp tất cả prompt thành 1 label map
    # ============================================================
    st.subheader("Composite overview (gộp tất cả prompt vào 1 ảnh)")

    co_a, co_b, co_c = st.columns(3)
    with co_a:
        st.markdown("**Ảnh gốc**")
        st.image(image_rgb, use_container_width=True)
    with co_b:
        st.markdown("**SAM3 baseline **")
        st.image(
            V.overlay_rgb_on_label(image_rgb, pred_sam3, alpha=alpha),
            use_container_width=True,
        )
    with co_c:
        st.markdown(f"**Phương pháp {variant_name})**")
        st.caption(
            f"4 weak class → {variant_name}. pottedplant → dual prompt. "
            "Class khác → SAM3 baseline."
        )
        st.image(
            V.overlay_rgb_on_label(image_rgb, pred_hybrid, alpha=alpha),
            use_container_width=True,
        )

    # Download button
    import io
    from PIL import Image as PILImage

    def _save_indexed(label_hw: np.ndarray, name: str) -> bytes:
        """Save label uint8 as indexed PNG with VOC palette."""
        buf = io.BytesIO()
        im = PILImage.fromarray(label_hw.astype(np.uint8), mode="P")
        im.putpalette(V.VOC_PALETTE)
        im.save(buf, format="PNG")
        return buf.getvalue()

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download `pred_sam3` (indexed PNG, VOC palette)",
            data=_save_indexed(pred_sam3, "pred_sam3"),
            file_name=f"pred_sam3_{Path(uploaded.name).stem}.png",
            mime="image/png",
        )
    with col_dl2:
        st.download_button(
            f"Download `pred_hybrid_{variant_name}` (indexed PNG)",
            data=_save_indexed(pred_hybrid, "pred_hybrid"),
            file_name=f"pred_hybrid_{variant_name}_{Path(uploaded.name).stem}.png",
            mime="image/png",
        )

    # Class legend
    st.subheader("Legend (hybrid output)")
    legend = V.legend_classes_present(pred_hybrid)
    if legend:
        cols = st.columns(min(len(legend), 6))
        for (idx, (cls_idx, name, rgb)), col in zip(enumerate(legend), cols * 2):
            with col:
                swatch = np.zeros((32, 32, 3), dtype=np.uint8)
                swatch[:] = rgb
                # Custom class (idx 50+) -> hiển thị tên user nhập nếu trong VOC list
                display_name = name if cls_idx <= 20 else f"custom#{cls_idx-50}"
                st.image(swatch, caption=f"{cls_idx}: {display_name}", width=64)
    else:
        st.caption("Không có foreground.")

    # Optional intermediate viz
    if show_intermediate:
        st.subheader("Bước trung gian")

        if result["has_target"] and variant_key in {"old", "v2", "v3"}:
            with st.expander("Co-occurrence mask", expanded=False):
                m = result["co_occ_mask"]
                st.markdown(f"Union person/cat/dog/bottle/plant: `{int(m.sum())} px`")
                st.image(V.mask_to_rgb(m, (200, 0, 200)), use_container_width=True)

            with st.expander("UNet+ASPP label (1..4 = 4 weak class)", expanded=False):
                label = result["unet_label"]
                st.image(V.label_to_rgb_target(label), use_container_width=True)

        if result["has_target"] and variant_key == "finetune":
            with st.expander("Finetuned SAM3 mask per target class", expanded=False):
                tcoarse = result["target_coarse_or_ft"]
                cols = st.columns(len(P.TARGET_CLASSES))
                for (cls, mask), col in zip(tcoarse.items(), cols):
                    with col:
                        st.markdown(f"**{cls}** `{int(mask.sum())} px`")
                        if mask.any():
                            st.image(V.mask_to_rgb(mask, (255, 100, 0)),
                                     use_container_width=True)
                        else:
                            st.caption("(empty)")

        # (Plant dual diff đã có sẵn ở per-prompt panel của pottedplant ở trên)
