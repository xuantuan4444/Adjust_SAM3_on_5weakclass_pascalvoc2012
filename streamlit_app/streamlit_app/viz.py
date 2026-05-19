"""VOC2012 color palette + mask/overlay visualization helpers."""
from __future__ import annotations

import numpy as np
from PIL import Image

from pipeline import INDEX_TO_CLASS


def voc_colormap(N: int = 256) -> np.ndarray:
    """PASCAL VOC 2012 colormap (256x3 uint8). Giống SegmentationClass PNG palette gốc."""

    def bitget(byteval: int, idx: int) -> int:
        return (byteval >> idx) & 1

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


VOC_CMAP = voc_colormap()
VOC_PALETTE = VOC_CMAP.flatten().tolist()


def colorize_label(label_hw: np.ndarray) -> np.ndarray:
    """Label int (H, W) -> RGB (H, W, 3) uint8 theo VOC palette."""
    label = label_hw.astype(np.int64).clip(0, 255)
    return VOC_CMAP[label]


def overlay_rgb_on_label(
    image_rgb: np.ndarray,
    label_hw: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlay color-coded label lên RGB ảnh. Bg (label=0) không tô."""
    color = colorize_label(label_hw).astype(np.float32)
    rgb = image_rgb.astype(np.float32)
    mask = (label_hw > 0) & (label_hw != 255)
    out = rgb.copy()
    out[mask] = alpha * color[mask] + (1 - alpha) * rgb[mask]
    return out.clip(0, 255).astype(np.uint8)


def legend_classes_present(label_hw: np.ndarray) -> list[tuple[int, str, tuple[int, int, int]]]:
    """Trả về list (cls_idx, cls_name, rgb) của các class có mặt trong label (bỏ bg + 255)."""
    present = sorted({int(c) for c in np.unique(label_hw)} - {0, 255})
    out = []
    for c in present:
        name = INDEX_TO_CLASS.get(c, f"class_{c}")
        rgb = tuple(int(x) for x in VOC_CMAP[c])
        out.append((c, name, rgb))
    return out


_TARGET_COLORS = {
    1: (255, 200, 50),    # diningtable
    2: (220, 50, 100),    # sofa
    3: (50, 180, 220),    # chair
    4: (120, 220, 50),    # bicycle
}


def label_to_rgb_target(label_hw: np.ndarray) -> np.ndarray:
    """Label (H, W) ∈ {0..4} (UNet local id) -> RGB; 0 = đen."""
    h, w = label_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, c in _TARGET_COLORS.items():
        out[label_hw == k] = c
    return out


def mask_to_rgb(mask_bool: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Binary mask (H, W) -> RGB ảnh (H, W, 3) với màu nhất định, bg đen."""
    h, w = mask_bool.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask_bool] = color
    return out


def overlay_mask_on_image(
    image_rgb: np.ndarray,
    mask_bool: np.ndarray,
    color: tuple[int, int, int] = (255, 80, 0),
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlay 1 binary mask lên ảnh RGB với màu chọn. Vùng ngoài mask = ảnh gốc."""
    rgb = image_rgb.astype(np.float32)
    out = rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    if mask_bool.any():
        out[mask_bool] = alpha * color_arr + (1.0 - alpha) * rgb[mask_bool]
    return out.clip(0, 255).astype(np.uint8)


def label_to_voc_png_bytes(label_hw: np.ndarray) -> bytes:
    """Label (H, W) int -> VOC indexed PNG bytes (với palette chuẩn)."""
    from io import BytesIO

    img = Image.fromarray(label_hw.astype(np.uint8), mode="P")
    # Flatten palette: [r0,g0,b0, r1,g1,b1, ...]
    palette = VOC_CMAP.flatten().tolist()
    img.putpalette(palette)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
