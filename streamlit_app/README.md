# Streamlit demo — SAM3 + 4 weight options

Giao diện inference 1 ảnh **theo prompt user nhập**, hỗ trợ 4 weight:

| Variant   | File / dir                       | Cách xử lý 4 weak class               |
|-----------|----------------------------------|---------------------------------------|
| OLD       | `unet_aspp_best.pth`             | UNet+ASPP binary sigmoid (4 forward)   |
| v2        | `unet_aspp_new_v2_best.pth`      | UNet+ASPP softmax 5ch (1 forward)      |
| v3        | `unet_aspp_new_v3_best.pth`      | UNet+ASPP + DINOv2 (1 forward + DINOv2)|
| finetune  | `sam3_voc_lora_v16_best/` (PEFT) | SAM3 LoRA fine-tuned trực tiếp         |

Pottedplant luôn dùng SAM3 base với dual prompt; mọi class khác fallback SAM3 base.

---

## 1. Cấu trúc thư mục

```
streamlit_app/
├── app.py              # Streamlit UI
├── pipeline.py         # Inference pipeline (4 variant)
├── models.py           # Kiến trúc UNet+ASPP (OLD / v2 / v3)
├── viz.py              # VOC colormap + overlay
├── requirements.txt
└── README.md

weights/                 # đặt checkpoint vào đây (hoặc override env)
├── sam3.pt
├── unet_aspp_best.pth
├── unet_aspp_new_v2_best.pth
├── unet_aspp_new_v3_best.pth
└── sam3_voc_lora_v16_best/         # PEFT directory
    ├── adapter_config.json
    └── adapter_model.safetensors
```

> Variant **finetune** lưu dưới dạng **directory PEFT** (output của
> `model.save_pretrained(...)` trong `finetunesam3+7.ipynb`). KHÔNG phải `.pth` file.

---

## 2. Cài đặt

```bash
# Python deps
pip install -r requirements.txt

# SAM3 (Meta)
pip install 'git+https://github.com/facebookresearch/sam3.git' --no-deps
pip install iopath ftfy portalocker

# (Tùy chọn, chỉ v3) DINOv2 — torch.hub tự clone lần đầu. Offline:
git clone https://github.com/facebookresearch/dinov2 \
    ~/.cache/torch/hub/facebookresearch_dinov2_main
```

---

## 3. Chạy

```bash
cd streamlit_app/
streamlit run app.py
```

Mở `http://localhost:8501`.

### Override đường dẫn checkpoint

```bash
WEIGHTS_DIR=/path/to/weights streamlit run app.py
# hoặc đặt riêng từng env
SAM3_CKPT=/custom/sam3.pt \
UNET_CKPT_OLD=/custom/old.pth \
UNET_CKPT_V2=/custom/v2.pth \
UNET_CKPT_V3=/custom/v3.pth \
SAM3_LORA_DIR=/custom/sam3_voc_lora_v16_best \
    streamlit run app.py
```

---

## 4. Sử dụng

1. **Sidebar**:
   - SAM3 base ckpt path (luôn dùng).
   - Chọn variant (OLD / v2 / v3 / finetune) → field weight tương ứng (file `.pth`
     cho 3 UNet, **directory** cho finetune).
   - Tham số (OLD threshold, alpha overlay, device).

2. **Main**:
   - Upload 1 ảnh.
   - Nhập danh sách prompt mỗi dòng theo format: `class | prompt`.
     Ví dụ:
     ```
     chair       | a wooden chair
     sofa        | a sofa with cushions
     pottedplant | potted plant with its pot or container
     person      | a person standing
     cat         | a fluffy cat
     custom_obj  | red ball on the floor
     ```
   - Click **Run inference**.

3. **Output**:
   - 3 ảnh: Input | SAM3-only composite | Hybrid (qua variant đã chọn).
   - Per-prompt detail: SAM3 raw mask vs final mask cho từng prompt.
   - Download indexed PNG (VOC palette) cho cả `pred_sam3` và `pred_hybrid`.
   - Optional: bước trung gian (co-occurrence, UNet label / FT mask, plant dual).

---

## 5. Logic route

```
Mỗi prompt = {'class', 'prompt'}
  │
  ├── class ∈ {diningtable, sofa, chair, bicycle}
  │       └── variant ∈ {OLD, v2, v3} → SAM3 base coarse + co-occurrence + UNet refine
  │           variant == finetune     → finetuned SAM3 trực tiếp với prompt
  │
  ├── class == pottedplant
  │       └── SAM3 base với prompt user + dual prompts
  │           ['potted plant with its pot or container', 'flowerpot']
  │
  └── class khác (kể cả custom)
          └── SAM3 base với prompt user @ thr 0.3
```

`class` chỉ là **key định tuyến** — `class='my_obj'` sẽ vào nhánh SAM3 base.
Chỉ 5 tên trong `WEAK_CLASS_SET = {diningtable, sofa, chair, bicycle, pottedplant}`
mới kích hoạt UNet/finetune/dual.

---

## 6. Ghi chú variant `finetune`

- LoRA chỉ áp lên Mask Decoder + Text Encoder của SAM3, Image Encoder frozen.
- Finetune chỉ training trên 4 lớp yếu, prompts giản nhất:
  ```
  bicycle    -> "bicycle"
  chair      -> "chair"
  diningtable-> "dining table"   ← ghi rời "dining table" (có space)
  sofa       -> "sofa"
  ```
  → khi test, prompt user nhập càng gần các text trên, finetuned model càng phát huy
  tác dụng.

- App load **2 SAM3 cùng lúc**: base (cho non-weak + dual + co_occ) + finetune
  (cho 4 weak). Cần ≥ 24 GB VRAM. Nếu thiếu, dùng `device=cpu`.

---

## 7. Threshold đã fixed (= training)

- SAM3 baseline mỗi prompt: 0.3
- Co-occurrence: 0.5 (plant trong co-occ: 0.3)
- Plant dual: 0.3
- OLD refine threshold: slider, default 0.5.

---

## 8. Troubleshooting

**Q: Load finetuned SAM3 báo `ImportError: peft`**
→ `pip install peft safetensors`.

**Q: PEFT báo `unexpected target_modules`**
→ Adapter LoRA train ở finetune notebook gắn vào Mask Decoder + Text Encoder cụ thể;
nếu base SAM3 ở local khác phiên bản với khi train → mismatch. Đảm bảo dùng cùng
`sam3.pt` lúc train.

**Q: DINOv2 load thất bại (no internet)**
→ Clone repo offline như mục 2, hoặc copy weights thủ công vào
`~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`.

**Q: VRAM cạn khi chọn v3 hoặc finetune**
→ Thử `device=cpu` (chậm), hoặc giảm `IMAGE_SIZE` trong `pipeline.py` (512→384) cho
UNet (DINOv2 vẫn 518). Với finetune, có thể merge LoRA vào base offline:
```python
from peft import PeftModel
m = PeftModel.from_pretrained(base, lora_dir)
m = m.merge_and_unload()
torch.save(m.state_dict(), 'merged_sam3.pt')
```
sau đó load merged_sam3.pt trực tiếp như base SAM3.
