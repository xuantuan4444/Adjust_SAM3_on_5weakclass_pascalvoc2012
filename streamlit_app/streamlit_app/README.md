# Streamlit Demo — SAM3 + 4 Weight Options

Single-image inference interface using **user-provided prompts**, supporting 4 different model variants:

| Variant   | File / Directory                 | Handling of 4 Weak Classes                    |
|-----------|----------------------------------|-----------------------------------------------|
| OLD       | `unet_aspp_best.pth`             | UNet+ASPP binary sigmoid (4 forward passes)   |
| v2        | `unet_aspp_new_v2_best.pth`      | UNet+ASPP softmax 5-channel (1 forward pass)  |
| v3        | `unet_aspp_new_v3_best.pth`      | UNet+ASPP + DINOv2                            |
| finetune  | `sam3_voc_lora_v16_best/` (PEFT) | Directly fine-tuned SAM3 LoRA                 |

`pottedplant` always uses the SAM3 base model with a dual-prompt strategy.  
All other classes fall back to the standard SAM3 base model.

---

# 1. Usage

## Sidebar

- SAM3 base checkpoint path (always required)
- Select variant (`OLD`, `v2`, `v3`, `finetune`)
  - `.pth` file for the 3 UNet variants
  - **directory** for the finetune variant
- Additional parameters:
  - OLD threshold
  - overlay alpha
  - device

---

## Main Interface

1. Upload an image
2. Enter prompts using the format:

```text
class | prompt
```

Example:

```text
chair       | a wooden chair
sofa        | a sofa with cushions
pottedplant | potted plant with its pot or container
person      | a person standing
cat         | a fluffy cat
custom_obj  | red ball on the floor
```

3. Click:

```text
Run inference
```

---

## Output

- 3 visualization panels:
  - Input image
  - SAM3-only composite
  - Hybrid output (processed by selected variant)

- Per-prompt details:
  - raw SAM3 mask
  - final refined mask

- Download indexed PNG masks (VOC palette):
  - `pred_sam3`
  - `pred_hybrid`

- Optional intermediate visualizations:
  - co-occurrence masks
  - UNet labels / fine-tuned masks
  - plant dual-prompt outputs

---

# 2. Routing Logic

```text
Each prompt = {'class', 'prompt'}
  │
  ├── class ∈ {diningtable, sofa, chair, bicycle}
  │       └── variant ∈ {OLD, v2, v3}
  │               → SAM3 base coarse
  │               + co-occurrence
  │               + UNet refinement
  │
  │           variant == finetune
  │               → directly use fine-tuned SAM3
  │
  ├── class == pottedplant
  │       └── SAM3 base with:
  │           - user prompt
  │           - dual prompts:
  │             ['potted plant with its pot or container', 'flowerpot']
  │
  └── all other classes (including custom classes)
          └── SAM3 base with user prompt @ threshold 0.3
```

The `class` field acts only as a **routing key**.

For example:

```text
class='my_obj'
```

will use the standard SAM3 base branch.

Only the following classes trigger specialized logic:

```text
WEAK_CLASS_SET = {
    diningtable,
    sofa,
    chair,
    bicycle,
    pottedplant
}
```

---

# 3. Notes on the `finetune` Variant

- LoRA adapters are applied only to:
  - Mask Decoder
  - Text Encoder

- The Image Encoder remains frozen.

---

## Fine-tuning Prompts

The fine-tuned model was trained only on the 4 weak classes using simple prompts:

```text
bicycle     -> "bicycle"
chair       -> "chair"
diningtable -> "dining table"
sofa        -> "sofa"
```

> Important:
>
> The closer the user prompt is to these exact texts,
> the better the fine-tuned model generally performs.

---

## VRAM Requirement

The application loads **two SAM3 models simultaneously**:

- base SAM3
- fine-tuned SAM3

This requires:

```text
>= 24 GB VRAM
```

If memory is insufficient:

```text
device=cpu
```

---

# 4. Fixed Thresholds (Matched to Training)

| Component | Threshold |
|---|---|
| SAM3 baseline | `0.3` |
| Co-occurrence | `0.5` |
| Plant inside co-occurrence | `0.3` |
| Plant dual-prompt | `0.3` |
| OLD refine threshold | slider (default = `0.5`) |

---

# 5. Troubleshooting

---

## `ImportError: peft`

Install required packages:

```bash
pip install peft safetensors
```

---

## `unexpected target_modules`

The LoRA adapters were trained on specific SAM3 Mask Decoder and Text Encoder modules.

If your local SAM3 checkpoint differs from the checkpoint used during training,
a mismatch will occur.

Make sure to use the exact same:

```text
sam3.pt
```

used during fine-tuning.

---

## DINOv2 Fails to Load (Offline Environment)

Clone the repository manually:

```bash
git clone https://github.com/facebookresearch/dinov2 \
    ~/.cache/torch/hub/facebookresearch_dinov2_main
```

Or manually copy the weights to:

```text
~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
```

---

## Out-of-Memory (OOM) with `v3` or `finetune`

Try:

```text
device=cpu
```

or reduce:

```text
IMAGE_SIZE
```

inside:

```text
pipeline.py
```

Example:

```text
512 → 384
```

for the UNet branch.

(DINOv2 remains at 518.)

---

## Optional: Merge LoRA into Base SAM3

To reduce memory usage:

```python
from peft import PeftModel

m = PeftModel.from_pretrained(base, lora_dir)
m = m.merge_and_unload()

torch.save(m.state_dict(), 'merged_sam3.pt')
```

Then load:

```text
merged_sam3.pt
```

directly as the SAM3 base model.