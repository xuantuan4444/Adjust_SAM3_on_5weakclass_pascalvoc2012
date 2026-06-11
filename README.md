# SAM3 VOC2012 Streamlit App

A Streamlit demo application for image inference using **SAM3** combined with multiple UNet/ASPP checkpoints and LoRA fine-tuned models.

The application is designed to run inside Docker, while the `weights/` directory is mounted from the host machine because the checkpoints are extremely large and should not be pushed to GitHub.

---

# 1. Project Structure

```text
.
├── Dockerfile
├── README.md
├── .dockerignore
├── streamlit_app/ 
│   └── streamlit_app/
│       ├── app.py
│       ├── pipeline.py
│       ├── models.py
│       ├── viz.py
│       ├── requirements.txt
│       └── README.md
└── weights/
    ├── sam3.pt
    ├── unet_aspp_best.pth
    ├── unet_aspp_new_v2_best.pth
    ├── unet_aspp_new_v3_best.pth
    └── sam3_voc_lora_v16_best/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

---

# Main Files and Folders

| File / Folder | Description |
|---|---|
| `Dockerfile` | Defines the Docker runtime environment for the application |
| `.dockerignore` | Excludes checkpoints, cache files, notebooks, and heavy files from Docker build context |
| `.gitignore` | Excludes `weights/`, checkpoints, and cache files from Git |
| `streamlit_app/streamlit_app/app.py` | Main Streamlit user interface |
| `streamlit_app/streamlit_app/pipeline.py` | SAM3 + UNet/ASPP + DINOv2 + LoRA inference pipeline |
| `streamlit_app/streamlit_app/models.py` | UNet/ASPP model architectures |
| `streamlit_app/streamlit_app/viz.py` | Visualization utilities, overlays, and VOC colormap |
| `streamlit_app/streamlit_app/requirements.txt` | Python dependencies |
| `weights/` | Contains all checkpoints and model weights |

---

# 2. Download Weights

The repository does not include the `weights/` directory because the files are too large for GitHub.

Download the weights from:

```text
https://drive.google.com/drive/folders/1xPDGTGXI1LbXVc6BKm4re0X-RnsPGxB9?usp=sharing
```

After downloading, place the `weights/` directory at the project root:

```text
Sam3_voc2012/
└── weights/
    ├── sam3.pt
    ├── unet_aspp_best.pth
    ├── unet_aspp_new_v2_best.pth
    ├── unet_aspp_new_v3_best.pth
    └── sam3_voc_lora_v16_best/
```

The filenames must remain unchanged because the application loads checkpoints from `/weights` by default.

---

# 3. Build Docker Image

Run from the project root:

```powershell
cd D:\Sam3_voc2012

docker build -t sam3-voc2012 .
```

To rebuild from scratch without cache:

```powershell
docker build --no-cache -t sam3-voc2012 .
```

---

# 4. Run with Docker

## CPU Mode

```powershell
docker run --rm -p 8501:8501 -v "${PWD}\weights:/weights" sam3-voc2012
```

---

## NVIDIA GPU Mode

```powershell
docker run --rm --gpus all -p 8501:8501 -v "${PWD}\weights:/weights" sam3-voc2012
```

After launching, open:

```text
http://localhost:8501
```

---

# Default Checkpoint Paths

Inside the Streamlit interface, the default checkpoint paths should be:

```text
/weights/sam3.pt
/weights/unet_aspp_best.pth
/weights/unet_aspp_new_v2_best.pth
/weights/unet_aspp_new_v3_best.pth
/weights/sam3_voc_lora_v16_best
```

---

# 5. Running via Docker Desktop

If using the **Run** button in Docker Desktop, manually configure:

## Ports

| Host | Container |
|---|---|
| `8501` | `8501` |

---

## Volumes

| Host Path | Container Path |
|---|---|
| `D:\Sam3_voc2012\weights` | `/weights` |

If this volume is not mounted, the application will fail to locate:

- `sam3.pt`
- `.pth` checkpoints
- LoRA adapters

---

# 6. Verify Mounted Weights

You can verify that the container detects the weights correctly:

```powershell
docker run --rm -v "${PWD}\weights:/weights" sam3-voc2012 ls -lh /weights
```

Expected output:

```text
sam3.pt
unet_aspp_best.pth
unet_aspp_new_v2_best.pth
unet_aspp_new_v3_best.pth
sam3_voc_lora_v16_best
```

---

# 7. GPU Notes

If selecting `cuda` inside the application, the container must be started with:

```powershell
--gpus all
```

---

## Check GPU on Host Machine

```powershell
nvidia-smi
```

---

## Check GPU Visibility Inside Docker

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If Docker cannot detect the GPU:

- switch to CPU mode inside Streamlit
- verify NVIDIA drivers
- verify Docker Desktop GPU support
- verify WSL2 GPU support

---

# 8. Troubleshooting

---

## `sam3.pt` Not Found

Usually caused by missing volume mounting.

Correct command:

```powershell
docker run --rm -p 8501:8501 -v "${PWD}\weights:/weights" sam3-voc2012
```

---

## `Found no NVIDIA driver`

The container is attempting to use CUDA without GPU access.

Run with GPU support:

```powershell
docker run --rm --gpus all -p 8501:8501 -v "${PWD}\weights:/weights" sam3-voc2012
```

Or switch to `cpu` inside the Streamlit sidebar.

---

## SAM3 Import Errors

Rebuild the Docker image:

```powershell
docker build --no-cache -t sam3-voc2012 .
```

This reinstalls dependencies and SAM3 correctly.

---

## `BFloat16` / `Float` dtype Errors

The application already wraps SAM3 inference using CUDA autocast `bfloat16` inside `pipeline.py`.

If the error still occurs, rebuild the image from the latest source:

```powershell
docker build --no-cache -t sam3-voc2012 .
```

---

# Notes

- The `weights/` directory is intentionally excluded from GitHub.
- Docker volume mounting is required.
- The application supports:
  - SAM3 baseline inference
  - UNet/ASPP refinement
  - DINOv2 refinement
  - LoRA fine-tuned SAM3
- Streamlit runs on port `8501`.

---

# Technologies Used

- Streamlit
- PyTorch
- SAM3
- DINOv2
- UNet
- ASPP
- PEFT / LoRA
- Docker
- CUDA
- OpenCV