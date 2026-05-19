# SAM3 VOC2012 Streamlit App

Demo Streamlit cho bài toán inference ảnh với SAM3 kết hợp các checkpoint UNet/ASPP và LoRA fine-tuned. Ứng dụng chạy trong Docker, còn thư mục `weights/` được mount từ máy host vì checkpoint rất nặng và không nên push lên GitHub.

## 1. Cấu Trúc Dự Án

```text
.
├── Dockerfile
├── README.md
├── .dockerignore
├── .gitignore
├── streamlit_app/
│   ├── README.md
│   ├── *.ipynb
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

Ý nghĩa các file/folder chính:

- `Dockerfile`: định nghĩa môi trường chạy app bằng Docker. Image copy code trong `streamlit_app/streamlit_app/`, cài Python dependencies, cài SAM3 từ GitHub và chạy Streamlit ở port `8501`.
- `.dockerignore`: loại bỏ checkpoint, cache, notebook và file nặng khỏi Docker build context.
- `.gitignore`: loại bỏ `weights/`, checkpoint và cache khỏi Git.
- `streamlit_app/streamlit_app/app.py`: giao diện Streamlit.
- `streamlit_app/streamlit_app/pipeline.py`: pipeline inference SAM3, UNet/ASPP, DINOv2 và LoRA.
- `streamlit_app/streamlit_app/models.py`: kiến trúc các model UNet/ASPP.
- `streamlit_app/streamlit_app/viz.py`: hàm hiển thị, colormap và overlay kết quả.
- `streamlit_app/streamlit_app/requirements.txt`: các thư viện Python cần cho app.
- `weights/`: chứa checkpoint/model weight. Folder này không push lên GitHub vì rất nặng.

## 2. Tải Weights

Do `weights/` rất nặng, repo không chứa folder này. Người dùng cần tải weights riêng từ link sau:

```text
TODO: dán link tải weights ở đây
```

Sau khi tải, giải nén hoặc đặt folder `weights/` tại root project:

```text
Sam3_voc2012/
└── weights/
    ├── sam3.pt
    ├── unet_aspp_best.pth
    ├── unet_aspp_new_v2_best.pth
    ├── unet_aspp_new_v3_best.pth
    └── sam3_voc_lora_v16_best/
```

Tên file cần giữ đúng như trên vì app mặc định đọc checkpoint từ `/weights`.

## 3. Build Docker Image

Chạy tại root project:

```powershell
cd D:\Sam3_voc2012
docker build -t sam3-voc2012 .
```

Nếu muốn build lại sạch hoàn toàn:

```powershell
docker build --no-cache -t sam3-voc2012 .
```

## 4. Chạy Bằng Docker

### Chạy CPU

```powershell
docker run --rm -p 8501:8501 -v "${PWD}\weights:/weights" sam3-voc2012
```

### Chạy GPU NVIDIA

```powershell
docker run --rm --gpus all -p 8501:8501 -v "${PWD}\weights:/weights" sam3-voc2012
```

Sau đó mở:

```text
http://localhost:8501
```

Trong giao diện Streamlit, đường dẫn checkpoint mặc định nên là:

```text
/weights/sam3.pt
/weights/unet_aspp_best.pth
/weights/unet_aspp_new_v2_best.pth
/weights/unet_aspp_new_v3_best.pth
/weights/sam3_voc_lora_v16_best
```

## 5. Chạy Bằng Docker Desktop

Nếu chạy bằng nút **Run** trong Docker Desktop, cần thêm cấu hình thủ công:

- Port:
  - Host port: `8501`
  - Container port: `8501`
- Volume:
  - Host path: `D:\Sam3_voc2012\weights`
  - Container path: `/weights`

Nếu không mount volume này, app sẽ báo không tìm thấy `sam3.pt` hoặc các checkpoint `.pth`.

## 6. Kiểm Tra Volume Weights

Có thể kiểm tra container có thấy weights hay không bằng lệnh:

```powershell
docker run --rm -v "${PWD}\weights:/weights" sam3-voc2012 ls -lh /weights
```

Output cần có các file:

```text
sam3.pt
unet_aspp_best.pth
unet_aspp_new_v2_best.pth
unet_aspp_new_v3_best.pth
sam3_voc_lora_v16_best
```

## 7. Lưu Ý GPU

Nếu chọn device `cuda` trong app, container phải được chạy với:

```powershell
--gpus all
```

Kiểm tra GPU trên máy host:

```powershell
nvidia-smi
```

Kiểm tra Docker có thấy GPU:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Nếu Docker không thấy GPU, hãy chạy bằng CPU trong sidebar của app hoặc kiểm tra lại NVIDIA driver, Docker Desktop và WSL2 GPU support.

## 8. Troubleshooting

### Không tìm thấy `sam3.pt`

Nguyên nhân thường là chưa mount `weights/` vào `/weights`.

Lệnh đúng:

```powershell
docker run --rm -p 8501:8501 -v "${PWD}\weights:/weights" sam3-voc2012
```

### Found no NVIDIA driver

Container đang chạy `cuda` nhưng không có GPU được cấp vào container. Dùng lệnh GPU:

```powershell
docker run --rm --gpus all -p 8501:8501 -v "${PWD}\weights:/weights" sam3-voc2012
```

Hoặc chọn `cpu` trong sidebar Streamlit.

### Lỗi import module khi chạy SAM3

Hãy build lại image để Docker cài dependencies từ `requirements.txt` và SAM3:

```powershell
docker build --no-cache -t sam3-voc2012 .
```

### Lỗi dtype `BFloat16` và `Float`

App đã bọc inference SAM3 bằng CUDA autocast `bfloat16` trong `pipeline.py`. Nếu gặp lại lỗi này, hãy build lại image từ code mới nhất:

```powershell
docker build --no-cache -t sam3-voc2012 .
```

