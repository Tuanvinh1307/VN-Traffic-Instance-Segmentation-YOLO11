# VN-Traffic-Instance-Segmentation-YOLO11

An instance segmentation model that detects and segments vehicles and people in Vietnamese street traffic, built on YOLO11x-seg. Labels were generated semi-automatically with SAM 2, then cleaned up by hand in CVAT.

This started as my graduation thesis project at the Faculty of Information Technology, Ho Chi Minh City University of Foreign Languages - Information Technology (HUFLIT).

## Demo

![demo](demo/demo.gif)

The full-length videos (30 FPS and a smoothed 15 FPS version) are in the [`demo/`](demo/) folder. The model draws segmentation masks and bounding boxes in real time on real traffic footage recorded in Vietnam. A small deque buffer smooths the object counts across frames, so the count doesn't jump around when the model misses a detection for a frame or two.

| Sample frame 1 | Sample frame 2 | Sample frame 3 |
|---|---|---|
| ![](results/sample_predictions/frame_01.jpg) | ![](results/sample_predictions/frame_02.jpg) | ![](results/sample_predictions/frame_03.jpg) |

## Why this project

Most public traffic segmentation datasets (BDD100K, Cityscapes, etc.) were collected in Europe or the US, where vehicle density is low and motorbikes are rare. Vietnamese traffic looks nothing like that: motorbikes dominate, lane discipline is loose, and many vehicle types overlap in the same frame. A model trained only on those foreign datasets tends to merge motorbikes that are packed close together into a single mask, or miss them entirely.

So I rebuilt a training set starting from BDD100K, added real Vietnamese traffic images on top, and trained/evaluated YOLO11x-seg specifically for this setting.

## Dataset

- **Training set:** 9,500 images — 7,000 base images + 2,000 additionally labeled images + 500 real photos taken in Vietnam.
- **Independent test set:** 1,247 real Vietnamese traffic images, fully labeled with segmentation polygons.
- **Labeling process:** SAM 2 produces a rough mask for each object, then I fix the edges and assign the class manually in CVAT. This cut down labeling time a lot compared to drawing every polygon from scratch.

10 object classes:

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | bicycle | 5 | person |
| 1 | bus | 6 | rider |
| 2 | car | 7 | trailer |
| 3 | caravan | 8 | train |
| 4 | motorcycle | 9 | truck |

The dataset itself (image + label zip files) isn't included in this repo because of its size. Download link: *(add your Google Drive / Kaggle / Hugging Face Datasets link here)*.

## Results

Trained for 100 epochs; the best checkpoint came from epoch 52:

| Metric | Value |
|---|---|
| Mask mAP@50 | 33.31% |
| Mask mAP@50:95 | 18.43% |
| Box mAP@50 | 38.59% |
| Box mAP@50:95 | 26.33% |

Mask mAP@50:95 is on the low side, and that's expected given the data: dense clusters of motorbikes cause a lot of mask overlap, and the strict 0.5–0.95 IoU range punishes even small errors on the mask boundary. The per-epoch loss/mAP curves are in `results/` (generated from the training notebook).

## Pipeline

```
Raw images (BDD100K + Vietnam images)
        │
        ▼
  SAM 2 (rough mask) ──► CVAT (manual edge cleanup + class labeling)
        │
        ▼
  data_docker.yaml (merges images/test + images/vn_500 into Train)
        │
        ▼
  YOLO11x-seg (pretrained → fine-tuned, imgsz=640, batch=16, 100 epochs)
        │
        ▼
  best.pt ──► Video inference (OpenCV, deque smoothing) ──► output video
```

## Repo structure

```
.
├── notebooks/
│   ├── 01_train_yolo11_bdd100k_seg.ipynb   # data setup, training, log export, charts, validation
│   └── 02_test_video_inference.ipynb        # loads best.pt, runs on video, exports the result
├── docs/
│   └── huong_dan_su_dung.pdf                # detailed setup & usage guide (Vietnamese)
├── results/
│   └── sample_predictions/                  # sample frames from the demo videos
├── demo/
│   ├── demo.gif
│   ├── demo_real_time_30fps.mp4
│   └── demo_real_time_15fps.mp4
├── requirements.txt
└── README.md
```

## Setup

Needs Python 3.10–3.12. For training, an NVIDIA GPU with 16GB+ VRAM is recommended (RTX 3090/4090, Colab A100/L4, etc.). For inference only, 4GB+ VRAM or even a CPU works fine.

```bash
git clone https://github.com/Tuanvinh1307/VN-Traffic-Instance-Segmentation-YOLO11.git
cd VN-Traffic-Instance-Segmentation-YOLO11
pip install -r requirements.txt
```

If you have an NVIDIA GPU with CUDA 12.x:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

1. Download the dataset from the link in the Dataset section above and unzip it so you have a `data.yaml` file.
2. Open `notebooks/01_train_yolo11_bdd100k_seg.ipynb`, update the dataset path in Section 1, and run the cells in order. The notebook builds `data_docker.yaml` automatically, writes per-epoch results to `trainlog.txt`, and validates on the 1,000-image val split at the end.
3. Once you have `best.pt`, open `notebooks/02_test_video_inference.ipynb`, set `MODEL_PATH` to your weights file and `VIDEO_DIR` to the folder with your test videos, then run it to get an annotated output video.

Step-by-step troubleshooting (missing `data.yaml`, CUDA out of memory, missing `ultralytics` module, etc.) is in [`docs/huong_dan_su_dung.pdf`](docs/huong_dan_su_dung.pdf).

## Limitations & next steps

- Mask accuracy is weaker on classes that barely show up in the Vietnamese images (caravan, train) — there just isn't much data for them.
- The demo video was shot in daylight with good visibility; I haven't tested night driving or rain yet.
- Next steps I'm planning: add more Vietnamese footage at night, try augmentation that simulates rain/glare, and benchmark the smaller YOLO11-seg variants (n/s/m) to see the speed/accuracy trade-off for edge deployment.

## Author

Trang Tuan Vinh
Advisor: MSc. La Nhu Hai — Faculty of Information Technology, HUFLIT.

## License

MIT — see [LICENSE](LICENSE).
