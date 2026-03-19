# 😷 Face Mask Detection

A real-time face mask detection system built as a Final Year Project using **MobileNetV2**, **OpenCV**, and **TensorFlow/Keras**. The model detects whether a person is wearing a face mask or not using a live webcam feed.

---

## 📸 Demo

> Real-time detection with bounding boxes labeled **Mask** ✅ or **No Mask** ❌

---

## 🧠 How It Works

The project has two main components:

1. **Training** (`train_mask_detector.py`) — Fine-tunes a MobileNetV2 model on a dataset of masked/unmasked faces.
2. **Detection** (`detect_mask_video.py`) — Uses a pre-trained Caffe SSD face detector to find faces in a video stream, then classifies each face using the trained mask model.

### Architecture
- **Base Model**: MobileNetV2 (pretrained on ImageNet, frozen layers)
- **Head**: AveragePooling2D → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(2, Softmax)
- **Face Detector**: OpenCV DNN with `res10_300x300_ssd_iter_140000.caffemodel`

---

## 📁 Project Structure

```
face-mask-detection/
│
├── dataset/
│   ├── with_mask/          # Images of people with masks
│   └── without_mask/       # Images of people without masks
│
├── face_detector/
│   ├── deploy.prototxt          # Face detector architecture
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained face detector weights
│
├── train_mask_detector.py  # Script to train the mask classifier
├── detect_mask_video.py    # Real-time video mask detection
├── mask_detector.model     # Saved trained model (generated after training)
├── plot.png                # Training accuracy/loss plot (generated after training)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/face-mask-detection.git
cd face-mask-detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🗃️ Dataset

The dataset should be organized into two folders inside `dataset/`:
- `with_mask/` — face images with masks
- `without_mask/` — face images without masks

You can use the [Face Mask Dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) or any similar dataset.

---

## 🏋️ Training

```bash
python train_mask_detector.py
```

This will:
- Load and preprocess images from the `dataset/` folder
- Train the MobileNetV2-based model for **20 epochs**
- Save the trained model as `mask_detector.model`
- Save a training plot as `plot.png`

---

## 🎥 Real-Time Detection

Make sure `mask_detector.model` exists (run training first), then:

```bash
python detect_mask_video.py
```

- Opens your webcam feed
- Detects faces and classifies them in real time
- Press **`q`** to quit

---

## 📊 Model Performance

| Metric     | Value     |
|------------|-----------|
| Epochs     | 20        |
| Batch Size | 32        |
| Input Size | 224×224   |
| Optimizer  | Adam      |
| Loss       | Binary Crossentropy |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model training |
| MobileNetV2 | Transfer learning backbone |
| OpenCV | Video stream & face detection |
| imutils | Video stream utilities |
| scikit-learn | Label encoding & evaluation |
| matplotlib | Training plots |
| NumPy | Array operations |

---

## 👨‍💻 Author

**Syed Aleem**  
Final Year Project — 2025  

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
