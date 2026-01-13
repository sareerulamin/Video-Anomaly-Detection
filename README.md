<div align="center">

# Video Anomaly Detection Utilizing Efficient Spatiotemporal Feature Fusion

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/sareerulamin/Video-Anomaly-Detection?style=social)](https://github.com/sareerulamin/Video-Anomaly-Detection)

Official implementation of **"Video Anomaly Detection Utilizing Efficient Spatiotemporal Feature Fusion"** - *Advanced Intelligent Systems, 2024*

[Paper (PDF)](Advanced%20Intelligent%20Systems%20-%202024%20-%20Ul%20Amin%20-%20Video%20Anomaly%20Detection%20Utilizing%20Efficient%20Spatiotemporal%20Feature%20Fusion.pdf) | [Wiley Journal](https://onlinelibrary.wiley.com/journal/26404567) | [Dataset](#2-download-datasets)

<img src="assets/architecture.png" alt="Model Architecture" width="800"/>

</div>

---

## 📖 About This Paper

This repository implements the methodology presented in **"Video Anomaly Detection Utilizing Efficient Spatiotemporal Feature Fusion"** published in *Advanced Intelligent Systems* (2024). The paper proposes a novel approach to video anomaly detection by leveraging the sEnDec (Simultaneous Encoder-Decoder) architecture combined with ConvLSTM for efficient spatiotemporal feature fusion.

**Key Contributions:**
- Novel sEnDec architecture for efficient feature extraction
- Integration of ConvLSTM for temporal dependency modeling
- State-of-the-art performance on benchmark datasets
- Efficient reconstruction-based anomaly detection framework

📄 **Full Paper:** [Advanced Intelligent Systems - 2024 - Ul Amin - Video Anomaly Detection Utilizing Efficient Spatiotemporal Feature Fusion.pdf](Advanced%20Intelligent%20Systems%20-%202024%20-%20Ul%20Amin%20-%20Video%20Anomaly%20Detection%20Utilizing%20Efficient%20Spatiotemporal%20Feature%20Fusion.pdf)

## Overview

Thi🏗️ Technical Framework

### sEnDec (Simultaneous Encoder-Decoder) Architecture

The **sEnDec block** is the core innovation of this work, combining encoder and decoder operations simultaneously:

```python
def sendec_block(input_tensor1, input_tensor2):
    # Decode previous features
    x = Conv3DTranspose(filters=16, kernel_size=(2, 3, 3), 
                        strides=(1, 2, 2), padding='same')(input_tensor1)
    # Fuse with encoder features via concatenation
    x = concatenate([input_tensor2, x], axis=-1)
    # Refine fused features
    x = BatchNormalization()(x)
    x = Conv3D(filters=16, kernel_size=(1, 3, 3), 
               activation='relu', padding='same')(x)
    return x
```

### Model Architecture

<div align="center">

<img src="results/framework.png" alt="sEnDec CNN-LSTM Framework" width="900"/>

*Figure: Proposed sEnDec CNN-LSTM framework for video anomaly detection*

</div>

| Component | Details |
|-----------|---------|
| **Input Shape** | 4 consecutive grayscale frames (240×320) |
| **Encoder Blocks** | 3× sEnDec blocks with progressive spatial downsampling |
| **Temporal Layer** | ConvLSTM2D (16 filters, 3×3 kernel) for motion modeling |
| **Decoder Path** | 4× Upsampling layers with skip connections from encoder |
| **Output Shape** | Reconstructed frame sequence (4×240×320×1) |
| **Activation** | ReLU (hidden layers), Sigmoid (output) |
| **Normalization** | Batch Normalization between conv operations |

---

## 1. Installation (Anaconda with Python 3.8+ recommended)

### Clone the Repository

```bash
git clone https://github.com/sareerulamin/Video-Anomaly-Detection.git
cd Video-Anomaly-Detection
```

### Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n anomaly python=3.8
conda activate anomaly

# Or using venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow>=2.4.0
keras>=2.4.0
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
Pillow>=8.0.0
```

---

## 2. Download Datasets

Download and organize the datasets in the following structure:

| Dataset | Description | Download Link |
|---------|-------------|---------------|
| **CUHK Avenue** | 16 training + 21 testing videos | [Avenue Dataset](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) |
| **UCSD Ped1** | 34 training + 36 testing clips | [UCSD Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) |
| **UCSD Ped2** | 16 training + 12 testing clips | [UCSD Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) |

### Dataset Structure

```
Video-Anomaly-Detection/
├── Avenue_Dataset/
│   ├── normal/              # Normal training videos (.mp4, .avi)
│   └── abnormal/            # Abnormal test videos (.mp4, .avi)
├── Avenue/
│   ├── train_set/
│   │   ├── normal/
│   │   └── abnormal/
│   └── val_set/
│       ├── normal/
│       └── abnormal/
├── ped1/
│   ├── train_set/
│   │   ├── normal/
│   │   └── abnormal/
│   └── val_set/
└── Ped2/
    ├── train_set/
    │   ├── normal/
    │   └── abnormal/
    └── val_set/
        ├── normal/
        └── abnormal/
```

---

## 3. Testing on Saved Models

### Quick Test with Pre-trained Model

```python
from keras.models import load_model
import numpy as np
import cv2

# Load pre-trained model
model = load_model('anomaly_mdel.h5')

# Load and preprocess test video
# ... (see model.ipynb for full implementation)

# Make predictions
reconstructions = model.predict(test_video)

# Calculate anomaly score (reconstruction error)
anomaly_score = np.mean(np.abs(test_video - reconstructions), axis=(1,2,3,4))
```

### Using Jupyter Notebook

```bash
jupyter notebook model.ipynb
```

Run all cells to:
1. Load and visualize data
2. Test the pre-trained model
3. Generate reconstruction visualizations
4. Plot ROC curves and anomaly scores

---

## 4. Training from Scratch

### 4.1 Configure Training Parameters

Edit the following parameters in `model.ipynb`:

```python
# Data parameters
video_folder = 'Avenue_Dataset/normal'  # Path to training videos
num_frames = 4                           # Frames per clip
frame_height = 240                       # Frame height
frame_width = 320                        # Frame width

# Model parameters
input_shape = (num_frames, frame_height, frame_width, 1)
dp = 0.3                                 # Dropout rate

# Training parameters
batch_size = 1
epochs = 5
validation_split = 0.3
```

### 4.2 Run Training

```python
# Create model
model = Models._sEnDec_cnn_lstm(input_shape, dp)

# Compile
model.compile(optimizer=Adadelta(), loss=MeanSquaredError())

# Train
model.fit(train_video, train_video, 
          batch_size=batch_size, 
          epochs=epochs, 
          validation_split=validation_split,
          verbose=1)

# Save model
model.save("anomaly_mdel.h5")
```

### 4.3 Hyperparameters

| Parameter | Default | Description |
|--📊 Experimental Results

### Benchmark Performance (Frame-Level AUC)

The proposed 3D CNN-LSTM architecture achieves **state-of-the-art** results on standard benchmarks:

| Dataset | AUC (%) | Improvement |
|---------|---------|-------------|
| **UCSD Ped1** | **94.5%** | +2.7% over prior methods |
| **UCSD Ped2** | **96.8%** | +0.6% over prior methods |
| **CUHK Avenue** | **93.0%** | +3.4% over prior methods |

### Comparison with State-of-the-Art Methods

| Method | Ped1 | Ped2 | Avenue |
|--------|------|------|--------|
| Lu et al. | 91.8 | – | 80.9 |
| Zhou et al. | 83.5 | 94.9 | 86.1 |
| Tang et al. | 82.6 | 96.2 | 83.7 |
| Wen et al. | 83.1 | 95.4 | 85.1 |
| Yao et al. | 84.5 | 95.9 | 85.9 |
| **Proposed Method** | **94.5** | **96.8** | **93.0** |

### Model Efficiency

| Model | Parameters (M) | Size (MB) | Time/Seq (ms) |
|-------|----------------|-----------|---------------|
| VGG19 + BD-LSTM | 143.00 | 605.50 | 220 |
| Inception V3 + BD-LSTM | 23.00 | 148.50 | 180 |
| ResNet-50 + BD-LSTM | 25.00 | 143.00 | 200 |
| **Proposed Method** | **0.224** | **2.83** | **160** |

### Key Advantages

- 🎯 **Efficient Feature Fusion**: sEnDec blocks enable simultaneous encoding/decoding
- ⚡ **Lightweight Model**: Only 0.224M parameters (2.83 MB)
- 📈 **Fast Inference**: 160ms per sequence (30 frames)
- 🔍 **Superior Detection**: Significant reconstruction error for abnormal frames

### Anomaly Score Visualization

<div align="center">

<img src="results/ascore.png" alt="Anomaly Score Visualization" width="800"/>

*Figure: Anomaly scores across video frames. The blue line represents the anomaly score, with peaks indicating detected anomalies. Ground truth anomalous regions are highlighted in cyan.*

</div>

For detailed experimental validation and ablation studies, refer to the [full paper](Advanced%20Intelligent%20Systems%20-%202024%20-%20Ul%20Amin%20-%20Video%20Anomaly%20Detection%20Utilizing%20Efficient%20Spatiotemporal%20Feature%20Fusion.pdf).

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{ulamin2024video,
  title={Video Anomaly Detection Utilizing Efficient Spatiotemporal Feature Fusion with 3D Convolutions and Long Short-Term Memory Modules},
  author={Ul Amin, Sareer and Kim, Bumsoo and Jung, Yonghoon and Seo, Sanghyun and Park, Sangoh},
  journal={Advanced Intelligent Systems},
  volume={6},
  number={7},
  pages={2300706},
  year={2024},
  publisher={Wiley},
  doi={10.1002/aisy.202300706}
}
```

**Paper Link:** [Advanced Intelligent Systems - 2024 - Ul Amin - Video Anomaly Detection Utilizing Efficient Spatiotemporal Feature Fusion.pdf](Advanced%20Intelligent%20Systems%20-%202024%20-%20Ul%20Amin%20-%20Video%20Anomaly%20Detection%20Utilizing%20Efficient%20Spatiotemporal%20Feature%20Fusion.pdf)5. Results & Visualization

### Reconstruction Comparison

The model reconstructs normal frames accurately while struggling with anomalous content:

| Original Frames | Reconstructed Frames | Reconstruction Error |
|-----------------|---------------------|---------------------|
| Normal activity | Low error | ✅ Normal |
| Anomalous activity | High error | ⚠️ Anomaly Detected |

### Evaluation Metrics

```python
# Calculate ROC curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
roc_auc = auc(fpr, tpr)

# Plot results
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
```

---

## Project Structure

```
Video-Anomaly-Detection/
├── model.ipynb              # 📓 Main notebook (training & evaluation)
├── anomaly_mdel.h5          # 🧠 Pre-trained model weights
├── requirements.txt         # 📦 Python dependencies
├── README.md                # 📖 This file
├── assets/                  # 🖼️ Images and diagrams
│   └── architecture.png
├── Avenue_Dataset/          # 📁 Avenue dataset
├── Avenue/                  # 📁 Avenue dataset (alternate)
├── ped1/                    # 📁 UCSD Ped1 dataset
└── Ped2/                    # 📁 UCSD Ped2 dataset
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{ulamin2024video,
  title={Video Anomaly Detection Utilizing Efficient Spatiotemporal Feature Fusion},
  author={Ul Amin, Sareer and others},
  journal={Advanced Intelligent Systems},
  year={2024},
  publisher={Wiley}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [CUHK Avenue Dataset](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
- [UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)
- TensorFlow and Keras teams

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[![GitHub Stars](https://img.shields.io/github/stars/sareerulamin/Video-Anomaly-Detection?style=social)](https://github.com/sareerulamin/Video-Anomaly-Detection)
[![GitHub Forks](https://img.shields.io/github/forks/sareerulamin/Video-Anomaly-Detection?style=social)](https://github.com/sareerulamin/Video-Anomaly-Detection/fork)

**Questions?** Open an [Issue](https://github.com/sareerulamin/Video-Anomaly-Detection/issues)

</div>
