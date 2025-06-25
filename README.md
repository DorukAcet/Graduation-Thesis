# 🎓 Graduation Thesis – Fish Species Detection with YOLOv11 & Faster R-CNN

This project implements an object detection system to classify various fish species using two different models:
- [YOLOv11](https://github.com/WongKinYiu/yolov7) (real-time inference)
- [Faster R-CNN](https://github.com/open-mmlab/mmdetection) (high-accuracy detector)

A Gradio-based web interface enables users to upload images and compare model results side-by-side.

---

## 🐟 Supported Classes

- Sprat
- Bream
- Mackerel
- Red Mullet
- Red Sea Bream
- Sea Bass
- Striped Red Mullet
- Trout
- Shrimp

---

## 🚀 Features

- 🔍 Object detection with two different models (YOLOv11 + Faster R-CNN)
- ⚖️ Confidence threshold adjustment
- 🧠 Wikipedia integration for each class
- 📊 Side-by-side model comparison
- 📤 Downloadable annotated image results

---

## 🛠️ Installation

```bash
git clone https://github.com/DorukAcet/Graduation-Thesis.git
cd Graduation-Thesis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Running the Web App

```bash
python app.py
```

Then open the Gradio interface at [http://localhost:7860](http://localhost:7860)

---

## 🧪 Run on Google Colab

Test the model directly on Colab using the following notebooks:

### 🔹 Faster R-CNN:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DorukAcet/Graduation-Thesis/blob/main/notebooks/inference_frcnn.ipynb)

### 🔹 YOLOv11:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DorukAcet/Graduation-Thesis/blob/main/notebooks/inference_yolo.ipynb)

---

## 📁 Model Files

Model weights are stored externally due to GitHub file size limits:

| Model        | File           | Download Link                                                                 |
|--------------|----------------|--------------------------------------------------------------------------------|
| YOLOv11      | `best.pt`      | [Download from Google Drive](https://drive.google.com/file/d/1889jOlpJBCfGdjZQKQoBlbYGHNuZjGKf/view?usp=share_link) |
| Faster R-CNN | `epoch_15.pth` | [Download from Google Drive](https://drive.google.com/file/d/1OuJP42G18haHwNE_8afz5C076utZrCPr/view?usp=share_link) |

> After downloading, place the model files into a `models/` folder at the root of the project:
```bash
mkdir models
mv best.pt models/
mv epoch_15.pth models/
```

---

## 📄 License

This project is for academic use only and licensed under the MIT License.

---

## 🙋‍♂️ Author

**Doruk Acet**  
Computer Engineering, Graduation Project 2025  
🔗 [LinkedIn](https://www.linkedin.com/in/doruk-acet-52b7b11b1/)
