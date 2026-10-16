# 🧠 Project 2 (Week 5): Scene Classification with Transfer Learning

Welcome to my deep learning repository! This project implements a robust scene classifier built with **PyTorch** and a pretrained **ResNet-18** architecture on the Intel Image Classification dataset.

---

### 📊 Key Project Specifications & Results

- **Dataset:** Intel Image Classification (`puneet6060/intel-image-classification`) across 6 categories (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`) with **11,929 training** and **3,000 test** images.
- **Model Architecture:** `ResNet-18` pretrained on ImageNet.
- **Phase 1 — Feature Extraction:** Froze the backbone and trained only the new classification head (**3,078 trainable parameters**) for 3 epochs $\rightarrow$ **90.1% validation accuracy**.
- **Phase 2 — Fine-Tuning:** Unfroze `layer4` (**11,179,590 trainable parameters**) using differential learning rates (`1e-4` for layer4, `1e-3` for the head) for 3 epochs $\rightarrow$ **93.7% validation accuracy** peak.
- **Test Performance (Evaluated Once):** Achieved an outstanding **93.5% test accuracy**.
- **Weakest Classes & Confusions:** **Glaciers** and **mountains** presented the highest mutual confusion due to shared rocky and icy visual textures.
- **Most Instructive Error:** A high-altitude snowy mountain peak was misclassified as a glacier because of overlapping surface ice and rugged terrain characteristics.
- **Model Checkpoint:** Successfully saved fine-tuned weights (`resnet18_intel.pt`, ~44.7 MB) to disk.

### 🚀 What's Next?
- **Week 6:** Expanding into my own custom domain classification challenges!
