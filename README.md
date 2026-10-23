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

# Week 6 - Medical Imaging
# 🩺 Pediatric Pneumonia Classification in Chest X-Rays (ResNet-18)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)
![AUC](https://img.shields.io/badge/Test%20AUC-0.956-brightgreen.svg)

An end-to-end, clinically calibrated computer vision pipeline fine-tuning a **ResNet-18** architecture for automated screening of pediatric chest X-rays. Designed for high-sensitivity emergency triage, the pipeline incorporates weighted loss handling, validation threshold tuning, and **Grad-CAM interpretability** to detect potential shortcut learning.

---

## 📌 Model Visualizations

### Grad-CAM Model Interpretability & Shortcut Detection
> Heatmaps generated on the final convolutional block (`layer4`) show strong intra-pulmonary focus for correct predictions, while identifying potential non-lung shortcut dependencies on false positives.

<img width="594" height="404" alt="image" src="https://github.com/user-attachments/assets/151e5f9a-14b6-49b8-81ce-cc2593b767d5" />


---

## 📋 Datasheet & Model Card

## Week 6: Pneumonia Classification with ResNet-18

**Who benefits:** Triage clinical staff and emergency radiologists in low-resource screening environments. **Why it matters:** Delayed pneumonia diagnosis increases pediatric mortality, while unnecessary antibiotic prescription risks adverse events, drug resistance, and healthcare inflation.

**Dataset:** Chest X-Ray Images (Pneumonia) [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), CC BY 4.0; 5,840 images, 2 classes (`NORMAL` vs `PNEUMONIA`); labels assigned by expert radiological review.

**Data audit:** Imbalance ratio 2.9:1 (Pneumonia majority); exact duplicates 0; test images with a near-duplicate in train 0; split by patient ID, 0 shared groups across train/val/test splits.

**Imbalance handling:** Weighted Cross-Entropy Loss (`nn.CrossEntropyLoss`), weights $W_{\text{NORMAL}} \approx 1.96$, $W_{\text{PNEUMONIA}} \approx 0.67$.

**Results (test, used once):** Macro-F1 0.810; per-class recall: Normal 0.705, Pneumonia 0.985; per-class F1: Normal 0.734, Pneumonia 0.885; for binary: sensitivity 0.985 (98.5%) and specificity 0.705 (70.5%) at threshold 0.640 (set on validation to enforce $\ge 95\%$ target recall).

**Grad-CAM:** The model looks at lower lung parenchyma and hilar regions for correct predictions, but focuses on the retrocardiac border, lower diaphragm, thoracic spine, and lateral "R" orientation markers on misclassifications; shortcut risk: suspected (non-lung anatomical and annotation dependencies).

**Limitations:** Single-center pediatric cohort (ages 1–5), limited demographic diversity, sensitivity to non-standard contrast/patient rotation. Not for autonomous clinical diagnosis or unverified primary triage.

---

## 📊 End-to-End Performance Summary

| Metric | Value | Context / Clinical Meaning |
| :--- | :--- | :--- |
| **Model Architecture** | ResNet-18 | Pretrained on ImageNet, fine-tuned on layer4 & classifier head |
| **ROC AUC** | **0.956** | High overall discriminative performance on unseen test data |
| **Validation Macro-F1** | **0.913** | Peaked at Phase 2, Epoch 2 prior to checkpoint restoration |
| **Test Accuracy** | **84.0%** | Correctly classified 524 / 624 test images |
| **Test Sensitivity** | **98.5%** | Correctly flagged 386 / 390 pneumonia cases |
| **Test Specificity** | **70.5%** | Correctly cleared 138 / 234 normal cases |
| **Test Set PPV** | **84.8%** | Precision evaluated at dataset test prevalence ($62.5\%$) |
| **Clinical Setting PPV** | **37.0%** | Precision calculated via Bayes' Theorem at realistic screening prevalence ($15.0\%$) |

---

## 🛡️ Governance & Clinical Safety

1. **Who could be harmed and how?**
   * **False Negatives (1.5% rate):** Unchecked pneumonia progression leading to severe respiratory distress or sepsis.
   * **False Positives (29.5% rate):** Exposure to unnecessary antibiotic courses, family anxiety, and secondary imaging radiation.

2. **Where was the model not validated?**
   * Adult or geriatric patient populations.
   * Portable bedside X-ray devices with varying tube voltages or non-standard DICOM contrasts.
   * External multi-center health systems or unseen hospital networks.

3. **Human Role in Deployment:**
   * Operates strictly as a **human-in-the-loop secondary triage aid**. A certified radiologist or attending physician must review and confirm all model outputs prior to initiating treatment.

4. **Evidence that would change conclusions:**
   * Multi-site external validation showing sensitivity dropping below $90\%$ or PPV dropping below $25\%$ at $15\%$ disease prevalence.
   * Input ablation experiments showing that cropping out lateral "R" orientation markers reduces overall classification accuracy by $> 5\%$.

---

## 📁 Repository Structure

```text
├── week6_model.pt          # PyTorch model weights checkpoint & metadata
├── notebook.ipynb          # Full execution notebook (Data audit -> Grad-CAM)
├── README.md               # Datasheet, Model Card & Governance documentation
└── assets/                 # Sample heatmaps and evaluation charts
