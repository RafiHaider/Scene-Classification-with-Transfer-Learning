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
```
# **Week 7: Text and Multimodal AI**

- **Text Classifier**: Benchmark Dataset, 1,000+ texts, Macro-F1 0.89 (word) vs 0.92 (char)
- **Top Positive / Negative Features**: `['urgent', 'fever', 'cough', 'invoice']` / `['routine', 'thanks', 'ok']`
- **NER on My Domain**: Out-of-the-box F1 0.65 $\rightarrow$ with Entity Ruler F1 0.94
- **Regex Fields Extracted**: `CNIC`, `Mobile Number`, `Invoice No`, `Amount`, `Date`; False positives tested with negative samples
- **OCR**: Character Error Rate (CER) raw 0.032 $\rightarrow$ after correction rules CER 0.000 (0.062 $\rightarrow$ 0.000 on digit substitution)
- **Pipeline**: Image vision model ($P(\text{Pneumonia}) = 0.88$) + Text evidence (`fever`, `breath`) $\rightarrow$ Structured JSON case record with automated decision (`REFER`)
- **Privacy**: CNIC, phone, and email masked before logging

---

### 💡 Key Technical Highlights

#### 1. Context-Aware PII Masking
Ensures sensitive user identifiers are redacted before downstream API/model processing:
```json
"note_masked": "Child aged 4, fever for 3 days and fast breath. Guardian mobile [PHONE]."
```
#### 2. OCR Post-Processing Heuristics
Repairs digit-letter confusion in scanned invoices:
- `Invoice N0. 4417` $\rightarrow$ `Invoice No. 4417` (CER reduced from **0.062** to **0.000**)

#### 3. Multimodal Clinical Triage Pipeline
```json
{
  "image": "invoice.png",
  "image_probabilities": { 
    "Normal": 0.12, 
    "Pneumonia": 0.88 
  },
  "decision": "REFER",
  "reasons": [
    "image model: P(Pneumonia) = 0.88",
    "note mentions: fever, breath"
  ],
  "disclaimer": "Decision support only; a qualified human decides."
}
```
# Week 8
# 🚀 P2-API: FastAPI Deep Learning Inference Service

A production-ready FastAPI backend serving deep learning image classification models with robust input validation, error handling, automated testing, and latency benchmarking. Developed for Lab 8 coursework.

---

## 📂 Project Structure
p2-api/

├── .venv/             # Isolated Python virtual environment

├── app.py   # Main FastAPI application & endpoint definitions

├── test_api.py # Pytest suite (includes parity and input validation tests)

├── bench.py    # Latency benchmarking script (n=50)

├── week6_model.pt   # PyTorch trained model weights

├── sample.jpeg   # Default validation test image

└── README.md            # Project documentation

---
###  Swagger UI Overview (App Interface):
<img width="512" height="272" alt="pic" src="https://github.com/user-attachments/assets/e5db2b09-d863-4b83-b39e-092f711b84f7" />

## 🛠️ API Endpoints

* **GET /health**
  *Description: Health check endpoint to verify server status, confirm model loading, and list supported classification classes.
  * Response:
   ``` JSON
  {
   "status": "ok",
   "model_loaded": true,
   "classes": [
     "NORMAL",
     "PNEUMONIA"
   ],
   "positive": "PNEUMONIA"
  }
Health Check (/health):
![Picture 2: GET /health Endpoint Response](<img width="739" height="576" alt="pic1" src="https://github.com/user-attachments/assets/2db15870-e4a3-47d2-ab1e-4fbfa4afac5a" />

)

* **POST /predict**
  Description: Accepts an image file upload (sample.jpeg, with optional x-api-key header), validates format constraints, runs inference through the PyTorch model, and returns class probabilities and latency.
  Image Prediction (/predict):
![Picture 3: POST /predict Request and Live Response](<img width="512" height="405" alt="pic2" src="https://github.com/user-attachments/assets/1fca7497-23f9-47a2-8e2d-cc9ec0135b7e" /> <img width="512" height="247" alt="pic_2" src="https://github.com/user-attachments/assets/5679b892-1e7e-469b-9884-18a856de6f8c" />

)
* **POST /analyze**
 Description: Accepts an image file upload along with a text note (and optional x-api-key header) to perform combined vision prediction, note handling, urgency/decision routing, and decision-support output.
Combined Vision + Note Analysis (/analyze):
![Picture 6: POST /analyze Response Schemas (200 & 422)](<img width="512" height="425" alt="pic3" src="https://github.com/user-attachments/assets/50c50393-67f8-4b58-9150-104587fdc202" /> <img width="732" height="562" alt="pic_3" src="https://github.com/user-attachments/assets/4aa58f4d-575a-47ed-94e4-f87996476279" />

)
---

## ⚙️ Local Setup & Execution

1. Clone the repository or navigate to the project directory:
   ``` Bash
   cd p2-api

3. Create and activate a virtual environment:
    ``` Bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

5. Install required dependencies:
    ``` Bash
   pip install fastapi uvicorn torch torchvision pillow pytest httpx

7. Run the FastAPI server:
    ``` Bash
   uvicorn app:app --reload
  (The server will run locally at http://127.0.0.1:8000/docs)

---
## 📸 Application & Endpoint Walkthrough
1. Swagger UI Overview (App Interface)
   Displays the Project 2 Vision + Text API Swagger documentation interface with the /health, /predict, and /analyze endpoints along with request/response schemas.
2. Displays the Project 2 Vision + Text API Swagger documentation interface with the /health, /predict, and /analyze endpoints along with request/response schemas.
3. mage Prediction (/predict)
Uploading sample.jpeg via multipart/form-data to POST /predict returns HTTP 200 with the predicted label ("NORMAL"), per-class probability scores, and inference latency (latency_ms), along with documented 200 and 422 response schemas.
4. Combined Vision + Note Analysis (/analyze)
Submitting sample.jpeg and a note to POST /analyze returns HTTP 200 containing the vision prediction (label, probabilities, latency_ms), note_masked, fields, urgent_terms, decision, reasons, and decision-support disclaimer, followed by the full 200 and 422 response schemas.



## 🧪 Testing & Performance Metrics

1. **Automated Testing (Pytest)**
   Run the test suite to verify endpoint correctness, input schema validation, and model parity:
    ``` Bash
   pytest -v test_api.py
  * **Result:** 8 passed successfully.

3. **Latency Benchmarking**
   To test throughput and latency metrics under load (n=50 requests):
    ``` Bash
   python bench.py
  * **Hardware Environment:** Windows Host Workstation
  * **Latency (p50):** 87 ms
  * **Latency (p95):** 115 ms
  * **Max Latency:** 134 ms
  * **Throughput (1 worker):** ~11.0 requests/s

---

## 📊 Little's Law Capacity Estimation
Using Little's Law:
* At a p50 latency of 87 ms (0.087 s), a single worker handles ~11.49 requests/second.
* Scaled to a full 24-hour day (86,400 seconds):
  Daily Capacity = 11.494 * 86,400 = ~993,100 requests/day per worker.

---

## 🛡️ Error Handling
The API includes strict request validation and handles invalid or malformed inputs safely:
* **Invalid File Type/Schema Mismatch:** Returns 415 Unsupported Media Type or 422 Unprocessable Entity.
* **Missing Payload:** Handled via Pydantic schemas without exposing raw stack traces.





