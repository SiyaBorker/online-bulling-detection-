# 🛡️ Cyberbullying Detection in Twitter Text Using DistilBERT

> Automatically detect toxic and cyberbullying content in tweets using fine-tuned DistilBERT — a transformer-based NLP model.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 👥 Team

| Name | Registration No. |
|---|---|
| Siya Borker | 23BAI1375 |
| Aditya Kachhot | 23BAI1373 |
| Vaibhavi Jaiswal | 23BAI1484 |

**Institution:** Vellore Institute of Technology (VIT)
**Course:** Speech and Language Processing (Project)

---

## 📌 Problem Statement

Manual moderation of large volumes of user-generated content on social media is difficult and time-consuming. This project automatically classifies tweets as **Bullying** or **Not Bullying** using NLP, helping platforms reduce exposure to harmful language and improve online safety.

---

## 📊 Results

| Metric | Baseline (TF-IDF + LR) | Fine-tuned DistilBERT |
|---|---|---|
| Accuracy | 0.87 | **0.84** |
| Macro F1 | 0.69 | **0.76** |
| Bullying F1 | 0.93 | **0.90** |
| Not Bullying F1 | 0.46 | **0.62** |
| Not Bullying Recall | 0.33 | **0.78** ⬆️ |

> Key improvement: Not Bullying Recall jumped from **0.33 → 0.78** after applying class weights to handle the 1:5 class imbalance.

---

## 🗂️ Project Structure

```
cyberbullying-detection/
│
├── cyberbullying_detection.py   # Main training & evaluation script
├── baseline_comparison.py       # TF-IDF + Logistic Regression baseline
├── requirements.txt             # All dependencies
├── README.md                    # You are here
│
└── model_output/                # Generated after training
    ├── training_curves.png
    ├── confusion_matrix_test.png
    ├── class_distribution.png
    ├── test_metrics.csv
    └── (saved model files)
```

---

## 📦 Dataset

**Cyberbullying Classification Dataset** by Andrew MVD on Kaggle

🔗 https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

The dataset contains ~47,000 tweets labeled across 6 categories:
`religion`, `age`, `ethnicity`, `gender`, `other_cyberbullying`, `not_cyberbullying`

We convert this to **binary classification**:
- `not_cyberbullying` → **0 (Not Bullying)**
- everything else → **1 (Bullying)**

> **Note:** Download the CSV from Kaggle and place `cyberbullying_tweets.csv` in the root of this project before running.

---

## ⚙️ Setup & Installation

### Option A — Google Colab (Recommended, Free GPU)

**Step 1:** Open [colab.research.google.com](https://colab.research.google.com) → New Notebook

**Step 2:** Enable GPU
```
Runtime → Change runtime type → T4 GPU → Save
```

**Step 3:** Upload files — run this in Cell 1:
```python
from google.colab import files
uploaded = files.upload()
# Upload: cyberbullying_detection.py, baseline_comparison.py, cyberbullying_tweets.csv
```

**Step 4:** Install and run — Cell 2:
```python
!pip install transformers scikit-learn seaborn -q
!python cyberbullying_detection.py
```

**Step 5:** Download results — Cell 3:
```python
from google.colab import files
import shutil
shutil.make_archive("model_output", "zip", "model_output")
files.download("model_output.zip")
```

---

### Option B — Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cyberbullying-detection.git
cd cyberbullying-detection

# 2. Create virtual environment
python -m venv cyberbully_env

# 3. Activate it
# Windows:
cyberbully_env\Scripts\activate
# Mac/Linux:
source cyberbully_env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add dataset
# Download cyberbullying_tweets.csv from Kaggle and place it here

# 6. Run baseline first
python baseline_comparison.py

# 7. Run full training
python cyberbullying_detection.py
```

> ⚠️ Local training on CPU takes ~2–4 hours. GPU (CUDA) takes ~20–25 minutes.

---

## 🧠 Model Architecture

```
Input Tweet
    ↓
Text Preprocessing (lowercase, remove URLs, @mentions, special chars)
    ↓
DistilBERT Tokenizer (max_length=128)
    ↓
DistilBERT Base Uncased (fine-tuned)
    ↓
Classification Head (Linear → 2 classes)
    ↓
Output: Not Bullying (0) / Bullying (1)
```

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | distilbert-base-uncased |
| Max Token Length | 128 |
| Batch Size | 32 |
| Epochs | 4 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Scheduler | Linear with warmup |
| Loss Function | CrossEntropyLoss (weighted) |
| Class Weights | Balanced (handles 1:5 imbalance) |

---

## 🔍 Inference Example

```python
from cyberbullying_detection import predict, clean_tweet
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Load saved model
model     = DistilBertForSequenceClassification.from_pretrained("model_output")
tokenizer = DistilBertTokenizerFast.from_pretrained("model_output")

# Predict on new tweets
tweets = [
    "I hope you have a wonderful day!",
    "you are so ugly, nobody likes you",
]
results = predict(tweets, model, tokenizer)

for r in results:
    print(f"[{r['label']}] {r['confidence']} — {r['text']}")
```

**Output:**
```
[Not Bullying] 94.2% — I hope you have a wonderful day!
[Bullying]     97.8% — you are so ugly, nobody likes you
```

---

## 📈 Outputs Generated

After training, the `model_output/` folder contains:

| File | Description |
|---|---|
| `training_curves.png` | Loss & Macro F1 across 4 epochs |
| `confusion_matrix_test.png` | Predicted vs Actual labels |
| `class_distribution.png` | Sample counts per split |
| `test_metrics.csv` | Final evaluation numbers |
| `config.json` + model files | Saved model for inference |

---

## ⚖️ Ethics & Data Considerations

| Concern | Mitigation |
|---|---|
| **Privacy** | Only publicly available tweet text used; no personal identifiers stored |
| **Consent** | Data sourced from open Kaggle dataset intended for research |
| **Bias** | Class-balanced training weights; diverse evaluation metrics reported |
| **Sensitive Content** | Model outputs used for detection support, not automated punishment |

---

## 📚 References

1. Philipo et al., *"Cyberbullying Detection: A Systematic Review"*, 2024 — reviews datasets, models, metrics. [arxiv.org/abs/2407.12154](https://arxiv.org/abs/2407.12154)
2. Yi et al., *"Detecting Harassment and Defamation in Cyberbullying"*, 2023 — benefits of deep contextual models. [arxiv.org/abs/2301.07861](https://arxiv.org/abs/2301.07861)
3. Rosa et al., *"Cyberbullying Detection Using Transformer-Based Models"*, 2022 — BERT-like model effectiveness. [ieeexplore.ieee.org/document/9746037](https://ieeexplore.ieee.org/document/9746037)
4. Al-Garadi et al., *"ML and DL Approaches for Cyberbullying Detection"*, 2021 — model comparison. [sciencedirect.com/science/article/pii/S0167404821001356](https://www.sciencedirect.com/science/article/pii/S0167404821001356)
5. Kaggle Dataset — labeled tweets for training. [kaggle.com/datasets/andrewmvd/cyberbullying-classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)

---

## 📄 License

This project is for academic purposes under VIT's Speech and Language Processing course.
