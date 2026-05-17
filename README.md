# 💬 Customer Review Sentiment Analysis
### BERT · LSTM · Bi-LSTM · TF-IDF · NLP · Kaggle · Self-Project

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![BERT](https://img.shields.io/badge/HuggingFace-BERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-Fine--tuning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Best_Model-BERT_(90%25_Accuracy)-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Amazon_Reviews_(Kaggle)-orange?style=flat-square)

---

## 🧩 Problem Statement

Amazon receives millions of product reviews daily — classifying them as positive or negative at scale requires robust NLP pipelines. This project benchmarks **7 models** from classical ML to fine-tuned transformers on the Amazon Review Polarity dataset.

> *"Which model best captures sentiment in Amazon product reviews — Logistic Regression, LSTM, or fine-tuned BERT?"*

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| **Source** | Amazon Review Polarity (Kaggle) |
| **Task** | Binary Sentiment Classification (Positive / Negative) |
| **Platform** | Kaggle (NVIDIA Tesla T4 GPU) |
| **Labels** | 0 = Negative, 1 = Positive |

---

## ⚙️ Pipeline Overview

```
Amazon Reviews (Kaggle)
        │
        ▼
Text Preprocessing (NLTK)
  ├── HTML tag removal, URL cleaning
  ├── Tokenization, stopword removal
  └── Lemmatization
        │
        ▼
Feature Extraction
  ├── TF-IDF → Classical ML models
  └── Word2Vec (Gensim) → Deep Learning models
        │
        ▼
Model Benchmarking
  ├── Logistic Regression   → 85.4%
  ├── Random Forest         → benchmarked
  ├── Decision Tree         → benchmarked
  ├── Naïve Bayes           → benchmarked
  ├── RNN                   → benchmarked
  ├── LSTM                  → 85.6%
  └── Bi-LSTM               → 86.3% (best DL before BERT)
        │
        ▼
Fine-tuned BERT (HuggingFace + PyTorch)
  └── 90% Accuracy ✅ — BERT outperforms all models
```

---

## 📊 Results

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 85.4% | Best classical ML model |
| LSTM | 85.6% | Slight improvement over LR |
| Bi-LSTM | 86.3% | Best before transformer |
| **BERT (fine-tuned)** | **90.0% ✅** | **Best overall — outperforms all** |

**BERT Classification Report:**
```
              precision    recall  f1-score
Negative         0.90      0.89      0.90
Positive         0.90      0.90      0.90
Accuracy                             0.90
```

---

## 🔑 Key Implementation Details

**Text Preprocessing:**
- NLTK tokenization, stopword removal, lemmatization
- HTML tag stripping, URL and special character removal

**Classical ML:** TF-IDF vectorization → Logistic Regression, Random Forest, Decision Tree, Naïve Bayes

**Deep Learning:** Word2Vec embeddings (Gensim) → RNN / LSTM / Bi-LSTM (TensorFlow/Keras)

**BERT Fine-tuning:**
```python
# HuggingFace bert-base-uncased
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_scheduler("linear", optimizer, ...)
# 10 epochs | GPU: NVIDIA Tesla T4
```

---

## 📁 Repository Structure

```
📦 Customer-Review-Sentiment-Analysis/
├── Customer_Review_Sentiment_Analysis_BERT.ipynb   # Full pipeline notebook
├── README.md
└── LICENSE
```

> **Note:** Dataset available on Kaggle — [Amazon Review Polarity](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)

---

## 🚀 Getting Started

```bash
pip install transformers torch scikit-learn nltk gensim tensorflow pandas numpy matplotlib seaborn
```

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🧠 Concepts Covered

`NLP` · `Text Classification` · `TF-IDF` · `Word2Vec` · `RNN` · `LSTM` · `Bi-LSTM` · `Transfer Learning` · `BERT Fine-tuning` · `HuggingFace Transformers` · `Sentiment Analysis`

---

## 👤 Author

**Dhruv Kumar Sahu**
M.Tech, Industrial & Management Engineering — IIT Kanpur
GATE 2024 AIR 33 | [LinkedIn](https://www.linkedin.com/in/dhruv-kumar-sahu-157ab9193/) · [GitHub](https://github.com/dhruvkumar24-ai)
