# 🔍 VeriFact — AI-Powered Fake News Detector for Students


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_DEPLOYED_LINK_HERE)

---

## 🧠 What is VeriFact?

VeriFact is an AI-powered fake news detection system that helps students identify misinformation in online news articles. It combines classical machine learning, NLP analysis, and LLM reasoning to deliver explainable verdicts in student-friendly language.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 ML Classifier | TF-IDF + best classifier (LogReg / PAC / Random Forest) |
| 🌍 Multilingual | Handles English and non-English via Gemini API |
| 💬 Explainability | Gemini explains WHY an article is likely fake/real |
| 📊 Tone Analysis | Sensationalism score (exclamation marks, caps, keywords) |
| 🎭 Sentiment | VADER sentiment scoring |
| 🧩 NER | Named entities via spaCy (people, orgs, places, dates) |
| 📈 Confidence | Transparent confidence % with hybrid ML+LLM routing |

---

## 🏗️ Architecture

```
Input Article
      │
      ▼
Language Detection (langdetect)
      ├── Non-English ──► Gemini Full Analysis
      └── English
            │
            ▼
      TF-IDF + ML Classifier
            ├── High Confidence (≥75%) ──► ML Verdict + Gemini Explanation
            └── Low Confidence (<75%)  ──► Gemini Second Opinion
                        │
                        ▼
              NLP Enrichment (Sentiment + Tone + NER)
                        │
                        ▼
                  Streamlit UI
```

---

## 📁 Project Structure

```
verifact/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_NLP_Layer.ipynb
│   └── 04_Gemini_Integration.ipynb
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
|
│─ app.py
│── requirements.txt
├── utils/
│   ├── nlp_utils.py
│   └── gemini_utils.py
├── data/          (dataset not committed — too large)
├── assets/        (EDA visualizations)
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/pratham-bits/verifact-fake-news-detector
cd verifact
pip install -r app/requirements.txt
python -m spacy download en_core_web_sm
export GEMINI_API_KEY=your_key_here
streamlit run app/app.py
```

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.9651 | 0.9946 |
| Passive Aggressive | 0.9788 | 0.9974 |
| Random Forest | 0.9591 | 0.9932 |



---

## 🛠️ Tech Stack

Pandas · Scikit-learn · NLTK · VADER · spaCy · langdetect · Gemini API · Streamlit

---

## ⚠️ Limitations

- ML model trained on US English news (2016–2018)
- Non-English articles rely on Gemini reasoning
- Always verify with: The Hindu, NDTV, PTI, Reuters India

---

## 📚 Dataset

**WELFake** — 72,134 articles | https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

---

*EDUNET Foundation × IBM SkillsBuild AI Internship | May 2026 Batch*