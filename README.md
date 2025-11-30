# Sentiment Analyzer

A simple sentiment‑analysis tool built in Python. Given text input (e.g. social‑media comments, reviews, etc.), this project cleans and vectorizes the data, trains a model, and allows making sentiment predictions — ideal for learning NLP and sentiment classification workflows.

---

## 🚀 Features

- Data preprocessing for text data
- Model training pipeline (`train_model.py`)
- Prediction interface (`predict.py`)

---

## 🛠️ Tech Stack & Dependencies

- Python 3.x
- Common NLP / ML libraries (scikit‑learn, pandas, numpy)

---

## 📦 Setup & Usage

```bash
git clone https://github.com/neerajcli/sentiment_analyzer.git
cd sentiment_analyzer
pip install -r requirements.txt
```

### Preprocess & vectorize:
```bash
python data_preprocessing.py
python vectorize_data.py
```

### Train model:
```bash
python train_model.py
```

### Predict sentiment:
```bash
python main.py
```

---

## 🧪 How It Works

- `data_preprocessing.py` — text cleaning
- `vectorize_data.py` — converts text to numeric features
- `train_model.py` — trains classifier (SVM / Logistic Regression)
- `predict.py` — inference for new text
