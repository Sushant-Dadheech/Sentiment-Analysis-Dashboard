# 🧠 Advanced Sentiment Analysis Dashboard

A dual-engine AI-powered sentiment analysis dashboard built with **Streamlit**, combining **VADER** and **TextBlob** for superior text classification accuracy across 40,000+ texts.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **Dual-Engine NLP** — Combines VADER (rule-based) + TextBlob (pattern-based) for higher accuracy
- **Confidence Scoring** — Measures agreement between both engines on every prediction
- **5-Level Classification** — Positive / Slightly Positive / Neutral / Slightly Negative / Negative
- **Real-Time Analysis** — Type any text and get instant sentiment with polarity bar & confidence meter
- **Rich Visualizations** — Pie charts, histograms, scatter plots, word clouds & subjectivity analysis
- **Modern UI** — Clean light theme with gradient hero banners, glassmorphism cards & hover effects

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **📊 Overview Dashboard** | Dataset metrics, sentiment distribution pie chart, polarity histogram, insight alerts, and data table |
| **✍️ Live Analysis** | Type any text for instant dual-engine sentiment analysis with confidence scoring |
| **🔬 Deep Insights** | VADER vs TextBlob scatter plot, word clouds, subjectivity & confidence distributions, 5-level breakdown |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Sushant-dadheech/Sentiment-Analysis-Dashboard.git
cd Sentiment-Analysis-Dashboard
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (required for VADER)

```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

4. **Run the dashboard**

```bash
streamlit run Src/app.py
```

5. **Open in browser**

The app will automatically open at [http://localhost:8501](http://localhost:8501)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web application framework |
| **NLTK (VADER)** | Rule-based sentiment analysis |
| **TextBlob** | Pattern-based sentiment analysis |
| **Pandas** | Data manipulation and processing |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization (charts, plots) |
| **WordCloud** | Word cloud generation |

---

## 📁 Project Structure

```
Sentiment-Analysis-Dashboard/
├── Data/
│   └── New Data.csv          # Dataset (40,000+ texts from Kaggle)
├── Src/
│   ├── app.py                # Main Streamlit dashboard application
│   └── Sentiment_dashboard.py # Original terminal-based script
├── Assets/                   # Project assets
├── Output/                   # Output files
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🔬 How It Works

The dashboard uses a **weighted dual-engine approach** for sentiment analysis:

1. **VADER** (65% weight) — Optimized for social media and informal text
2. **TextBlob** (35% weight) — Pattern-based analysis using linguistic rules

```
Combined Score = (0.65 × VADER Compound) + (0.35 × TextBlob Polarity)
Confidence = (1 - |VADER - TextBlob| / 2) × 100
```

### Sentiment Thresholds

| Combined Score | Label |
|----------------|-------|
| ≥ 0.30 | Positive |
| 0.05 to 0.30 | Slightly Positive |
| -0.05 to 0.05 | Neutral |
| -0.30 to -0.05 | Slightly Negative |
| ≤ -0.30 | Negative |

---

## 👤 Author

**Sushant Dadheech**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
