# 📰 Hyperpartisan News Detection

> 🗂️ All modeling, evaluation, and analysis steps are provided as easy-to-run Jupyter notebooks.


## 📌 Project Overview

This project explores ***hyperpartisan news detection*** — identifying news articles exhibiting extreme political bias — using machine learning. It was conducted as part of a university course and contributes 40% to the final grade.

We aim to:
- Compare baseline (Naive Bayes, Logistic Regression, Random Forest, XGBoost) and advanced (XLM-RoBERTa) models.
- Evaluate models using clean (by-article) and noisy (by-publisher) labels from the SemEval-2019 Task 4 datasets.
- Analyze model confidence, sentiment, and publisher-level inconsistencies.
- Critically reflect on dataset limitations, distant supervision, and label quality.
- Visualize findings through ROC curves, confusion matrices, and feature importance plots.

---
## 🔁 Reproducibility

To explore or rerun our work:

1. Make sure you have **Python** installed (≥3.8 recommended)
2. Make sure you have **Jupyter Notebook** or **JupyterLab** installed
3. Clone the GitHub repository:

```bash
git clone <repo_url>
```

---
## 📂 Datasets

### 🔗 Main Source

- **PAN @ SemEval 2019 Task 4: Hyperpartisan News Detection**  
  [Dataset on Zenodo](https://doi.org/10.5281/zenodo.1489920)  
  [Task Website](https://webis.de/data/pan-semeval-hyperpartisan-news-detection-19.html)  
  [Official Paper (Kiesel et al., 2019)](https://downloads.webis.de/publications/papers/kiesel_2019c.pdf)

---

### 📁 Clean Labeled Data (by-article)

Used for primary model training, testing, and evaluation:

- `articles-training-byarticle-20181122.xml`
- `ground-truth-training-byarticle-20181122.xml`
- `articles-test-byarticle-20181207.xml`
- `ground-truth-test-byarticle-20181207.xml`

Training set: 645 articles  
- 238 (37%) hyperpartisan  
- 407 (63%) not hyperpartisan

Test set: 628 articles (balanced)  
- 314 (50%) hyperpartisan  
- 314 (50%) not hyperpartisan

>✅ Balanced test set
>✅ Used for main model development
> ⚠️ Train/test sets are publisher-disjoint to prevent publisher bias leakage

---

### 📁 Noisy Labeled Data (by-publisher)

Used for pretraining and experimental setups:

- `articles-training-bypublisher-20181122.xml` (600k articles)
- `ground-truth-training-bypublisher-20181122.xml`
- *(Optional)* Validation: 150k articles with labels

- Publisher-level weak supervision (inferred bias).
- Large-scale data (~750,000 articles).
- 
> ⚠️ Labels inferred from domain-level bias, so may be noisy  
> ⚠️ Models risk learning publisher style instead of true bias.



---

## 🎯 Task Description

- **Task Type:** Supervised binary classification  
- **Goal:** Predict whether an article is *hyperpartisan* (`1`) or *not* (`0`)  
- **Input:** News article (title + body)  
- **Output:** Binary prediction  
- **Challenges:**
  - Label noise (especially by-publisher)
  - Imbalanced classes in training data
  - Long-text modeling (truncation/sliding window issues)
  - Preventing publisher-based leakage

---

## 🧠 Approach

### ✅ Data Preparation
- XML parsing and cleaning (title + body merged)
- Standardized preprocessing for all models
- Stylistic feature extraction (uppercase ratio, exclamation count, etc.)
- Balanced splits, cross-validation folds

### 🧪 Baseline Models
- **Naive Bayes**, **Logistic Regression**, **Random Forest**, **XGBoost**
- Preprocessing variants: TF-IDF vs. Count, ROS sampling, lemmatization, stopword removal
- Feature engineering experiments

Final baseline: **Random Forest** with TF-IDF + stylistic features.

### 🤖 Advanced Model
- Fine-tuned **XLM-RoBERTa-base** on the by-article dataset
- 5-fold weighted ensemble for prediction
- Early stopping, learning rate tuning, input truncation at 512 tokens

---

## 📊 Evaluation and Analysis

Metrics used:
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Score
- AUC (ROC Area Under Curve)

Analysis included:
- Confidence-based binning
- Sentiment vs. confidence correlation
- Error analysis (correct vs misclassified articles)
- Feature importance for Random Forest
- Aggregation threshold experiments (Appendix A.4)

---

## 🔄 Phase 2: Publisher-Level Prediction

After selecting models based on article-level evaluation:

- **Random Forest** and **XLM-RoBERTa** applied to by-publisher test set
- Aggregated article predictions into publisher labels (threshold = 50%)
- Tested aggregation thresholds from 5% to 95% (reported in Appendix A.4)
- Investigated inconsistencies between model predictions and ground truth labels
- Conducted sentiment-confidence and error analyses

---

## 🧪 Key Results

| Model                | F1 (Article) | AUC (Article) | F1 (Publisher) | AUC (Publisher) |
|----------------------|--------------|---------------|----------------|-----------------|
| Random Forest        | 0.784        | 0.845         | 0.39           | 0.54            |
| XLM-RoBERTa Ensemble | 0.856        | 0.921         | 0.54           | 0.63            |

- XLM-RoBERTa outperformed classical baselines
- Publisher-level prediction was more difficult due to noisy labels
- Models revealed label inconsistencies in PAN publisher annotations

---

## 👥 Team

| Member | Contributions |
|--------|---------------|
| Darya  | Data processing, baselines, Phase 2 inference, report writing |
| Mikah  | Baselines, evaluation, visualizations, presentation |
| Kjell  | Advanced model (XLM-R), Phase 2 inference, report writing |

> ✏️ Collaboration managed via GitHub pull requests.

---

## 🗓️ Timeline

| Week          | Focus |
|---------------|-------|
| Mar 21–23     | Registration, dataset exploration |
| Mar 24–Apr 10 | Preprocessing, baselines, advanced model|
| Apr 10–20     | Advanced model tuning, Phase 2 |
| Apr 21–29     | Phase 2, Final evaluation, report writing, presentation |
| Apr 30        | 🎯 Submission |
---

## 📝 Report & Presentation

- **Report**: 10-page ACM 2-column LaTeX format (Overleaf)
- **Includes**: Methodology, evaluation, insights, limitations, GitHub link  
- **Presentation**: 5–10 mins live in-class presentation

