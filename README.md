# 📰 Hyperpartisan News Detection

## 📌 Project Overview

This project explores the task of detecting **hyperpartisan news articles** — content that exhibits extreme political bias — using machine learning. The project is part of a university course and contributes 25% to the final grade.

We aim to:
- Compare baseline (Naive Bayes, Logistic Regression, Random Forest, XGBoost) and advanced models (including XLM-RoBERTa)
- Evaluate model performance using both clean (by-article) and noisy (by-publisher) labels
- Analyze feature importance, model confidence, and sentiment in predictions
- Conduct critical reflections on dataset quality and limitations
- Visualize and document key findings, including error analysis

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

> ✅ High-quality, crowdsourced article-level annotations  
> ⚠️ Train/test sets are publisher-disjoint to prevent publisher bias leakage

---

### 📁 Noisy Labeled Data (by-publisher)

Used for pretraining and experimental setups:

- `articles-training-bypublisher-20181122.xml` (600k articles)
- `ground-truth-training-bypublisher-20181122.xml`
- *(Optional)* Validation: 150k articles with labels

> ⚠️ Labels inferred from domain-level bias, so may be noisy  
> ⚠️ Risk of models learning publisher style over true bias content

---

## 🎯 Task Description

- **Task Type:** Supervised binary classification  
- **Goal:** Predict whether an article is *hyperpartisan* (`1`) or *not* (`0`)  
- **Input:** News article (title + body)  
- **Output:** Binary prediction  
- **Challenges:**
  - Label quality differences across datasets
  - Data imbalance in smaller, clean set
  - Long-text modeling and subjective definitions of partisanship
  - Avoiding publisher-based leakage in training

---

## 🧠 Approach

### ✅ Preprocessing
- Cleaned and parsed both XML and TSV versions of datasets
- Built structured versions of the data with aligned labels
- Extracted stylistic features and sentiment features
- Created balanced train/test sets and cross-validation splits

### 🧪 Baseline Models
Evaluated multiple baseline classifiers:
- **Naive Bayes**
- **Logistic Regression**
- **Random Forest** *(final baseline)*
- **XGBoost**

Tested combinations of:
- Vectorizers: Count, TF-IDF
- Preprocessing: Lemmatization, stopword removal
- Sampling: Random oversampling (ROS)
- Features: Stylistic + sentiment + lexical

### 🤖 Advanced Model
- Fine-tuned **XLM-RoBERTa (base)** using Hugging Face Transformers
- Cross-validated on the by-article dataset
- Stored predicted labels and probabilities for analysis
- Compared against traditional models in terms of performance and confidence

---

## 📊 Evaluation & Analysis

- Metrics used: Accuracy, Balanced Accuracy, Precision, Recall, F1, AUC
- Visuals: ROC curves, confusion matrices, metric heatmaps, radar charts
- Confidence-based error analysis (e.g., low-confidence misclassifications)
- Sentiment and stylistic feature impact studies
- Discussion on labeling quality, dataset bias, and model robustness

---

## 🔄 Phase 2: Relabeling the Noisy Dataset (By-Publisher)

After establishing strong models using the clean **by-article** dataset, we extended our project to include the larger, noisily-labeled **by-publisher** dataset.

### ⚙️ Procedure

We used both our best baseline and transformer models to relabel the noisy dataset:

- **Random Forest** (best classical model)
- **XLM-RoBERTa** (fine-tuned transformer)

Steps:
- Generated predictions (`.npy`) and probabilities for each article in the by-publisher set
- Compared predictions to:
  - The original PAN labels
  - Each other (model agreement/disagreement)
- Focused on **confidence-based filtering** to identify possibly mislabeled examples

### 🧐 Observations

- **XLM-RoBERTa** predicted a higher number of hyperpartisan articles than Random Forest
- Many **low-confidence PAN labels** were contradicted by both models
- High agreement between the two models on *non-hyperpartisan* articles
- Divergent predictions often showed signs of **ambiguous or borderline language**
- Clear cases of publisher bias in PAN labels were revealed when articles lacked strong partisan cues

### 🔬 Analytical Goals

- Assess **label quality** in the PAN by-publisher dataset using model predictions
- Explore **model disagreement** to highlight weak spots in distant supervision
- Identify **potentially cleaner subsets** of the noisy data for future training
- Reflect on whether models learned **true hyperpartisan cues** or just publisher style

> ✅ Both classical and deep models help expose limitations of the noisy dataset  
> 🔁 Could be extended into semi-supervised or active learning strategies


## 🧪 Results Summary (Example)

| Model                | Setup                     | F1   | AUC  |
|---------------------|---------------------------|------|------|
| Naive Bayes          | TF-IDF + Style + ROS + Lemmatization | 0.739 | 0.815 |
| Random Forest (Final Baseline) | TF-IDF + Style + ROS + Lemmatization | 0.753 | 0.826 |
| XLM-RoBERTa (Fine-tuned) | Raw text input + Cross-validation | **0.80+** | **~0.85** |

> 📌 Best classical setup used TF-IDF, stylistic features, and oversampling  
> 📌 Transformer-based model outperformed others but at higher compute cost

---

## 👥 Team

- **Member 1**: [Name] - preprocessing, baselines + evoluation/tuning, baseline model label-prediction, report writing
- **Member 2**: [Name] - baselines + evaluation, report writing, presentation
- **Member 3**: [Name] - advanced model + evoluation/tuning, advanced model label-prediction, report writing

> Contributions were distributed equally across project phases.

---

## 🗓️ Timeline

| Week         | Tasks |
|--------------|-------|
| Mar 21–23    | Team registration, setup, dataset download |
| Mar 23–Apr 10 | Data cleaning, preprocessing, baseline models, DL setup |
| Apr 1–10     | Report Part 1, tuning & planning Part 2 |
| Apr 11–20    | Phase 2, Final model evaluation, analysis, report writing |
| Apr 21–29    | Report polishing, presentation prep |
| Apr 30       | 🎉 Submission deadline 🎉 |

---

## 📝 Report & Presentation

- **Report**: Written in ACM 2-column LaTeX format (Overleaf)  
- **Includes**: Methodology, evaluation, insights, limitations, GitHub link  
- **Presentation**: 5–10 mins live in-class presentation

---

## 🔁 Reproducibility

To reproduce our results:

```bash
git clone <repo_url>
cd <repo>
pip install -r requirements.txt
python main.py  # (optional, WIP main pipeline)
