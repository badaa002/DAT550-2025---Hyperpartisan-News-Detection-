# 📰 Hyperpartisan News Detection

## 📌 Project Overview

This project explores the task of detecting **hyperpartisan news articles** — content that exhibits extreme political bias — using machine learning. The project is part of a university course and contributes 25% to the final grade.

We aim to:
- Compare baseline and advanced models (including deep learning)
- Evaluate model performance using both clean and noisy labels
- Analyze and reflect on the quality of labeling and feature importance

---

## 📂 Datasets

### 🔗 Main Source

- **PAN @ SemEval 2019 Task 4: Hyperpartisan News Detection**  
  [Dataset on Zenodo](https://doi.org/10.5281/zenodo.1489920)  
  [Task Website](https://webis.de/data/pan-semeval-hyperpartisan-news-detection-19.html)  
  [Official Paper (Kiesel et al., 2019)](https://downloads.webis.de/publications/papers/kiesel_2019c.pdf)

### 📁 Clean Labeled Data (by-article)

Used for initial training, testing, and evaluation:

- `articles-training-byarticle-20181122.xml` — Article texts with `<title>` and `<p>` tags
- `ground-truth-training-byarticle-20181122.xml` — Manually labeled `hyperpartisan: true/false`

> ✅ Crowdsourced, article-level labels with high quality (645 samples)

---

### 📁 Noisy Labeled Data (by-publisher)

Used for large-scale experiments and weak-supervision training:

- `articles-training-bypublisher-20181122.xml` — 600,000 articles
- `ground-truth-training-bypublisher-20181122.xml` — Labels based on site-level political bias
- *(Optional)* `articles-validation-bypublisher-20181122.xml` — 150,000 validation articles
- *(Optional)* `ground-truth-validation-bypublisher-20181122.xml` — Labels for validation

> ⚠️ Labels in this set are based on domain-level bias (not on article content) and may be noisy.

---

## 🎯 Task Description

- **Task Type:** Supervised binary classification
- **Goal:** Predict whether a news article is *hyperpartisan* (`1`) or *not hyperpartisan* (`0`)
- **Input:** News article text (title + body)
- **Output:** Binary label
- **Challenges:**
  - Imbalanced classes in clean data
  - Noisy labeling in large dataset
  - Long-form text
  - Differentiating strong vs moderate bias

---

## 🧠 Approach (Planned)

### 1. **Data Preprocessing**
- Parse XML articles
- Clean and normalize text
- Match articles with correct labels
- Create structured dataset for modeling


  TODO.....

## 👥 Team

- **Member 1**: [Name] – Data processing, baseline models, visualization
- **Member 2**: [Name] – Deep learning model, evaluation, report writing
- **Member 3**: [Name] – Deep learning model, evaluation, report writing

> Workload will be shared equally across all project phases.

---

## 🧪 Experiments

We plan to:
- Tune hyperparameters (learning rate, dropout, regularization)
- Compare traditional ML vs deep learning
- Test multiple vectorization techniques
- Document model strengths/weaknesses

---

## 🗓️ Timeline

| Week | Tasks |
|------|-------|
| Mar 21–23 | Finalize topic, register team, set up repo |
| Mar 23–31 | Data exploration, define task, build baseline |
| Apr 1–10  | Deep learning model, embeddings |
| Apr 11–20 | Evaluation, error analysis, write report |
| Apr 21–29 | Final tuning, presentation slides |
| Apr 30    | Final submission 🎉 |

---

## 📝 Report & Presentation

- **Report**: 4–10 pages in ACM two-column LaTeX format  
- **Includes**: Dataset, methodology, evaluation, limitations, GitHub link
- **Presentation**: 5–10 minutes (in person preferred)

---

## 🔁 Reproducibility

To reproduce our results:

1. Clone the repo  
2. Install dependencies with `requirements.txt`  
3. Follow the instructions in `notebooks/` or `src/`  
4. Run the main pipeline: `python main.py` *(WIP)*  
5. All models and outputs will be saved in `/models` and `/results` directories

---

## 📅 Key Deadlines

- **Project Registration:** March 26, 2025
- **Final Submission:** April 30, 2025

---

Stay tuned for updates! 🚀
