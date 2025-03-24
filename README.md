# 📰 Hyperpartisan News Detection

## 📌 Project Overview

This project explores the task of detecting **hyperpartisan news articles** — content that exhibits extreme political bias — using machine learning techniques. It is part of a course assignment and contributes 25% to the final grade.

We aim to compare baseline and advanced models (including deep learning), evaluate performance, and identify patterns in partisan media.

---

## 📂 Dataset

### 🔗 Source
- **BuzzFeed News & Webis Hyperpartisan Dataset**  
  [GitHub Repository](https://github.com/BuzzFeedNews/2017-08-partisan-sites-and-facebook-pages)  
  [Kiesel et al. (2019) - Paper PDF](https://downloads.webis.de/publications/papers/kiesel_2019c.pdf)

### 📁 Raw Data
The dataset includes:
- `data/all-partisan-sites.csv` — 667 websites identified as partisan (left or right), with info about Facebook pages and origins (e.g. Macedonia)
- `data/pages-info.csv` — Metadata on 452 Facebook pages (title, description, fan count, etc.)
- `data/domaintools-whois-results.csv` — WHOIS registration data for each domain

> ⚠️ Full Facebook post data (2015–2017) is too large for the repository but downloadable via the original repo.

---

### 📁 External Evaluation Dataset

The project also uses a clean, manually labeled dataset for training and evaluation:

- `articles-training-byarticle-20181122.xml` — Full-text news articles with `<title>`, `<p>` (paragraphs), and metadata
- `ground-truth-training-byarticle-20181122.xml` — Article-level binary labels (`hyperpartisan: true/false`), matched by article ID

> ✅ This dataset comes from the [SemEval-2019 Task 4](https://doi.org/10.5281/zenodo.1489920) and includes **1,273 articles** manually annotated for hyperpartisanship.  
> 📌 Used for model training, validation, and benchmarking on high-quality labels.

## 🎯 Task Description (editing ....) 

- **Task Type:** Supervised classification
- **Goal:** ....
- **Input:** 
- **Output:** Binary label — `Hyperpartisan` or `Not Hyperpartisan`
- **Challenges:**
  - Long-form text processing
  - Imbalanced class distributions
  - Bias in sources and annotation
  - To be filled....

---

## 🧠 Approach (Planned)

1. **Data Cleaning & Exploration**
   - Handle missing values
   - Normalize text
   - Visualize with TSNE/PCA

2. **Feature Engineering**
   - CountVectorizer / TF-IDF
   - Word embeddings (Word2Vec, BERT)

3. **Modeling**
   - **Baseline**: Logistic Regression, Naive Bayes, Decision Tree
   - **Advanced**: BERT (HuggingFace Transformers)
   - **Optional**: Ensemble techniques

4. **Evaluation**
   - Accuracy, Precision, Recall, F1, ROC Curve
   - Significance testing

---

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
| Mar 21–24 | Finalize topic, register team, set up repo |
| Mar 25–31 | Data exploration, define task, build baseline |
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
