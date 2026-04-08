# ArXiv Bayesian Classifier

A Multinomial Naive Bayes classifier built from scratch using NumPy to categorize ArXiv research papers into subject categories based on their title and abstract text.

## Overview

The classifier learns probability distributions for each ArXiv category (e.g., `hep-th`, `cs.AI`, `math.CO`) using a bag-of-words model with Laplace smoothing. All computations are performed in log-space to prevent floating-point underflow. The model supports multi-label classification, where a single paper can belong to multiple categories.

## Dataset

- **Source**: [Cornell University ArXiv Metadata](https://www.kaggle.com/datasets/Cornell-University/arxiv) (via Kaggle)
- **Size**: 100,000 papers
- **Features**: Title + abstract text, vectorized into a 5,000-feature bag-of-words representation with unigrams and bigrams
- **Labels**: 126 ArXiv categories (after filtering out categories with fewer than 50 papers)

## Pipeline

1. **Data loading and cleaning** -- streaming JSON loader with quality filters and text normalization (removing version numbers, comment artifacts)
2. **Label filtering** -- categories appearing in fewer than 50 papers are removed to improve model stability
3. **Vectorization** -- `CountVectorizer` with English stop words removed, bigram support, and a 5,000 feature cap
4. **Training** -- log-prior and log-likelihood computation via vectorized NumPy operations (`y.T @ X`)
5. **Prediction** -- log-posterior scoring with per-sample 95th percentile thresholding for multi-label output

## Results

- **Hamming Loss**: 0.046
- **Micro avg recall**: 0.88
- **Micro avg precision**: 0.19

The model achieves high recall but low precision, tending to over-predict related categories. It fails gracefully -- incorrect predictions are typically in neighboring fields (e.g., predicting `hep-th` for a `hep-ph` paper) rather than completely unrelated domains.

## Files

- `arxiv_generative_classifier_skeleton.ipynb` -- main notebook with implementation
- `report.pdf` -- analysis report covering class imbalance, smoothing analysis, feature importance, and the naive independence assumption
- `bayes_classifier.pdf` -- assignment specification

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install kagglehub pandas numpy matplotlib scikit-learn tqdm
```

Run all cells in the notebook. The dataset will be downloaded automatically on first run.
