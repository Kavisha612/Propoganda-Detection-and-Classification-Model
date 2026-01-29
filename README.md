# Propaganda-Detect

A lightweight, end-to-end NLP pipeline for detecting propagandistic text spans and classifying 14 distinct propaganda techniques in news articles.

## Features

- **Span Detection**  
  Trains a binary TF-IDF → Logistic Regression model to flag propaganda spans (F1 ↑ from 0.40 → 0.75 on 200 articles / 2 000+ annotations).  
- **Technique Classification**  
  Builds per-technique classifiers using SMOTE-ENN oversampling, custom text features, and ensemble voting for robust multi-label performance across 14 tactics.  
  Macro F1 : 0.38, Micro : 0.51
- **Batch Processing & Analysis**  
  CLI scripts for corpus ingestion, label generation, model training, evaluation, and exporting a serialized `task2_ensemble_technique_model.pkl`.  

## Tech Stack

- **Core:** Python, pandas, NumPy  
- **ML & NLP:** scikit-learn, TfidfVectorizer, LogisticRegression  
- **Imbalance Handling:** imbalanced-learn (SMOTE-ENN)  
- **Persistence:** pickle

## Installation
### 1. Clone and enter directory
git clone https://github.com/Kavisha612/propoganda_detection_model.git
cd propoganda_detection_model
### 2. Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate
### Install Dependencies
pip install -r requirements.txt

## Usage
### 1. Generate labels files
If using original database
```
python compile.py
python test_dataset.py
```
Otherwise the generated label files are already in the repo

### 2. Run the pipeline
python final_implementation.py


## Repo Structure

```text
.
├── compile.py               # Generate span & technique label files
├── initial_implementatons   # Initial implemenations using BERT-BIO labelling
    ├── first_full_train.py
    ├── full_train_nonmodular.py
    ├── initial_10_files_train.py
    ├── second_train.py           
├── test_dataset.py          # Validate / augment annotation files
├── final_implementation.py  # Main pipeline: feature extraction, training, evaluation
├── requirements.txt         # Python dependencies
└── data/
    ├── pilot_train_articles/      # Raw .txt articles
    ├── pilot_train-task1-SI.labels
    └── pilot_train-task2-TC.labels



