import glob
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000, min_df=2)),
    ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42))
])
# Read the spans file - convert doc_id to string immediately to avoid float conversion issues
spans_df = pd.read_csv("pilot_train-task1-SI.labels", sep="\t", header=None, 
                      names=["doc_id", "start", "end"])
spans_df["doc_id"] = spans_df["doc_id"].astype(str)  # Convert to string to avoid float issues
spans_df["start"] = pd.to_numeric(spans_df["start"])
spans_df["end"] = pd.to_numeric(spans_df["end"])

# Read the techniques file - convert doc_id to string immediately
tech_df = pd.read_csv("pilot_train-task2-TC.labels", sep="\t",
                      header=None,
                      names=["doc_id", "start", "end", "technique"])
tech_df["doc_id"] = tech_df["doc_id"].astype(str)  # Convert to string to avoid float issues
tech_df["start"] = pd.to_numeric(tech_df["start"])
tech_df["end"] = pd.to_numeric(tech_df["end"])

# Load articles into a dict: {doc_id: text} - with better error handling
articles = {}
for fn in glob.glob("pilot_train_articles/*.txt"):
    doc_id = os.path.basename(fn).replace(".txt", "")
    try:
        with open(fn, encoding="utf-8", errors="replace") as f:
            articles[doc_id] = f.read()
    except Exception as e:
        print(f"Error reading {fn}: {e}")

# Add debug print to check article loading
print(f"Loaded {len(articles)} articles")
if not articles:
    print("WARNING: No articles loaded! Check the path 'pilot_train_articles/*.txt'")

# Debug span loading
print(f"Loaded {len(spans_df)} span annotations")
# Print a few span examples
if not spans_df.empty:
    print("First 5 spans:")
    print(spans_df.head())
else:
    print("WARNING: No spans loaded! Check the file 'pilot_train-task1-SI.labels'")

# # Print article IDs for debugging
# print("First few article IDs:", list(articles.keys())[:5])
# print("First few span doc_ids:", spans_df['doc_id'].unique()[:5])

# Create a mapping from document IDs in spans to article filenames
article_id_to_filename = {}
for doc_id in articles.keys():
    # Extract the numeric part from filenames like "article696694316"
    if doc_id.startswith("article"):
        numeric_id = doc_id.replace("article", "")
        article_id_to_filename[numeric_id] = doc_id

print("\n--- Article ID Mapping ---")
print(f"Number of articles loaded: {len(articles)}")
print(f"Number of unique IDs in spans file: {len(spans_df['doc_id'].unique())}")
print(f"First few span IDs: {spans_df['doc_id'].unique()[:5]}")
print(f"First few article files: {list(articles.keys())[:5]}")

# TASK 1: Binary Sequence Labeling
# We'll create an alternative approach using character-level features

# Function to extract labeled spans as positive examples
def extract_labeled_spans(doc_id, text):
    doc_spans = spans_df[spans_df.doc_id == doc_id][["start", "end"]].values.tolist()
    spans = []
    
    # Extract labeled spans (positive examples)
    for start, end in doc_spans:
        if start < len(text) and end <= len(text):
            span_text = text[start:end]
            spans.append({
                "doc_id": doc_id,
                "start": start,
                "end": end,
                "text": span_text,
                "label": 1  # This is propaganda
            })
    
    # Extract some random negative examples (non-propaganda)
    # We'll sample a similar number of negative examples to balance the dataset
    num_negative = len(doc_spans)
    avg_span_len = sum(end-start for start, end in doc_spans) / len(doc_spans) if doc_spans else 20
    
    # Try up to 3x the number of spans to find non-overlapping negative examples
    attempts = 0
    negative_spans = []
    while len(negative_spans) < num_negative and attempts < num_negative * 3:
        attempts += 1
        # Choose a random position in the text
        neg_start = np.random.randint(0, max(1, len(text) - int(avg_span_len)))
        neg_end = min(len(text), neg_start + int(avg_span_len))
        
        # Check if this random span overlaps with any propaganda span
        is_overlapping = any(
            (start <= neg_start < end) or 
            (start < neg_end <= end) or
            (neg_start <= start < neg_end) or
            (neg_start < end <= neg_end)
            for start, end in doc_spans
        )
        
        if not is_overlapping:
            negative_spans.append({
                "doc_id": doc_id,
                "start": neg_start,
                "end": neg_end,
                "text": text[neg_start:neg_end],
                "label": 0  # This is NOT propaganda
            })
    
    spans.extend(negative_spans)
    return spans

# Process articles and extract spans for Task 1
all_spans = []
for span_doc_id in spans_df['doc_id'].unique():
    article_id = article_id_to_filename.get(span_doc_id)
    
    if article_id and article_id in articles:
        text = articles[article_id]
        # print(f"\nProcessing document: {article_id} (span ID: {span_doc_id}), length: {len(text)} chars")
        
        # Extract spans from this article
        article_spans = extract_labeled_spans(span_doc_id, text)
        all_spans.extend(article_spans)
        
        print(f"Extracted {len(article_spans)} spans (propaganda + non-propaganda)")
    else:
        print(f"\nWARNING: Could not find article file for span ID {span_doc_id}")

# Convert to DataFrame for easier processing
spans_data = pd.DataFrame(all_spans)
print(f"\nTotal spans extracted: {len(spans_data)}")
print(f"Class distribution: {spans_data['label'].value_counts().to_dict()}")

# TASK 1: Train a binary classifier to detect propaganda spans
windows = [
    (i, spans_data.iloc[i]['label'], spans_data.iloc[i]['text'],
     spans_data.iloc[i]['doc_id'],
     spans_data.iloc[i]['start'], spans_data.iloc[i]['end'])
    for i in range(len(spans_data))
]

# shuffle once
import random
random.seed(42)
random.shuffle(windows)

# desired test size
n_test = int(0.2 * len(windows))

train, test = [], []
# keep track of train ranges per doc
train_ranges = {}

for idx, label, txt, doc, start, end in windows:
    # try to put into test if we still need more
    if len(test) < n_test:
        # check overlap against all train ranges for this doc
        overlap = False
        for (ts, te) in train_ranges.get(doc, []):
            if not (end <= ts or start >= te):
                overlap = True
                break
        if not overlap:
            test.append((idx, label, txt))
            continue
    # otherwise goes to train
    train.append((idx, label, txt))
    train_ranges.setdefault(doc, []).append((start, end))

# unpack back into X/y
train_idx = [t[0] for t in train]
test_idx  = [t[0] for t in test]

X_train = spans_data['text'].iloc[train_idx]
y_train = spans_data['label'].iloc[train_idx]
X_test  = spans_data['text'].iloc[test_idx]
y_test  = spans_data['label'].iloc[test_idx]

# now fit & evaluate exactly as before:
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("\nTask 1 Results (no-overlap split):")
print(f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# TASK 2: Multi-label Technique Classification

# Create pivot table while keeping doc_id as string
multi = (tech_df
         .assign(flag=1)
         .pivot_table(
            index=["doc_id", "start", "end"],
            columns="technique",
            values="flag",
            fill_value=0
         )
         .reset_index())

# Now merge with spans_df (both already have doc_id as strings)
train_spans = pd.merge(spans_df, multi,
                       on=["doc_id", "start", "end"],
                       how="inner")

# Debug merged dataframe
print("\nMerged spans with techniques:")
print(f"Shape: {train_spans.shape}")
print(train_spans.head())

# For each span, pull out its substring
def get_text(row):
    # Construct the article key properly from the doc_id
    article_key = f"article{row.doc_id}"
    
    # First check if the key exists directly
    if article_key in articles:
        txt = articles[article_key]
        return txt[row.start:row.end]
    
    # If not, try to find it in the mapping
    if row.doc_id in article_id_to_filename:
        article_key = article_id_to_filename[row.doc_id]
        txt = articles[article_key]
        return txt[row.start:row.end]
    
    # If still not found, print debug info and return empty string
    print(f"WARNING: Could not find article for doc_id {row.doc_id}")
    return ""

# Debug before applying get_text
print("\nBefore extracting span text:")
print(f"Article keys sample: {list(articles.keys())[:5]}")
print(f"Doc IDs sample: {train_spans['doc_id'].head().tolist()}")

# Apply the fixed get_text function
train_spans["span_text"] = train_spans.apply(get_text, axis=1)

# Debug after applying get_text
print("\nAfter extracting span text:")
print(f"Number of spans with text: {train_spans['span_text'].str.len().gt(0).sum()}")
print(f"Sample span texts:\n{train_spans['span_text'].head()}")

# Select only the text + technique columns
label_cols = [c for c in train_spans.columns if c not in ["doc_id", "start", "end", "span_text"]]
print(f"\nTechnique classes: {label_cols}")

# Split data for Task 2
X = train_spans["span_text"]
Y = train_spans[label_cols].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\n--- Training Multi-label Technique Classifier (Task 2) ---")

# Create a pipeline with TF-IDF and Logistic Regression for multi-label
pipeline_multi = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2
    )),
    ('clf', MultiOutputClassifier(LogisticRegression(
        C=1.0,
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )))
])

# Train the model
pipeline_multi.fit(X_train, y_train)

# Evaluate
y_pred_multi = pipeline_multi.predict(X_test)
micro_f1 = f1_score(y_test, y_pred_multi, average='micro')
macro_f1 = f1_score(y_test, y_pred_multi, average='macro')

print(f"\nTask 2 - Multi-label Classification Results:")
print(f"Micro F1 Score: {micro_f1:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

# Print per-class metrics
y_test_df = pd.DataFrame(y_test, columns=label_cols)
y_pred_df = pd.DataFrame(y_pred_multi, columns=label_cols)

print("\nPer-technique performance:")
for i, technique in enumerate(label_cols):
    if y_test_df[technique].sum() > 0:  # Only show metrics for techniques that appear in test set
        technique_f1 = f1_score(y_test_df[technique], y_pred_df[technique])
        support = y_test_df[technique].sum()
        print(f"{technique}: F1={technique_f1:.4f}, Support={support}")

# Save both models
import pickle

# Save Task 1 model
with open('task1_propaganda_detection_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Save Task 2 model
with open('task2_technique_classification_model.pkl', 'wb') as f:
    pickle.dump(pipeline_multi, f)

print("\nModels saved successfully!")