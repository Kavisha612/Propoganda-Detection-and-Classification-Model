import glob
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
import re
import pickle
import random
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Ensure NLTK resources are available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Custom column selector transformer to replace lambda functions
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.column_name]

# Custom context combiner
class ContextCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, before_col='context_before', after_col='context_after'):
        self.before_col = before_col
        self.after_col = after_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return [f"{row[self.before_col]} {row[self.after_col]}" for _, row in X.iterrows()]

# Custom transformer for extracting text features
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = np.zeros((len(X), 6))
        
        for i, text in enumerate(X):
            # Text length features
            features[i, 0] = len(text)
            features[i, 1] = len(text.split())
            
            # Sentiment features
            sentiment = self.sia.polarity_scores(text)
            features[i, 2] = sentiment['pos']
            features[i, 3] = sentiment['neg']
            features[i, 4] = sentiment['neu']
            features[i, 5] = sentiment['compound']
            
        return features

# Read the spans file
spans_df = pd.read_csv("pilot_train-task1-SI.labels", sep="\t", header=None, 
                      names=["doc_id", "start", "end"])
spans_df["doc_id"] = spans_df["doc_id"].astype(str)
spans_df["start"] = pd.to_numeric(spans_df["start"])
spans_df["end"] = pd.to_numeric(spans_df["end"])

# Read the techniques file
tech_df = pd.read_csv("pilot_train-task2-TC.labels", sep="\t",
                      header=None,
                      names=["doc_id", "start", "end", "technique"])
tech_df["doc_id"] = tech_df["doc_id"].astype(str)
tech_df["start"] = pd.to_numeric(tech_df["start"])
tech_df["end"] = pd.to_numeric(tech_df["end"])

# Print technique distribution for better understanding
print("\nTechnique distribution:")
technique_counts = tech_df['technique'].value_counts()
print(technique_counts)

# Find articles with more propaganda techniques for better training
doc_counts = tech_df["doc_id"].value_counts()
top_docs = doc_counts.head(100).index.tolist()
print(f"\nSelected top 100 docs with most propaganda techniques: {top_docs}")

# Load only those 100 articles
articles = {}
loaded_count = 0

for fn in glob.glob("pilot_train_articles/*.txt"):
    doc_id = os.path.basename(fn).replace(".txt", "")
    numeric_id = doc_id.replace("article", "")
    
    # Only load articles that correspond to our top 100 docs
    if numeric_id in top_docs:
        try:
            with open(fn, encoding="utf-8", errors="replace") as f:
                articles[doc_id] = f.read()
                loaded_count += 1
                print(f"Loaded article {doc_id} ({loaded_count}/100)")
        except Exception as e:
            print(f"Error reading {fn}: {e}")
    
    # Stop once we've loaded 10 articles
    if loaded_count >= 100:
        break

print(f"\nLoaded {len(articles)} articles for mini pilot")

# Create a mapping from document IDs to article filenames
article_id_to_filename = {}
for doc_id in articles.keys():
    if doc_id.startswith("article"):
        numeric_id = doc_id.replace("article", "")
        article_id_to_filename[numeric_id] = doc_id

# Filter spans_df and tech_df to only include our 10 articles
spans_df = spans_df[spans_df['doc_id'].isin(top_docs)]
tech_df = tech_df[tech_df['doc_id'].isin(top_docs)]

print(f"\nAfter filtering: {len(spans_df)} spans and {len(tech_df)} technique annotations")

# TASK 2: Create a more robust Multi-label Technique Classification model
# Create pivot table to transform techniques into columns
multi = (tech_df
         .assign(flag=1)
         .pivot_table(
            index=["doc_id", "start", "end"],
            columns="technique",
            values="flag",
            fill_value=0
         )
         .reset_index())

# Merge with spans_df to get all propaganda spans
train_spans = pd.merge(spans_df, multi,
                      on=["doc_id", "start", "end"],
                      how="inner")

print(f"\nMerged spans with techniques: {train_spans.shape[0]} rows")

# Function to extract text from spans
def get_text(row):
    article_key = f"article{row.doc_id}"
    
    if article_key in articles:
        txt = articles[article_key]
        if row.start < len(txt) and row.end <= len(txt):
            return txt[row.start:row.end]
    
    return ""

# Extract span text
train_spans["span_text"] = train_spans.apply(get_text, axis=1)

# Remove empty spans
train_spans = train_spans[train_spans["span_text"].str.len() > 0]

print(f"Spans with text: {len(train_spans)}")

# Additional feature extraction: context windows
def get_context(row, window_size=50):
    article_key = f"article{row.doc_id}"
    
    if article_key in articles:
        txt = articles[article_key]
        start = max(0, row.start - window_size)
        end = min(len(txt), row.end + window_size)
        
        # Get text before and after the span
        before = txt[start:row.start] if start < row.start else ""
        after = txt[row.end:end] if row.end < end else ""
        
        return before, after
    
    return "", ""

# Add context features
contexts = train_spans.apply(lambda row: get_context(row), axis=1)
train_spans["context_before"] = contexts.apply(lambda x: x[0])
train_spans["context_after"] = contexts.apply(lambda x: x[1])

# Select technique columns
label_cols = [c for c in train_spans.columns if c not in ["doc_id", "start", "end", "span_text", "context_before", "context_after"]]
print(f"\nTechnique classes: {len(label_cols)}")

# Extract training data
X = train_spans[["span_text", "context_before", "context_after"]]
Y = train_spans[label_cols].values

# Custom feature extraction class to combine context and span text
class ContextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X should be a dataframe with span_text, context_before, context_after
        features = np.zeros((len(X), 12))
        
        for i, row in enumerate(X.itertuples()):
            span_text = row.span_text
            context_before = row.context_before
            context_after = row.context_after
            
            # Span features
            span_sentiment = self.sia.polarity_scores(span_text)
            features[i, 0] = len(span_text)
            features[i, 1] = len(span_text.split())
            features[i, 2] = span_sentiment['compound']
            
            # Context before features
            before_sentiment = self.sia.polarity_scores(context_before)
            features[i, 3] = len(context_before)
            features[i, 4] = len(context_before.split())
            features[i, 5] = before_sentiment['compound']
            
            # Context after features
            after_sentiment = self.sia.polarity_scores(context_after)
            features[i, 6] = len(context_after)
            features[i, 7] = len(context_after.split())
            features[i, 8] = after_sentiment['compound']
            
            # Sentiment contrast features
            features[i, 9] = span_sentiment['compound'] - before_sentiment['compound']
            features[i, 10] = span_sentiment['compound'] - after_sentiment['compound']
            features[i, 11] = np.abs(features[i, 9]) + np.abs(features[i, 10])
            
        return features

# Custom preprocessor for TfidfVectorizer
def text_preprocessor(text):
    # Convert to lowercase
    text = text.lower()
    # Replace URLs with a placeholder
    text = re.sub(r'https?://\S+', 'URL', text)
    # Replace emails with a placeholder
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    # Replace numbers with a placeholder
    text = re.sub(r'\b\d+\b', 'NUMBER', text)
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

# Extract text features (indicators of propaganda techniques)
def extract_text_features(text):
    features = []
    # Check for emotional language
    emotional_words = ['urgent', 'crisis', 'danger', 'threat', 'evil', 'terrible', 
                      'amazing', 'incredible', 'extraordinary', 'outrageous']
    for word in emotional_words:
        features.append(1 if word in text.lower() else 0)
    
    # Check for exaggeration
    exaggeration_phrases = ['all', 'none', 'always', 'never', 'everyone', 'nobody', 
                           'completely', 'totally', 'absolute']
    for phrase in exaggeration_phrases:
        features.append(1 if phrase in text.lower() else 0)
        
    # Check for loaded language
    loaded_words = ['disaster', 'catastrophe', 'chaos', 'scandal', 'fraud', 'corrupt', 
                   'evil', 'disgrace', 'shame', 'horrific']
    for word in loaded_words:
        features.append(1 if word in text.lower() else 0)
    
    return features

# Custom feature extractor for propaganda techniques
class PropagandaFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Assuming X is a dataframe with necessary columns
        features = np.zeros((len(X), 30))  # Adjust size based on your features
        
        for i, row in enumerate(X.itertuples()):
            span_text = row.span_text
            context = row.context_before + " " + row.context_after
            
            # Get propaganda technique indicators
            span_features = extract_text_features(span_text)
            
            # Context features
            context_features = extract_text_features(context)
            
            # Combine features
            all_features = span_features + context_features
            
            # Add features like presence of quotes, question marks, etc.
            all_features.append(1 if '"' in span_text or "'" in span_text else 0)
            all_features.append(1 if '?' in span_text else 0)
            all_features.append(1 if '!' in span_text else 0)
            all_features.append(span_text.count(',') / max(1, len(span_text.split())))
            
            # Add features like capitalization ratio
            caps_ratio = sum(1 for c in span_text if c.isupper()) / max(1, len(span_text))
            all_features.append(caps_ratio)
            
            # Copy features to output array
            for j, feature in enumerate(all_features):
                if j < features.shape[1]:  # Ensure we don't exceed array bounds
                    features[i, j] = feature
                    
        return features

# Split data with stratification to ensure class balance
# Use a larger test set to better evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42, stratify=Y.sum(axis=1)
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create a feature union pipeline that combines multiple feature extractors
feature_pipeline = FeatureUnion([
    # TF-IDF features from the span text
    ('span_tfidf', Pipeline([
        ('selector', ColumnSelector('span_text')),
        ('tfidf', TfidfVectorizer(
            preprocessor=text_preprocessor,
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            max_features=3000,
            min_df=2,
            use_idf=True,
            sublinear_tf=True  # Apply sublinear tf scaling (log scaling)
        ))
    ])),
    
    # TF-IDF features from the context (before and after)
    ('context_tfidf', Pipeline([
        ('combiner', ContextCombiner()),
        ('tfidf', TfidfVectorizer(
            preprocessor=text_preprocessor,
            ngram_range=(1, 2),
            max_features=1000,
            min_df=2
        ))
    ])),
    
    # Count vector features (bag of words) for specific propaganda indicators
    ('bow', Pipeline([
        ('selector', ColumnSelector('span_text')),
        ('countvec', CountVectorizer(
            vocabulary=[
                'all', 'never', 'always', 'everyone', 'nobody', 'completely', 
                'best', 'worst', 'perfect', 'disaster', 'urgent', 'must', 
                'they', 'them', 'we', 'us', 'our', 'their'
            ],
            binary=True
        ))
    ])),
    
    # Custom feature extractors
    ('context_features', ContextFeatureExtractor()),
    
    ('propaganda_features', PropagandaFeatureExtractor())
])

# Build the final pipeline for Task 2 - Multi-label classification
pipeline_multi = Pipeline([
    ('features', feature_pipeline),
    ('clf', MultiOutputClassifier(LogisticRegression(
        C=1.0,
        solver='liblinear',  # Fast solver for small datasets
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )))
])

# Train the model
print("\n--- Training Enhanced Multi-label Classifier (Task 2) ---")
pipeline_multi.fit(X_train, y_train)

# Evaluate
y_pred_multi = pipeline_multi.predict(X_test)
micro_f1 = f1_score(y_test, y_pred_multi, average='micro')
macro_f1 = f1_score(y_test, y_pred_multi, average='macro')

print(f"\nTask 2 - Multi-label Classification Results:")
print(f"Micro F1 Score: {micro_f1:.4f} (original was 0.5127)")
print(f"Macro F1 Score: {macro_f1:.4f} (original was 0.3766)")

# Print per-class metrics
y_test_df = pd.DataFrame(y_test, columns=label_cols)
y_pred_df = pd.DataFrame(y_pred_multi, columns=label_cols)

print("\nPer-technique performance:")
for i, technique in enumerate(label_cols):
    if y_test_df[technique].sum() > 0:  # Only show metrics for techniques that appear in test set
        technique_f1 = f1_score(y_test_df[technique], y_pred_df[technique])
        support = y_test_df[technique].sum()
        print(f"{technique}: F1={technique_f1:.4f}, Support={support}")

# Try another classifier to compare results
print("\n--- Training with RandomForest for comparison ---")
rf_pipeline = Pipeline([
    ('features', feature_pipeline),
    ('clf', MultiOutputClassifier(RandomForestClassifier(
        n_estimators=100,
        max_depth=100,
        class_weight='balanced',
        random_state=42
    )))
])

rf_pipeline.fit(X_train, y_train)
rf_y_pred = rf_pipeline.predict(X_test)
rf_micro_f1 = f1_score(y_test, rf_y_pred, average='micro')
rf_macro_f1 = f1_score(y_test, rf_y_pred, average='macro')

print(f"\nRandom Forest Results:")
print(f"Micro F1 Score: {rf_micro_f1:.4f}")
print(f"Macro F1 Score: {rf_macro_f1:.4f}")

# Take the best model
if rf_micro_f1 > micro_f1:
    print("\nRandom Forest performed better - using it as the final model")
    best_model = rf_pipeline
    best_micro_f1 = rf_micro_f1
    best_macro_f1 = rf_macro_f1
    best_pred = rf_y_pred
else:
    print("\nLogistic Regression performed better - using it as the final model")
    best_model = pipeline_multi
    best_micro_f1 = micro_f1
    best_macro_f1 = macro_f1
    best_pred = y_pred_multi

# Save the best model
with open('mini_pilot_propaganda_classifier.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\nFinal model saved as 'mini_pilot_propaganda_classifier.pkl'")
print(f"Final Micro F1: {best_micro_f1:.4f}, Macro F1: {best_macro_f1:.4f}")

# Create a sample prediction function to demonstrate model usage
def predict_propaganda_techniques(text, start=0, end=None):
    if end is None:
        end = len(text)
    
    test_data = pd.DataFrame([{
        'span_text': text[start:end],
        'context_before': text[max(0, start-50):start],
        'context_after': text[end:min(len(text), end+50)]
    }])
    
    # Make prediction
    prediction = best_model.predict(test_data)
    
    # Get techniques with positive predictions
    techniques = []
    for i, technique in enumerate(label_cols):
        if prediction[0][i] == 1:
            techniques.append(technique)
    
    return techniques

# Example usage
print("\nExample prediction:")
sample_text = "This is an absolute disaster! Everyone knows the government is lying to you."
sample_techniques = predict_propaganda_techniques(sample_text)
print(f"Text: {sample_text}")
print(f"Detected propaganda techniques: {sample_techniques}")