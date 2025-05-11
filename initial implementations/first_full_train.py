# Fixed propaganda detection code
import glob
import pandas as pd
import os
from transformers import BertTokenizerFast

print("Starting propaganda detection model training...")

# 1) Read span annotations with robust header handling
print("Reading span annotations...")
try:
    # Try reading the raw file first to inspect
    with open("pilot_train-task1-SI.labels", "r") as f:
        first_line = f.readline().strip()
    
    # Check if first line has header-like content
    if "doc_id" in first_line or "start" in first_line:
        print("Detected header row in spans file")
        # Read with explicit header
        spans_df = pd.read_csv("pilot_train-task1-SI.labels", sep="\t", skiprows=1, 
                              names=["doc_id", "start", "end"])
    else:
        # No header
        spans_df = pd.read_csv("pilot_train-task1-SI.labels", sep="\t", header=None,
                              names=["doc_id", "start", "end"])
    
    # Convert start/end to integers
    spans_df["start"] = pd.to_numeric(spans_df["start"])
    spans_df["end"] = pd.to_numeric(spans_df["end"])
    
except Exception as e:
    print(f"Error in initial parsing: {e}")
    # Fallback: manual parsing
    data = []
    with open("pilot_train-task1-SI.labels", "r") as f:
        lines = f.readlines()
        # Skip first line if it looks like a header
        start_idx = 1 if ("doc_id" in lines[0] or "start" in lines[0]) else 0
        
        for i in range(start_idx, len(lines)):
            parts = lines[i].strip().split('\t')
            if len(parts) == 3:
                try:
                    doc_id = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    data.append([doc_id, start, end])
                except ValueError:
                    print(f"Warning: Skipping invalid line: {lines[i].strip()}")
    
    spans_df = pd.DataFrame(data, columns=["doc_id", "start", "end"])

# Print some info about the spans
print(f"Loaded {len(spans_df)} span annotations")
print("First 5 spans:")
print(spans_df.head())

# 2) Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased",
    model_max_length=512,
    truncation=True
)

# 3) Load articles into a dict
print("\nLoading articles...")
articles = {}
for fn in glob.glob("pilot_train_articles/*.txt"):
    doc_id = os.path.basename(fn).replace(".txt", "")
    try:
        with open(fn, encoding="utf-8", errors="replace") as f:
            articles[doc_id] = f.read()
    except Exception as e:
        print(f"Error reading {fn}: {e}")

print(f"Loaded {len(articles)} articles")

# 4) Create mapping from span IDs to article filenames
print("\nCreating ID mappings...")
article_id_to_filename = {}
for doc_id in articles.keys():
    # Extract the numeric part from filenames like "article696694316"
    if doc_id.startswith("article"):
        numeric_id = doc_id.replace("article", "")
        article_id_to_filename[numeric_id] = doc_id

# Display some mapping info
print(f"Number of articles loaded: {len(articles)}")
print(f"Number of unique IDs in spans file: {len(spans_df['doc_id'].unique())}")
print(f"First few span IDs: {spans_df['doc_id'].unique()[:5]}")
print(f"First few article files: {list(articles.keys())[:5]}")

# 5) Sliding-window chunking & labeling function
def chunk_and_make_bio(text, article_spans, max_len=512, stride=128):
    """
    Create BIO labels for text using sliding windows
    article_spans: list of (start, end) tuples for propaganda spans
    """
    # tokenize with overflow to create windows
    tok = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=max_len,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
    )
    
    # build char mask for the text
    char_mask = [0] * len(text)
    for s, e in article_spans:
        for i in range(s, min(e, len(char_mask))):  # Safety check for bounds
            char_mask[i] = 1
    
    # Count propaganda characters
    prop_chars = sum(char_mask)
    
    windows = []
    for idx, input_ids in enumerate(tok.input_ids):
        offsets = tok.offset_mapping[idx]
        
        # determine char span for this window (skip special tokens)
        real_offsets = [o for o in offsets if o[1] > o[0]]
        if not real_offsets:  # Skip empty windows
            continue
            
        win_start, win_end = real_offsets[0][0], real_offsets[-1][1]
        
        # label each token using char_mask
        bio_labels = []
        prev_was_prop = False
        
        for start, end in offsets:
            if start == end:  # Special token
                bio_labels.append(-100)
                continue
                
            # Check for propaganda in this token
            is_propaganda = False
            is_beginning = False
            
            if start < len(char_mask) and end <= len(char_mask):
                segment = char_mask[start:end]
                is_propaganda = any(segment)
                
                # Check if this is the beginning of a span
                if is_propaganda:
                    # It's the beginning if the first char is marked OR
                    # if previous token wasn't propaganda
                    is_beginning = (segment[0] == 1) or not prev_was_prop
            
            if is_propaganda:
                if is_beginning:
                    bio_labels.append(1)  # B-PROP
                else:
                    bio_labels.append(2)  # I-PROP
                prev_was_prop = True
            else:
                bio_labels.append(0)  # O
                prev_was_prop = False
        
        # Count labels
        b_count = bio_labels.count(1)
        i_count = bio_labels.count(2)
        
        windows.append({
            "input_ids": input_ids,
            "labels": bio_labels,
            "char_span": (win_start, win_end),
            "b_count": b_count,
            "i_count": i_count
        })
    
    return windows

# 6) Process each document
print("\nProcessing documents...")
records = []
processed_docs = 0
found_propaganda = 0

# Process each unique document ID in the spans file
for span_doc_id in spans_df['doc_id'].unique():
    # Get corresponding article filename
    article_id = article_id_to_filename.get(str(span_doc_id))
    
    if article_id and article_id in articles:
        text = articles[article_id]
        processed_docs += 1
        
        # print(f"\nProcessing document {processed_docs}: {article_id} (span ID: {span_doc_id})")
        # print(f"Text length: {len(text)} chars")
        
        # Get spans for this document
        doc_spans = spans_df[spans_df.doc_id == span_doc_id][["start", "end"]].values.tolist()
        # print(f"Found {len(doc_spans)} propaganda spans")
        
        if doc_spans:
            found_propaganda += 1
            windows = chunk_and_make_bio(text, doc_spans)
            records.extend(windows)
            # print(f"Created {len(windows)} windows")
    else:
        print(f"\nWARNING: Could not find article file for span ID {span_doc_id}")

# 7) Results summary
print("\n--- Results Summary ---")
print(f"Total unique documents in spans file: {len(spans_df['doc_id'].unique())}")
print(f"Documents successfully processed: {processed_docs}")
print(f"Documents with propaganda spans: {found_propaganda}")
print(f"Total windows created: {len(records)}")

# Count windows with propaganda
windows_with_prop = sum(1 for rec in records if rec["b_count"] > 0 or rec["i_count"] > 0)
if records:
    print(f"Windows containing propaganda: {windows_with_prop} ({windows_with_prop/len(records)*100:.1f}%)")
else:
    print("No windows were created!")

# Print details of first 10 windows with propaganda
if windows_with_prop > 0:
    print("\nFirst 10 windows with propaganda:")
    count = 0
    for i, rec in enumerate(records):
        if rec["b_count"] > 0 or rec["i_count"] > 0:
            print(f"Window {i:3d} chars={rec['char_span']}  B-PROP={rec['b_count']:2d}  I-PROP={rec['i_count']:2d}")
            count += 1
            if count >= 10:
                break
else:
    print("\nDEBUG: No windows contain propaganda labels! Potential issues:")
    print("1. Span file format mismatch")
    print("2. Article IDs in span file don't match article filenames")
    print("3. Character offsets in span file may be incorrect")

# 8) Prepare data for model training
if records:
    print("\nPreparing data for model training...")
    train_input_ids = [rec["input_ids"] for rec in records]
    train_labels = [rec["labels"] for rec in records]
    
    print(f"Dataset size: {len(train_input_ids)} examples")
    print("Example input shape:", len(train_input_ids[0]))
    print("Example labels:", train_labels[0][:10], "...")
else:
    print("\nERROR: Cannot prepare training data because no records were created.")
    
    
    
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

# 1) Build a list of window‐texts and window‐labels
window_texts = []
window_labels = []
for rec in records:
    # decode the token ids back into text
    txt = tokenizer.decode(rec["input_ids"], skip_special_tokens=True)
    window_texts.append(txt)
    # label 1 if any B‑PROP or I‑PROP in that window, else 0
    window_labels.append(1 if (rec["b_count"] + rec["i_count"])>0 else 0)

print(f"Total windows: {len(window_texts)}, positives: {sum(window_labels)}, negatives: {len(window_labels)-sum(window_labels)}")

# 2) Split into train / test
X_train, X_test, y_train, y_test = train_test_split(
    window_texts, window_labels,
    test_size=0.2,
    stratify=window_labels,
    random_state=42
)

# 3) Build a TF‑IDF + LogisticRegression pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=2)),
    ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42))
])

# 4) Train
pipe.fit(X_train, y_train)

# 5) Predict on test
y_pred = pipe.predict(X_test)

# 6) Compute F1
f1 = f1_score(y_test, y_pred)
print(f"\nWindow‐level F1: {f1:.4f}\n")
print("Detailed classification report:")
print(classification_report(y_test, y_pred, target_names=["no‐prop","prop"]))

# print("\nProcessing documents...")
# records = []
# processed_docs = 0
# found_propaganda = 0

# # Process each unique document ID in the spans file
# for span_doc_id in spans_df['doc_id'].unique():
#     # Get corresponding article filename
#     article_id = article_id_to_filename.get(str(span_doc_id))
    
#     if article_id and article_id in articles:
#         text = articles[article_id]
#         processed_docs += 1
        
#         # Get spans for this document
#         doc_spans = spans_df[spans_df.doc_id == span_doc_id][["start", "end"]].values.tolist()
        
#         if doc_spans:
#             found_propaganda += 1
#             # generate windows with BIO labels
#             windows = chunk_and_make_bio(text, doc_spans)
#             # attach document ID to each window record for grouped splitting
#             for w in windows:
#                 w["doc_id"] = span_doc_id
#             records.extend(windows)
#     else:
#         print(f"\nWARNING: Could not find article file for span ID {span_doc_id}")

# # 7) Results summary
# print("\n--- Results Summary ---")
# print(f"Total unique documents in spans file: {len(spans_df['doc_id'].unique())}")
# print(f"Documents successfully processed: {processed_docs}")
# print(f"Documents with propaganda spans: {found_propaganda}")
# print(f"Total windows created: {len(records)}")

# # Count windows with propaganda
# windows_with_prop = sum(1 for rec in records if rec["b_count"] > 0 or rec["i_count"] > 0)
# if records:
#     print(f"Windows containing propaganda: {windows_with_prop} ({windows_with_prop/len(records)*100:.1f}%)")
# else:
#     print("No windows were created!")

# # 8) Prepare data for model training
# if records:
#     print("\nPreparing data for model training...")
#     train_input_ids = [rec["input_ids"] for rec in records]
#     train_labels = [rec["labels"] for rec in records]
    
#     print(f"Dataset size: {len(train_input_ids)} examples")
#     print("Example input shape:", len(train_input_ids[0]))
#     print("Example labels:", train_labels[0][:10], "...")
# else:
#     print("\nERROR: Cannot prepare training data because no records were created.")
    
# # 9) Build window-level texts, labels, and doc_ids for grouped split
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import f1_score, classification_report

# window_texts = []
# window_labels = []
# window_doc_ids = []
# for rec in records:
#     # decode the token ids back into text
#     txt = tokenizer.decode(rec["input_ids"], skip_special_tokens=True)
#     window_texts.append(txt)
#     # label 1 if any B-PROP or I-PROP in that window, else 0
#     window_labels.append(1 if (rec["b_count"] + rec["i_count"]) > 0 else 0)
#     # use the attached doc_id for grouping
#     window_doc_ids.append(rec["doc_id"])

# print(f"Total windows: {len(window_texts)}, positives: {sum(window_labels)}, negatives: {len(window_labels)-sum(window_labels)}")

# # 10) Group-wise train/test split
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, test_idx = next(gss.split(window_texts, window_labels, groups=window_doc_ids))

# X_train = [window_texts[i] for i in train_idx]
# y_train = [window_labels[i] for i in train_idx]
# X_test  = [window_texts[i] for i in test_idx]
# y_test  = [window_labels[i] for i in test_idx]

# print("Train size:", len(X_train), "Test size:", len(X_test))
# print(f"Train positives: {sum(y_train)}, Test positives: {sum(y_test)}")

# # 11) TF-IDF + LogisticRegression pipeline
# pipe = Pipeline([
#     ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=2)),
#     ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42))
# ])

# # 12) Train and evaluate
# pipe.fit(X_train, y_train)

# y_pred = pipe.predict(X_test)

# f1 = f1_score(y_test, y_pred)
# print(f"\nWindow-level F1: {f1:.4f}\n")
# print("Detailed classification report:")
# print(classification_report(y_test, y_pred, target_names=["no-prop","prop"]))