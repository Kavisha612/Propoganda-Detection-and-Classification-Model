# check_pilot_consistency.py

from pathlib import Path

BASE = Path(__file__).parent

ART_DIR   = BASE / "pilot_train-articles"
SI_DIR    = BASE / "pilot_train-labels-task1-span-identification"
TC_DIR    = BASE / "pilot_train-labels-task2-technique-classification"

# Gather all article IDs in pilot_train-articles
article_ids = sorted(p.stem for p in ART_DIR.glob("article*.txt"))

missing_si = []
missing_tc = []
count = 0
for art_id in article_ids:
    # expected per-article filenames in each label folder
    count = count + 1
    si_file = SI_DIR / f"{art_id}.task1-SI.labels"
    tc_file = TC_DIR / f"{art_id}.task2-TC.labels"
    if not si_file.exists():
        missing_si.append(art_id)
    if not tc_file.exists():
        missing_tc.append(art_id)

if not missing_si and not missing_tc:
    print(f"✅ All {count} pilot articles have matching SI and TC files.")
else:
    if missing_si:
        print("⚠️ Missing SI spans for:", missing_si)
    if missing_tc:
        print("⚠️ Missing TC labels for:", missing_tc)
