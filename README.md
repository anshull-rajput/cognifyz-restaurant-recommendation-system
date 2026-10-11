# Cognifyz ML Internship — Task 2: Restaurant Recommendation

A content-based restaurant recommender that ranks restaurants by how closely their cuisines and price range match a user's preferences. It uses the Cognifyz restaurant dataset and scikit-learn cosine similarity.

## Requirements

- Python 3.10 or later
- The Cognifyz restaurant CSV supplied with the internship materials (`Dataset .csv`). The dataset is not included in this repository because the supplied file is 2.2 MB and its redistribution terms were not provided.

## Setup

```bash
git clone https://github.com/anshull-rajput/cognifyz-restaurant-recommendation-system.git
cd cognifyz-restaurant-recommendation-system
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Place the supplied CSV at `data/Dataset .csv` (create the `data` folder), or provide its location with `--dataset`.

## Run recommendations

```bash
python recommender.py --cuisines Italian --price-range 2 --top 10
```

Multiple cuisines can be supplied together. City is optional:

```bash
python recommender.py --dataset "data/Dataset .csv" --cuisines Japanese Seafood --price-range 3 --city "Makati City" --top 5
```

Dataset `Price range` values are ordinal levels (typically 1–4), not currency amounts. Cuisine matching is case-insensitive. Missing cuisine values are represented as `Unknown`; rows without a usable price range are skipped. City filtering is exact apart from capitalization. Similarity ties use rating and vote count, when available.

## Method and evaluation

The implementation splits comma-separated cuisines into categories, encodes them with `MultiLabelBinarizer`, scales the ordinal price range, then ranks restaurant feature vectors against the preference vector with cosine similarity. Cuisine and price contribute 80% and 20% of the feature weight by default. These are transparent, tunable design choices rather than weights learned from user feedback.

Run the reproducible sample checks with:

```bash
python evaluate.py --dataset "data/Dataset .csv"
```

On the supplied 9,551-row CSV, Japanese/Seafood at price level 3 produced 5/5 results sharing at least one requested cuisine and a mean absolute price-level difference of 0.20. Italian at price level 2 produced 5/5 Italian results at price level 2. These checks measure preference alignment, not real-world user satisfaction; the dataset has no user interaction or relevance labels for offline ranking metrics.

## Files

- `recommender.py` — data preparation, content-based ranking, and command-line interface
- `requirements.txt` — Python dependencies
