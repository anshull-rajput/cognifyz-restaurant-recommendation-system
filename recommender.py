from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


class RestaurantRecommender:
    """Content-based recommendations from cuisine and price preferences."""

    def __init__(self, cuisine_weight: float = 0.8, price_weight: float = 0.2):
        if cuisine_weight < 0 or price_weight < 0 or cuisine_weight + price_weight == 0:
            raise ValueError("Weights must be non-negative and at least one must be positive.")
        total = cuisine_weight + price_weight
        self.cuisine_weight = cuisine_weight / total
        self.price_weight = price_weight / total
        self.restaurants: pd.DataFrame | None = None
        self.cuisine_encoder: MultiLabelBinarizer | None = None
        self.feature_matrix: np.ndarray | None = None

    @staticmethod
    def _clean_name(value: object) -> str:
        return re.sub(r"\s+", " ", str(value).replace("\ufeff", "")).strip()

    @staticmethod
    def _cuisines(value: object) -> list[str]:
        if pd.isna(value) or not str(value).strip():
            return ["Unknown"]
        return sorted({part.strip().casefold() for part in str(value).split(",") if part.strip()}) or ["Unknown"]

    def fit(self, data: pd.DataFrame) -> "RestaurantRecommender":
        """Clean the input and encode cuisines and price range for similarity search."""
        frame = data.copy()
        frame.columns = [self._clean_name(column) for column in frame.columns]
        required = {"Restaurant Name", "Cuisines", "Price range"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

        frame["Restaurant Name"] = frame["Restaurant Name"].fillna("Unknown restaurant").astype(str).str.strip()
        frame["Cuisines"] = frame["Cuisines"].fillna("Unknown").astype(str)
        frame["_cuisine_tokens"] = frame["Cuisines"].map(self._cuisines)
        frame["Price range"] = pd.to_numeric(frame["Price range"], errors="coerce")
        frame = frame.dropna(subset=["Price range"]).reset_index(drop=True)
        if frame.empty:
            raise ValueError("No restaurants have a usable price range.")

        self.cuisine_encoder = MultiLabelBinarizer()
        cuisine_features = self.cuisine_encoder.fit_transform(frame["_cuisine_tokens"])
        price_values = frame["Price range"].to_numpy(dtype=float)
        price_min, price_max = float(price_values.min()), float(price_values.max())
        price_scaled = np.zeros_like(price_values) if price_max == price_min else (price_values - price_min) / (price_max - price_min)
        # Square-root feature weights make cosine similarity reflect the configured weights.
        self.feature_matrix = np.column_stack([
            cuisine_features * np.sqrt(self.cuisine_weight),
            price_scaled * np.sqrt(self.price_weight),
        ])
        frame["_price_scaled"] = price_scaled
        self.restaurants = frame
        return self

    def recommend(
        self,
        cuisines: Iterable[str] | None = None,
        price_range: int | float | None = None,
        city: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Rank restaurants by cosine similarity to the requested preferences."""
        if self.restaurants is None or self.feature_matrix is None or self.cuisine_encoder is None:
            raise RuntimeError("Call fit(data) before recommend(...).")
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        requested = sorted({str(c).strip().casefold() for c in (cuisines or []) if str(c).strip()})
        if not requested and price_range is None:
            raise ValueError("Provide at least a cuisine or a price range preference.")

        query_cuisines = np.zeros(len(self.cuisine_encoder.classes_), dtype=float)
        known = set(self.cuisine_encoder.classes_)
        for cuisine in requested:
            if cuisine in known:
                query_cuisines[list(self.cuisine_encoder.classes_).index(cuisine)] = 1.0
        price_scaled = 0.0
        if price_range is not None:
            low = float(self.restaurants["Price range"].min())
            high = float(self.restaurants["Price range"].max())
            price_scaled = 0.0 if high == low else (float(price_range) - low) / (high - low)
        query = np.concatenate([
            query_cuisines * np.sqrt(self.cuisine_weight),
            [price_scaled * np.sqrt(self.price_weight)],
        ]).reshape(1, -1)
        scores = cosine_similarity(query, self.feature_matrix).ravel()

        result = self.restaurants.copy()
        result["Similarity"] = scores
        if city:
            if "City" not in result.columns:
                raise ValueError("City filtering was requested, but the dataset has no City column.")
            result = result[result["City"].fillna("").astype(str).str.casefold() == city.strip().casefold()]
        result = result.sort_values(
            ["Similarity", "Aggregate rating", "Votes"] if {"Aggregate rating", "Votes"}.issubset(result.columns) else ["Similarity"],
            ascending=[False, False, False] if {"Aggregate rating", "Votes"}.issubset(result.columns) else [False],
            kind="stable",
        )
        visible = [column for column in ["Restaurant Name", "Cuisines", "Price range", "City", "Aggregate rating", "Similarity"] if column in result.columns]
        return result[visible].head(top_n).reset_index(drop=True)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read a CSV while handling the UTF-8 BOM present in some exports."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}. See README for setup.")
    return pd.read_csv(file_path, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend restaurants from cuisine and price preferences.")
    parser.add_argument("--dataset", default="data/Dataset .csv", help="Path to the restaurant CSV")
    parser.add_argument("--cuisines", nargs="*", default=["Italian"], help="Preferred cuisines")
    parser.add_argument("--price-range", type=float, default=2, help="Preferred dataset price range (typically 1-4)")
    parser.add_argument("--city", help="Optional city filter")
    parser.add_argument("--top", type=int, default=10, help="Number of recommendations")
    args = parser.parse_args()

    recommender = RestaurantRecommender().fit(load_dataset(args.dataset))
    results = recommender.recommend(args.cuisines, args.price_range, args.city, args.top)
    if results.empty:
        print("No restaurants matched that city. Try another city or remove --city.")
    else:
        print(results.to_string(index=False, formatters={"Similarity": "{:.3f}".format}))


if __name__ == "__main__":
    main()
