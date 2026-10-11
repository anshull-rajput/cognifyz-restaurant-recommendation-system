"""Run the sample preference checks described in the Task 2 brief."""
import argparse
from pathlib import Path

from recommender import RestaurantRecommender, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/Dataset .csv", help="Path to the supplied restaurant CSV")
    parser.add_argument("--top", type=int, default=5, help="Recommendations to inspect per preference")
    args = parser.parse_args()

    model = RestaurantRecommender().fit(load_dataset(args.dataset))
    examples = [
        ({"cuisines": ["Italian"], "price_range": 2}, "Italian, price level 2"),
        ({"cuisines": ["Japanese", "Seafood"], "price_range": 3}, "Japanese/Seafood, price level 3"),
    ]
    print(f"Restaurants loaded: {len(model.restaurants):,}\n")
    for preferences, label in examples:
        results = model.recommend(**preferences, top_n=args.top)
        requested = {value.casefold() for value in preferences["cuisines"]}
        overlap = results["Cuisines"].fillna("").map(
            lambda value: bool(requested.intersection(part.strip().casefold() for part in value.split(",")))
        )
        price_distance = (results["Price range"] - preferences["price_range"]).abs().mean()
        print(f"{label}: {overlap.sum()}/{len(results)} recommendations share a requested cuisine; "
              f"mean absolute price-level difference = {price_distance:.2f}")
        print(results.to_string(index=False, formatters={"Similarity": "{:.3f}".format}))
        print()


if __name__ == "__main__":
    main()
