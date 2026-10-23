"""Compare all models on a chronological per-user split and write results/.

    python scripts/download_data.py
    python scripts/evaluate.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys import (  # noqa: E402
    ContentBasedRecommender,
    HybridRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    load_movielens,
    temporal_split,
)
from recsys.evaluate import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    dataset = load_movielens(args.data)
    train, test = temporal_split(dataset.ratings, test_fraction=args.test_fraction)
    print(
        f"{dataset.ratings['user_id'].nunique()} users, {len(dataset.items)} items, "
        f"{len(dataset.ratings)} ratings -> train {len(train)}, test {len(test)}"
    )

    models = [
        PopularityRecommender(),
        ContentBasedRecommender(),
        ItemKNNRecommender(),
        HybridRecommender(cf_weight=0.7),
    ]

    results = {}
    for model in models:
        started = time.time()
        model.fit(train, dataset.items)
        metrics = evaluate(model, test, k=args.k)
        metrics["seconds"] = round(time.time() - started, 1)
        results[model.name] = metrics
        print(f"{model.name:<11} " + "  ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    k = args.k
    lines = [
        f"| Model | Precision@{k} | Recall@{k} | Hit rate@{k} | NDCG@{k} | Coverage |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in results.items():
        lines.append(
            f"| `{name}` | {m[f'precision@{k}']:.3f} | {m[f'recall@{k}']:.3f} | "
            f"{m[f'hit_rate@{k}']:.3f} | {m[f'ndcg@{k}']:.3f} | {m['coverage']:.1%} |"
        )
    (args.out / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
