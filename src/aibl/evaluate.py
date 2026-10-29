"""Score count predictions for image and category pairs."""

import argparse
import json
from pathlib import Path

from aibl.counting import evaluate_counting


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON array: sample_id, category, predicted, ground_truth")
    parser.add_argument("--output", type=Path, help="Optional output JSON")
    args = parser.parse_args(argv)
    rows = json.loads(args.input.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must be a nonempty JSON array of image/category count pairs")
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or not all(key in row for key in
                ("sample_id", "category", "predicted", "ground_truth")):
            raise ValueError("Every row requires sample_id, category, predicted, ground_truth")
        identity = (str(row["sample_id"]), str(row["category"]))
        if identity in identities:
            raise ValueError(f"Duplicate image/category count pair: {identity}")
        identities.add(identity)
    metrics = evaluate_counting([row["predicted"] for row in rows],
                                [row["ground_truth"] for row in rows]).to_dict()
    metrics["n_category_pairs"] = len(rows)
    metrics["accuracy_unit"] = "fraction"
    result = json.dumps(metrics, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
