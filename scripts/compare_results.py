"""Build a portable Markdown comparison from completed training runs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    reference = rows[0]
    for row in rows:
        for key in ("identity_protocol", "data_meta"):
            if row.get(key) != reference.get(key):
                raise ValueError(f"Different {key}: compare protocols before combining runs")
        if row["verification"]["num_pairs"] != reference["verification"]["num_pairs"]:
            raise ValueError("Different test pair counts")
    lines = ["# Test verification results", "",
             "| Loss | ROC AUC | TPR @ 0.1% FPR | TPR @ 1% FPR | TPR @ 10% FPR |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        v = row["verification"]
        lines.append(f"| {row['loss']} | {v['roc_auc']:.8f} | " +
                     " | ".join(f"{v[f'tpr@fpr={fpr}']:.2%}" for fpr in (0.001, 0.01, 0.1)) + " |")
    lines += ["", f"Test pairs per run: {int(reference['verification']['num_pairs']):,}.",
              "Matching metadata does not prove identical manifest contents or pair lists.",
              "TPR at an FPR operating point is not identity classification accuracy.", ""]
    result = "\n".join(lines)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
