"""Compare accuracy metrics of several runs against a baseline (Windows or Jetson, Python 3.6+).

Reads the metrics.json files written by coco_metrics.py / eval_onnx_coco.py and
prints each run's box and mask mAP, precision, recall and F1, with the change
versus the baseline in brackets (percentage points; negative = accuracy lost).
Also saves comparison.csv in the accuracy folder.

    python scripts/compare_metrics.py
        baseline : data/results/accuracy/windows_onnx/metrics.json
        others   : every other data/results/accuracy/*/metrics.json

    python scripts/compare_metrics.py --baseline A/metrics.json B/metrics.json C/metrics.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # Edge-AI-Jetson/
ACCURACY_DIR = ROOT / "data" / "results" / "accuracy"

MAP_COLUMNS = [("box", "mAP50", "box mAP50"), ("box", "mAP50_95", "box mAP50-95"),
               ("mask", "mAP50", "mask mAP50"), ("mask", "mAP50_95", "mask mAP50-95")]
PRF_COLUMNS = [("box", "precision", "box P"), ("box", "recall", "box R"), ("box", "f1", "box F1"),
               ("mask", "precision", "mask P"), ("mask", "recall", "mask R"), ("mask", "f1", "mask F1")]
# Settings that must match for a fair comparison.
SETTINGS = ("num_images", "pr_conf", "pr_iou", "run.conf", "run.iou", "run.max_det", "run.input")


def load(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["name"] = path.parent.name
    return metrics


def setting(metrics: dict, key: str):
    value = metrics
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def table(title: str, columns, runs, baseline) -> str:
    width = 16
    lines = [title, "{:<16}".format("run") + "".join("{:>{}}".format(c[2], width) for c in columns)]
    for run in runs:
        cells = []
        for kind, key, _ in columns:
            value = 100.0 * run[kind][key]
            if run is baseline:
                cells.append("{:>{}.2f}".format(value, width))
            else:
                delta = value - 100.0 * baseline[kind][key]
                cells.append("{:>{}}".format("{:.2f} ({:+.2f})".format(value, delta), width))
        lines.append("{:<16}".format(run["name"][:15]) + "".join(cells))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, default=ACCURACY_DIR / "windows_onnx" / "metrics.json")
    parser.add_argument("others", type=Path, nargs="*", help="metrics.json files (default: all others)")
    parser.add_argument("--csv", type=Path, default=ACCURACY_DIR / "comparison.csv")
    args = parser.parse_args()

    if not args.baseline.exists():
        sys.exit("Baseline not found: {}".format(args.baseline))
    others = args.others or sorted(p for p in ACCURACY_DIR.glob("*/metrics.json")
                                   if p.resolve() != args.baseline.resolve())
    if not others:
        sys.exit("No other metrics.json found in {}".format(ACCURACY_DIR))

    baseline = load(args.baseline)
    runs = [baseline] + [load(p) for p in others]

    print(table("mAP (%), change vs {} in brackets".format(baseline["name"]), MAP_COLUMNS, runs, baseline))
    print()
    print(table("Precision / recall / F1 (%) at conf {}, IoU {}".format(baseline["pr_conf"], baseline["pr_iou"]),
                PRF_COLUMNS, runs, baseline))

    # Warn when runs were not produced under the same conditions.
    for key in SETTINGS:
        values = {run["name"]: setting(run, key) for run in runs}
        if len({json.dumps(v) for v in values.values() if v is not None}) > 1:
            print("\nWARNING: '{}' differs between runs: {}".format(key, values))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["run"]
        for kind, key, label in MAP_COLUMNS + PRF_COLUMNS:
            header += [label, label + " delta"]
        writer.writerow(header + ["num_images", "backend", "engine"])
        for run in runs:
            row = [run["name"]]
            for kind, key, _ in MAP_COLUMNS + PRF_COLUMNS:
                value = 100.0 * run[kind][key]
                row += ["{:.2f}".format(value), "{:+.2f}".format(value - 100.0 * baseline[kind][key])]
            info = run.get("run", {})
            writer.writerow(row + [run.get("num_images"), info.get("backend", ""),
                                   info.get("engine", info.get("model", ""))])
    print("\nSaved {}".format(args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
