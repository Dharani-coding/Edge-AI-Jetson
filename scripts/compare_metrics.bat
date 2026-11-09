cd C:\Users\ManikandanSR\workspace\jetson_nano\Edge-AI-Jetson

@REM  1. Score each Jetson run (writes metrics.json next to its predictions.json)
uv run --project windows --with pycocotools python scripts/coco_metrics.py --pred data/results/accuracy/jetson_fp16/predictions.json
uv run --project windows --with pycocotools python scripts/coco_metrics.py --pred data/results/accuracy/jetson_fp32/predictions.json

@REM  2. Compare both against the Windows ONNX baseline (also writes data/results/accuracy/comparison.csv)
uv run --project windows python scripts/compare_metrics.py
