# Windows

Model export, COCO subset creation and the ONNX Runtime accuracy baseline that the Jetson results are compared against.

**Stack:** Python 3.12, [uv](https://docs.astral.sh/uv/), Ultralytics 8.2.0, ONNX Runtime (CPU), pycocotools.

## Setup

```powershell
cd Edge-AI-Jetson\windows
uv sync
```

## Export the model (once)

```powershell
uv run yolo export model=yolov8n-seg.pt format=onnx imgsz=320,416 opset=12
```

- Produces `windows/yolov8n-seg.onnx` with input `1x3x320x416`. Copy it to the Jetson at `/mnt/usb-drive/models/`.
- TensorRT 8.2 on the Jetson needs ONNX opset 13 or lower.
- Model files (`*.pt`, `*.onnx`) are gitignored.

## Scripts

All scripts run from any folder.

| Step | Command | Output |
| --- | --- | --- |
| 1. COCO subset (500 images, seed 0) | `scripts\create_dataset_windows.bat` | `data/coco_val500/` |
| 2. ONNX sanity check on the test images | `scripts\test_onnx_predict.bat` | `results/predict/windows_ultralytics/` |
| 3. ONNX Runtime COCO baseline | `scripts\eval_onnx_coco.bat` | `results/accuracy/windows_onnx/` |
| 4. Score the Jetson runs and compare | `scripts\compare_metrics.bat` | `results/accuracy/*/metrics.json`, `comparison.csv` |

- Copy `data/coco_val500` to the Jetson before running its evaluation.
- Step 4 needs the Jetson's `results/accuracy/jetson_*/predictions.json` downloaded first.
- To test step 3 on a few images first, add `--limit 20`.

## Evaluation settings (same on Windows and Jetson)

| Setting | Value |
| --- | --- |
| Input size | 320x416, letterbox |
| Confidence threshold | 0.001 |
| NMS IoU | 0.7 |
| Max detections | 300 |
| P/R/F1 thresholds | confidence 0.25, IoU 0.5 |
| Metrics | box and mask mAP@50 and mAP@50–95 (pycocotools) |
