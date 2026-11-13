# Windows

ONNX Runtime baseline and COCO dataset preparation (uv project, Python 3.12).

## Session setup (PowerShell)

```powershell
cd .\Edge-AI-Jetson\windows\
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\activate
```

## Scripts (`scripts/`, run from anywhere)

| Script | Does |
| --- | --- |
| `create_dataset_windows.bat` | Fixed COCO val subset (500 images, seed 0) -> `data/coco_val500` |
| `test_onnx_predict.bat` | Ultralytics ONNX predict on the test images -> `results/predict/windows_ultralytics` |
| `eval_onnx_coco.bat` | ONNX Runtime COCO evaluation (baseline) -> `results/accuracy/windows_onnx` |
| `compare_metrics.bat` | Score the Jetson runs and compare with the baseline -> `results/accuracy/comparison.csv` |
