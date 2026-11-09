@echo off
rem Windows ONNX Runtime baseline: box/mask mAP, precision, recall, F1 on data\coco_val500.
rem Options are passed through, e.g.  eval_onnx_coco.bat --limit 20
uv run --project "%~dp0.." --with pycocotools python "%~dp0eval_onnx_coco.py" %*
