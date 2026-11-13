"""Windows baseline: YOLOv8-seg accuracy on the fixed COCO subset with ONNX Runtime.

Preprocessing and post-processing copy the Jetson C++ app step by step
(jetson/cpp/src/preprocess.cpp and yolo_seg.cpp), so differences in the
results come from the inference backend (ONNX Runtime FP32 vs TensorRT FP32/FP16),
not from different pre- or post-processing code:

    letterbox    scale = min(W/w, H/h), round-half-up size, centred, grey 114 padding, INTER_LINEAR
    input        BGR -> RGB, /255, CHW float32
    decode       best class per anchor, keep score >= conf, boxes clipped to the input
    NMS          class-aware greedy, suppress IoU > iou, at most max_det boxes
    masks        coefficients @ prototypes, exact affine warp to the box, logit > 0

Thresholds default to the usual COCO-evaluation settings (conf 0.001, IoU 0.7, 300 boxes).
Use the SAME values on the Jetson. Precision/recall/F1 then only count detections
with score >= 0.25, which gives the same result as running NMS at conf 0.25.

Outputs (default results/accuracy/windows_onnx/):
    predictions.json   COCO results: bbox + RLE mask per detection
    metrics.json       box/mask mAP@50, mAP@50-95, P, R, F1 + run settings
"""

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[2]   # Edge-AI-Jetson/
sys.path.insert(0, str(ROOT / "scripts"))     # shared Windows/Jetson scripts
from coco_metrics import evaluate, format_table  # noqa: E402

DATA = ROOT / "data" / "coco_val500"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=ROOT / "windows" / "yolov8n-seg.onnx")
    p.add_argument("--images", type=Path, default=DATA / "images")
    p.add_argument("--ann", type=Path, default=DATA / "instances_val2017_subset500.json")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "accuracy" / "windows_onnx")
    p.add_argument("--conf", type=float, default=0.001, help="score threshold before NMS")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--pr-conf", type=float, default=0.25, help="score threshold for P/R/F1")
    p.add_argument("--pr-iou", type=float, default=0.5, help="match IoU for P/R/F1")
    p.add_argument("--limit", type=int, default=0, help="only the first N images (quick test)")
    return p.parse_args()


# ---------------------------------------------------------------- preprocessing (= preprocess.cpp)

def letterbox(image: np.ndarray, in_w: int, in_h: int):
    h, w = image.shape[:2]
    scale = min(np.float32(in_w) / np.float32(w), np.float32(in_h) / np.float32(h))   # float, like C++
    new_w = min(in_w, int(math.floor(w * scale + 0.5)))                               # std::round
    new_h = min(in_h, int(math.floor(h * scale + 0.5)))
    pad_x, pad_y = (in_w - new_w) // 2, (in_h - new_h) // 2

    canvas = np.full((in_h, in_w, 3), 114, np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(image, (new_w, new_h),
                                                                  interpolation=cv2.INTER_LINEAR)
    blob = (canvas[:, :, ::-1].astype(np.float32) * np.float32(1.0 / 255.0)).transpose(2, 0, 1)
    return np.ascontiguousarray(blob[None]), float(scale), pad_x, pad_y


# ---------------------------------------------------------------- post-processing (= yolo_seg.cpp)

def decode(out: np.ndarray, num_classes: int, conf: float, in_w: int, in_h: int):
    scores = out[4:4 + num_classes]                         # (classes, anchors)
    classes = scores.argmax(0)
    best = scores[classes, np.arange(out.shape[1])]
    anchors = np.nonzero(best >= conf)[0]

    cx, cy, w, h = out[0:4, anchors]
    x1 = np.maximum(0.0, cx - 0.5 * w)
    y1 = np.maximum(0.0, cy - 0.5 * h)
    x2 = np.minimum(float(in_w), cx + 0.5 * w)
    y2 = np.minimum(float(in_h), cy + 0.5 * h)
    valid = (x2 > x1) & (y2 > y1)

    boxes = np.stack([x1, y1, x2, y2], axis=1)[valid]
    return boxes, best[anchors][valid], classes[anchors][valid], anchors[valid]


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float, max_det: int) -> np.ndarray:
    """Class-aware greedy NMS (suppress IoU > iou_thr); kept indices, highest score first."""
    xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]).tolist()
    kept = np.asarray(cv2.dnn.NMSBoxesBatched(xywh, scores.tolist(), classes.tolist(), 0.0, iou_thr),
                      dtype=np.int64).ravel()
    return kept[np.argsort(-scores[kept], kind="stable")][:max_det]


def masks_to_rle(logits: np.ndarray, int_boxes: list[tuple[int, int, int, int]],
                 scale: float, pad_x: int, pad_y: int, in_w: int, in_h: int,
                 image_w: int, image_h: int) -> list[dict]:
    """Same affine warp as YoloSegDecoder::buildMasks, pasted into a full-image mask."""
    proto_h, proto_w = logits.shape[1:]
    sx, sy = proto_w / in_w, proto_h / in_h
    rles = []
    for logit, (bx, by, bw, bh) in zip(logits, int_boxes):
        m = np.array([[scale * sx, 0.0, ((bx + 0.5) * scale + pad_x) * sx - 0.5],
                      [0.0, scale * sy, ((by + 0.5) * scale + pad_y) * sy - 0.5]], np.float32)
        warped = cv2.warpAffine(logit, m, (bw, bh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
        full = np.zeros((image_h, image_w), np.uint8, order="F")
        full[by:by + bh, bx:bx + bw] = warped > 0
        rle = mask_utils.encode(full)
        rle["counts"] = rle["counts"].decode("ascii")
        rles.append(rle)
    return rles


def predict_image(session, input_name, image, info, args, cat_ids, image_id):
    in_h, in_w, num_classes, mask_ch = info
    image_h, image_w = image.shape[:2]

    blob, scale, pad_x, pad_y = letterbox(image, in_w, in_h)
    t0 = time.perf_counter()
    out0, out1 = session.run(None, {input_name: blob})
    infer_ms = (time.perf_counter() - t0) * 1000.0
    out0, protos = out0[0], out1[0]

    boxes, scores, classes, anchors = decode(out0, num_classes, args.conf, in_w, in_h)
    if len(boxes) == 0:
        return [], infer_ms
    keep = nms(boxes, scores, classes, args.iou, args.max_det)

    # Map to original pixels: float box for COCO, integer box (floor/ceil) for the mask, as in C++.
    predictions, int_boxes, mask_rows = [], [], []
    for k in keep:
        x1, y1, x2, y2 = (boxes[k] - [pad_x, pad_y, pad_x, pad_y]) / scale
        ix1, iy1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        ix2, iy2 = min(image_w, math.ceil(x2)), min(image_h, math.ceil(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue                                   # entirely in the padding
        fx1, fy1 = min(max(x1, 0.0), image_w), min(max(y1, 0.0), image_h)
        fx2, fy2 = min(max(x2, 0.0), image_w), min(max(y2, 0.0), image_h)
        predictions.append({
            "image_id": image_id,
            "category_id": cat_ids[int(classes[k])],
            "bbox": [round(fx1, 2), round(fy1, 2), round(fx2 - fx1, 2), round(fy2 - fy1, 2)],
            "score": round(float(scores[k]), 5),
        })
        int_boxes.append((ix1, iy1, ix2 - ix1, iy2 - iy1))
        mask_rows.append(anchors[k])

    if predictions:
        coeffs = out0[4 + num_classes:4 + num_classes + mask_ch, mask_rows].T          # (n, K)
        logits = (coeffs @ protos.reshape(mask_ch, -1)).reshape(-1, *protos.shape[1:]).astype(np.float32)
        for p, rle in zip(predictions, masks_to_rle(logits, int_boxes, scale, pad_x, pad_y,
                                                    in_w, in_h, image_w, image_h)):
            p["segmentation"] = rle
    return predictions, infer_ms


def main() -> int:
    args = parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    in_h, in_w = inp.shape[2], inp.shape[3]
    det_shape, proto_shape = session.get_outputs()[0].shape, session.get_outputs()[1].shape
    mask_ch = proto_shape[1]
    num_classes = det_shape[1] - 4 - mask_ch
    print(f"Model {args.model.name}: input {in_w}x{in_h}, {num_classes} classes, {mask_ch} mask channels")

    gt = COCO(str(args.ann))
    cat_ids = sorted(gt.getCatIds())            # YOLO class index -> COCO category id
    if len(cat_ids) != num_classes:
        sys.exit(f"Annotation has {len(cat_ids)} categories, model has {num_classes}")
    img_ids = sorted(gt.getImgIds())
    if args.limit:
        img_ids = img_ids[:args.limit]

    predictions, infer_times = [], []
    start = time.perf_counter()
    for n, img in enumerate(gt.loadImgs(img_ids), 1):
        image = cv2.imread(str(args.images / img["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            sys.exit(f"Cannot read {args.images / img['file_name']}")
        preds, ms = predict_image(session, inp.name, image, (in_h, in_w, num_classes, mask_ch),
                                  args, cat_ids, img["id"])
        predictions.extend(preds)
        infer_times.append(ms)
        if n % 50 == 0 or n == len(img_ids):
            print(f"  {n}/{len(img_ids)} images, {len(predictions)} predictions")
    elapsed = time.perf_counter() - start

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")

    metrics = evaluate(args.ann, predictions, img_ids, args.pr_conf, args.pr_iou)
    metrics["run"] = {
        "platform": "windows",
        "backend": f"onnxruntime {ort.__version__} (CPUExecutionProvider)",
        "precision": "fp32",
        "model": args.model.name,
        "input": f"{in_h}x{in_w}",
        "conf": args.conf, "iou": args.iou, "max_det": args.max_det,
        "annotations": args.ann.name,
        "cpu": platform.processor(),
        "mean_inference_ms": round(float(np.mean(infer_times)), 2),
        "total_seconds": round(elapsed, 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print(format_table(metrics, f"Windows ONNX Runtime FP32 on {len(img_ids)} images "
                                f"(conf {args.conf}, NMS IoU {args.iou}, max_det {args.max_det})"))
    print(f"\nSaved {args.out / 'predictions.json'} and metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
