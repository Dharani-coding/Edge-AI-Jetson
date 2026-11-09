"""COCO accuracy metrics for YOLOv8-seg predictions (any platform).

Input: the subset annotation JSON and a COCO results JSON, i.e. a list of
    {"image_id", "category_id", "bbox": [x, y, w, h], "score",
     "segmentation": RLE {"size": [h, w], "counts": str | list}}
Uncompressed RLE (counts as a list of ints, easy to write from C++) is accepted.

Reports, for boxes and masks:
    mAP@50, mAP@50-95                  pycocotools COCOeval (maxDets 100)
    precision, recall, F1              detections with score >= --pr-conf, matched at IoU --pr-iou
                                       using COCOeval's own matching (crowd regions ignored)

    uv run --with pycocotools python coco_metrics.py --ann <gt.json> --pred <predictions.json> [--out metrics.json]
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRIC_KEYS = ("mAP50", "mAP50_95", "precision", "recall", "f1")


def _compress_rle(predictions: list[dict]) -> None:
    """Convert uncompressed RLE (counts list) to compressed, as loadRes requires."""
    for p in predictions:
        seg = p.get("segmentation")
        if seg and isinstance(seg.get("counts"), list):
            h, w = seg["size"]
            rle = mask_utils.frPyObjects(seg, h, w)
            rle["counts"] = rle["counts"].decode("ascii")
            p["segmentation"] = rle


def _precision_recall(ev: COCOeval, conf: float, iou: float) -> dict:
    """P/R/F1 at a fixed score threshold, from COCOeval's per-image matches."""
    t = int(np.argmin(np.abs(ev.params.iouThrs - iou)))
    all_area = ev.params.areaRng[0]
    tp = fp = num_gt = 0

    for e in ev.evalImgs:
        if e is None or e["aRng"] != all_area:
            continue
        scores = np.asarray(e["dtScores"])
        matched = e["dtMatches"][t] > 0
        ignored = e["dtIgnore"][t].astype(bool)          # e.g. matched to a crowd region
        keep = (scores >= conf) & ~ignored
        tp += int((keep & matched).sum())
        fp += int((keep & ~matched).sum())
        num_gt += int((~np.asarray(e["gtIgnore"]).astype(bool)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "num_gt": num_gt}


def _evaluate_type(gt: COCO, predictions: list[dict], iou_type: str, img_ids: list[int],
                   pr_conf: float, pr_iou: float) -> dict:
    # Give each evaluation only its own field: with both present, loadRes would
    # take the bbox path and compute mask areas from the boxes.
    drop = "segmentation" if iou_type == "bbox" else "bbox"
    need = "bbox" if iou_type == "bbox" else "segmentation"
    dets = [{k: v for k, v in p.items() if k != drop} for p in predictions if need in p]
    if not dets:
        return {k: 0.0 for k in METRIC_KEYS}

    with contextlib.redirect_stdout(io.StringIO()):    # pycocotools is chatty
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iou_type)
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    return {"mAP50": float(ev.stats[1]), "mAP50_95": float(ev.stats[0]),
            **_precision_recall(ev, pr_conf, pr_iou)}


def evaluate(ann_path: Path, predictions: list[dict], img_ids: list[int] | None = None,
             pr_conf: float = 0.25, pr_iou: float = 0.5) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_path))
    img_ids = sorted(img_ids if img_ids is not None else gt.getImgIds())

    predictions = [dict(p) for p in predictions]
    _compress_rle(predictions)

    return {
        "num_images": len(img_ids),
        "num_predictions": len(predictions),
        "pr_conf": pr_conf,
        "pr_iou": pr_iou,
        "box": _evaluate_type(gt, predictions, "bbox", img_ids, pr_conf, pr_iou),
        "mask": _evaluate_type(gt, predictions, "segm", img_ids, pr_conf, pr_iou),
    }


def format_table(metrics: dict, title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"{'':6} {'mAP@50':>8} {'mAP@50-95':>10} {'P':>7} {'R':>7} {'F1':>7}"
                 f"   (P/R/F1 at conf {metrics['pr_conf']}, IoU {metrics['pr_iou']})")
    for kind in ("box", "mask"):
        m = metrics[kind]
        lines.append(f"{kind:6} {m['mAP50']:8.4f} {m['mAP50_95']:10.4f} "
                     f"{m['precision']:7.4f} {m['recall']:7.4f} {m['f1']:7.4f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write metrics JSON here")
    parser.add_argument("--pr-conf", type=float, default=0.25)
    parser.add_argument("--pr-iou", type=float, default=0.5)
    args = parser.parse_args()

    # Evaluated on every image in the annotation file, including images with no predictions.
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    metrics = evaluate(args.ann, predictions, None, args.pr_conf, args.pr_iou)

    print(format_table(metrics, f"{args.pred}  ({metrics['num_images']} images)"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
