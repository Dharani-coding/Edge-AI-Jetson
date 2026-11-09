"""Create a fixed COCO val2017 subset with a matching annotation JSON.

The same --num and --seed always select the same images, so every engine

Output:
    <out>/images/*.jpg                      only the selected images
    <out>/instances_val2017_subset<N>.json  COCO-format annotations for them
    <out>/image_ids.txt                     selected image ids, one per line
"""

import argparse
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pycocotools.coco import COCO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann", type=Path, required=True, help="instances_val2017.json (relative to project root)")
    parser.add_argument("--out", type=Path, required=True, help="output directory (relative to project root)")
    parser.add_argument("--num", type=int, default=500, help="number of images")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--workers", type=int, default=8, help="parallel image downloads")
    return parser.parse_args()


def download(url: str, dest: Path, retries: int = 3) -> str | None:
    """Download url to dest, skipping files already present. Returns an error message or None."""
    if dest.exists() and dest.stat().st_size > 0:
        return None
    tmp = dest.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return None
        except OSError as error:
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                return f"{url}: {error}"
    return None


def main() -> int:
    args = parse_args()
    args.ann = project_path(args.ann)
    args.out = project_path(args.out)
    coco = COCO(str(args.ann))

    # Sort before sampling so the selection does not depend on JSON order.
    all_ids = sorted(coco.getImgIds())
    if not 0 < args.num <= len(all_ids):
        print(f"--num must be between 1 and {len(all_ids)}", file=sys.stderr)
        return 1
    chosen = sorted(random.Random(args.seed).sample(all_ids, args.num))

    images = coco.loadImgs(chosen)
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=chosen))   # includes iscrowd, as in the full set

    info = dict(coco.dataset.get("info", {}))
    info["description"] = f"COCO val2017 subset: {args.num} images, seed {args.seed}"

    subset = {
        "info": info,
        "licenses": coco.dataset.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco.loadCats(coco.getCatIds()),   # all 80, keeps category ids unchanged
    }

    args.out.mkdir(parents=True, exist_ok=True)
    ann_out = args.out / f"instances_val2017_subset{args.num}.json"
    ann_out.write_text(json.dumps(subset), encoding="utf-8")
    (args.out / "image_ids.txt").write_text("\n".join(map(str, chosen)) + "\n", encoding="utf-8")

    # Check that the file loads cleanly with pycocotools.
    check = COCO(str(ann_out))
    print(f"Wrote {ann_out}: {len(check.getImgIds())} images, "
          f"{len(check.getAnnIds())} annotations, {len(check.getCatIds())} categories")

    image_dir = args.out / "images"
    image_dir.mkdir(exist_ok=True)
    print(f"Downloading {len(images)} images to {image_dir} ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        errors = [e for e in pool.map(lambda im: download(im["coco_url"], image_dir / im["file_name"]), images) if e]

    if errors:
        print(f"{len(errors)} downloads failed (run again to retry):", *errors, sep="\n  ", file=sys.stderr)
        return 1

    print(f"Done: {len(images)} images in {image_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
