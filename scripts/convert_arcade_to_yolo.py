"""
ARCADE stenosis(COCO JSON) -> YOLO 변환 (Colab 실행용)

전제 (ARCADE zip 압축 해제 후 기본 구조, 버전에 따라 폴더명이 조금 다를 수 있음):
    /content/arcade/stenosis/train/images/*.png
    /content/arcade/stenosis/train/annotations/train.json
    /content/arcade/stenosis/val/...
    /content/arcade/stenosis/test/...

- syntax/ 폴더는 혈관 SYNTAX 세그멘테이션용이라 탐지 메인에는 쓰지 않는다.
  (stenosis/ 폴더만 사용)
- categories 중 이름에 "sten"이 포함된 카테고리만 lesion(class 0)으로 변환.
  일부 배포판은 stenosis 서브셋에도 SYNTAX 세그먼트(1~25) + stenosis(26) 카테고리가
  섞여 있을 수 있어서, "stenosis"라는 이름이 명시된 것만 걸러낸다.
  매칭되는 카테고리를 못 찾으면(스키마가 다른 버전) 전체 annotation을 lesion으로
  간주하고 경고를 출력한다 -- 이 경우 반드시 labels.jpg / 샘플 bbox를 눈으로 확인할 것.
- CADICA 파일명("pX_vY_00000")과 겹치지 않도록 "arcade_" prefix를 붙인다.
- ARCADE test는 변환하지 않는다. (평가는 CADICA val/test만 쓰기로 합의)
  --include-val 옵션을 주면 ARCADE val도 train 보강에 포함할 수 있다.

출력:
    /content/arcade_yolo/images/train/arcade_<split>_<image_id>.png (symlink)
    /content/arcade_yolo/labels/train/arcade_<split>_<image_id>.txt
"""

import argparse
import json
import os
from pathlib import Path


def find_stenosis_category_ids(categories: list) -> list:
    return [c["id"] for c in categories if "sten" in c.get("name", "").lower()]


def _find_annotation_file(arcade_root: Path, split: str) -> Path | None:
    ann_path = arcade_root / "stenosis" / split / "annotations" / f"{split}.json"
    if ann_path.exists():
        return ann_path
    candidates = list((arcade_root / "stenosis" / split).glob("**/*.json"))
    return candidates[0] if candidates else None


def convert_split(arcade_root: Path, split: str, out_root: Path, out_split: str = "train") -> int:
    img_dir = arcade_root / "stenosis" / split / "images"
    ann_path = _find_annotation_file(arcade_root, split)
    if ann_path is None:
        print(f"[WARN] {split} annotation json을 못 찾음, skip "
              f"(경로 확인: {arcade_root / 'stenosis' / split})")
        return 0

    data = json.loads(ann_path.read_text())
    images = {im["id"]: im for im in data["images"]}
    categories = data.get("categories", [])
    stenosis_ids = find_stenosis_category_ids(categories)

    if stenosis_ids:
        names = [c["name"] for c in categories if c["id"] in stenosis_ids]
        print(f"[{split}] stenosis category id(s)={stenosis_ids} (name={names})")
    else:
        print(f"[{split}] [주의] 'sten' 포함 카테고리를 못 찾음 -> 전체 annotation을 "
              f"lesion으로 처리합니다. 전체 카테고리={[c['name'] for c in categories]} "
              f"-- 반드시 시각적으로 bbox를 확인하세요.")

    boxes_per_image: dict = {}
    for ann in data["annotations"]:
        if stenosis_ids and ann["category_id"] not in stenosis_ids:
            continue
        boxes_per_image.setdefault(ann["image_id"], []).append(ann["bbox"])

    out_img_dir = out_root / "images" / out_split
    out_lbl_dir = out_root / "labels" / out_split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for image_id, boxes in boxes_per_image.items():
        im = images.get(image_id)
        if im is None:
            continue
        img_w, img_h = im["width"], im["height"]
        src_img = img_dir / im["file_name"]
        if not src_img.exists():
            continue

        stem = f"arcade_{split}_{image_id}"
        dst_img = out_img_dir / f"{stem}{src_img.suffix}"
        if not dst_img.exists():
            os.symlink(src_img.resolve(), dst_img)

        lines = []
        for x, y, w, h in boxes:
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")

        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        n_written += 1

    print(f"[{split}] {n_written}장 변환 완료 -> {out_img_dir}")
    return n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-root", default="/content/arcade")
    parser.add_argument("--out-root", default="/content/arcade_yolo")
    parser.add_argument("--include-val", action="store_true",
                         help="ARCADE val도 train 보강에 포함 (기본: train만 사용)")
    args = parser.parse_args()

    arcade_root = Path(args.arcade_root)
    out_root = Path(args.out_root)

    convert_split(arcade_root, "train", out_root, out_split="train")
    if args.include_val:
        convert_split(arcade_root, "val", out_root, out_split="train")
    # test는 절대 변환하지 않음 (CADICA test만 평가 기준으로 유지)


if __name__ == "__main__":
    main()
