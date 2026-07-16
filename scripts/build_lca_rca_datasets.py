"""
LCA / RCA 분리 데이터셋 빌더 (Colab 실행용)

이미 만들어져 있는 통합 YOLO 탐지 데이터셋(/content/cadica_yolo)을
common_split.csv의 view(LCA/RCA) 컬럼을 기준으로 두 개의 독립 데이터셋으로
나눈다.

전제:
- /content/cadica_yolo/{images,labels}/{train,val,test}/ 가 이미 존재
  (전처리_파이프라인.md 3번 섹션 스크립트로 생성된 상태)
- 파일명이 "{patient_id}_{video_id}_{frame_id}.png" / ".txt" 형식
  (예: p1_v2_00012.png) — CADICA groundtruth 파일 명명 규칙을 그대로 따름
- common_split.csv 컬럼: patient_id, video_id, label, view, n_frames,
  has_groundtruth, split

새로 원본 프레임을 복사하지 않고 symlink만 만들기 때문에 디스크를 거의
쓰지 않는다. Colab에서 재실행해도 안전(이미 있는 symlink는 skip).

사용법 (Colab 셀에 그대로 붙여넣기):
    !python build_lca_rca_datasets.py
또는 import 해서 build_view_datasets() 호출.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SRC_ROOT = Path("/content/cadica_yolo")
SPLIT_CSV = Path("/content/common_split.csv")

VIEW_ROOTS = {
    "LCA": Path("/content/cadica_yolo_lca"),
    "RCA": Path("/content/cadica_yolo_rca"),
}

SPLITS = ["train", "val", "test"]

# groundtruth 파일명 규칙: pX_vY_000ZZ(.png/.txt) -> patient_id, video_id 추출
FRAME_PREFIX_RE = re.compile(r"^(p\d+)_(v\d+)_")


def load_view_map(split_csv: Path = SPLIT_CSV) -> dict:
    df = pd.read_csv(split_csv)
    return {(row.patient_id, row.video_id): row.view for row in df.itertuples()}


def get_view(filename: str, view_map: dict):
    m = FRAME_PREFIX_RE.match(filename)
    if not m:
        return None
    patient_id, video_id = m.group(1), m.group(2)
    return view_map.get((patient_id, video_id))


def build_view_datasets(src_root: Path = SRC_ROOT, split_csv: Path = SPLIT_CSV):
    view_map = load_view_map(split_csv)
    counts = defaultdict(lambda: defaultdict(int))
    unmatched = []

    for split in SPLITS:
        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split
        if not img_dir.exists():
            print(f"[WARN] {img_dir} 없음, skip")
            continue

        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file():
                continue
            view = get_view(img_path.name, view_map)
            if view not in VIEW_ROOTS:
                unmatched.append(img_path.name)
                continue

            dst_root = VIEW_ROOTS[view]
            dst_img_dir = dst_root / "images" / split
            dst_lbl_dir = dst_root / "labels" / split
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            dst_img = dst_img_dir / img_path.name
            if not dst_img.exists():
                os.symlink(img_path.resolve(), dst_img)

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            dst_lbl = dst_lbl_dir / lbl_path.name
            if lbl_path.exists() and not dst_lbl.exists():
                os.symlink(lbl_path.resolve(), dst_lbl)

            counts[view][split] += 1

    for view, root in VIEW_ROOTS.items():
        yaml_text = (
            f"path: {root}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names: ['lesion']\n"
        )
        (root / "data.yaml").write_text(yaml_text)

    print("=== LCA / RCA 분리 결과 (bbox frame 수) ===")
    for view in VIEW_ROOTS:
        row = counts[view]
        total = sum(row.values())
        print(f"{view}: train={row.get('train', 0)} val={row.get('val', 0)} "
              f"test={row.get('test', 0)}  total={total}")

    if unmatched:
        print(f"\n[주의] view 매칭 실패 {len(unmatched)}개 (common_split.csv에 "
              f"없는 patient/video 조합, 또는 unknown view). 예: {unmatched[:5]}")

    return counts


if __name__ == "__main__":
    build_view_datasets()
