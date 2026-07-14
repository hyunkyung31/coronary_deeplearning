"""
CADICA + ARCADE 병합 데이터셋 빌더 (Colab 실행용)

원칙 (팀 합의):
    Train: CADICA train + ARCADE train (convert_arcade_to_yolo.py로 변환된 것)
    Val  : CADICA val만
    Test : CADICA test만

ARCADE는 학습량 보강용일 뿐이고, Group5(YOLO)와 공정 비교하려면 평가는
CADICA val/test로 고정해야 하기 때문. symlink 방식이라 원본 이미지 복사 없음.

전제:
    /content/cadica_yolo/{images,labels}/{train,val,test}/  (기존 CADICA-only 데이터셋)
    /content/arcade_yolo/{images,labels}/train/              (convert_arcade_to_yolo.py 출력)

출력:
    /content/cadica_arcade_yolo/{images,labels}/{train,val,test}/
    /content/cadica_arcade_yolo/data.yaml
"""

import os
from pathlib import Path

CADICA_ROOT = Path("/content/cadica_yolo")
ARCADE_ROOT = Path("/content/arcade_yolo")
OUT_ROOT = Path("/content/cadica_arcade_yolo")


def link_dir(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        return 0
    n = 0
    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        dst = dst_dir / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)
        n += 1
    return n


def build_merged_dataset(cadica_root: Path = CADICA_ROOT,
                          arcade_root: Path = ARCADE_ROOT,
                          out_root: Path = OUT_ROOT) -> dict:
    counts: dict = {}

    # val / test: CADICA만 (ARCADE 섞지 않음 -> 공정 비교 유지)
    for split in ("val", "test"):
        n_img = link_dir(cadica_root / "images" / split, out_root / "images" / split)
        link_dir(cadica_root / "labels" / split, out_root / "labels" / split)
        counts[split] = {"CADICA": n_img}

    # train: CADICA train + ARCADE train
    n_cadica = link_dir(cadica_root / "images" / "train", out_root / "images" / "train")
    link_dir(cadica_root / "labels" / "train", out_root / "labels" / "train")
    n_arcade = link_dir(arcade_root / "images" / "train", out_root / "images" / "train")
    link_dir(arcade_root / "labels" / "train", out_root / "labels" / "train")
    counts["train"] = {"CADICA": n_cadica, "ARCADE": n_arcade}

    (out_root / "data.yaml").write_text(
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['lesion']\n"
    )

    print("=== CADICA + ARCADE 병합 결과 ===")
    for split, d in counts.items():
        detail = ", ".join(f"{k}={v}" for k, v in d.items())
        print(f"{split:6} {detail}  total={sum(d.values())}")

    return counts


if __name__ == "__main__":
    build_merged_dataset()
