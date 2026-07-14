"""
RT-DETR CADICA+ARCADE 병합 학습 + CADICA 전용 평가 (Colab 실행용)

Train: /content/cadica_arcade_yolo (CADICA train + ARCADE train)
Eval : 같은 data.yaml의 val/test  == CADICA val/test
       (build_cadica_arcade_dataset.py에서 val/test는 CADICA만 symlink 했으므로
        추가 조치 없이 그대로 공정 비교 가능)

CADICA-only baseline(train_v3_amp_on)과 하이퍼파라미터를 동일하게 고정하고,
"ARCADE 추가" 한 가지 변수만 바꿔서 비교한다.
"""

from pathlib import Path

from ultralytics import RTDETR

BASE = Path(
    "/content/drive/MyDrive/🐥new_2_project/04_DL_project/RT-DETR_result"
)
DATA_YAML = "/content/cadica_arcade_yolo/data.yaml"
RUN_NAME = "train_cadica_arcade_v1"

COMMON_TRAIN_KWARGS = dict(
    epochs=50,
    imgsz=640,
    batch=8,
    lr0=0.0005,
    optimizer="AdamW",
    amp=True,
    patience=20,
    project=str(BASE),
)

# CADICA-only baseline (train_v3_amp_on) — 비교용 하드코딩
BASELINE = {
    "val": dict(P=0.229, R=0.219, mAP50=0.117, mAP50_95=0.030),
    "test": dict(P=0.592, R=0.289, mAP50=0.308, mAP50_95=0.150),
}


def train():
    model = RTDETR("rtdetr-l.pt")  # 원본 pretrained에서 새로 시작
    model.train(data=DATA_YAML, name=RUN_NAME, **COMMON_TRAIN_KWARGS)
    return model


def eval_and_compare(imgsz: int = 640):
    weights = BASE / RUN_NAME / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} 없음 — train() 먼저 실행")

    model = RTDETR(str(weights))
    val = model.val(data=DATA_YAML, split="val", imgsz=imgsz)
    test = model.val(data=DATA_YAML, split="test", imgsz=imgsz)

    result = {
        "val": dict(P=val.box.mp, R=val.box.mr, mAP50=val.box.map50, mAP50_95=val.box.map),
        "test": dict(P=test.box.mp, R=test.box.mr, mAP50=test.box.map50, mAP50_95=test.box.map),
    }

    header = f"{'split':6} {'group':16} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}"
    print(header)
    print("-" * len(header))
    for split in ("val", "test"):
        b = BASELINE[split]
        print(f"{split:6} {'CADICA-only(합)':16} {b['P']:6.3f} {b['R']:6.3f} "
              f"{b['mAP50']:7.3f} {b['mAP50_95']:9.3f}")
        r = result[split]
        print(f"{split:6} {'CADICA+ARCADE':16} {r['P']:6.3f} {r['R']:6.3f} "
              f"{r['mAP50']:7.3f} {r['mAP50_95']:9.3f}")

    return result


if __name__ == "__main__":
    train()
    eval_and_compare()
