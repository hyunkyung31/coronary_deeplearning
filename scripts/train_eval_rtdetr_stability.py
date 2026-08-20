"""
RT-DETR CADICA-only 강한 안정화 학습 + 평가 (Colab)

데이터/split은 baseline과 동일 (`/content/cadica_yolo`).
바꾸는 것은 학습 스케줄·augmentation 강도뿐.
"""

from pathlib import Path

from ultralytics import RTDETR

BASE = Path(
    "/content/drive/MyDrive/🐥new_2_project/04_DL_project/RT-DETR_result"
)
DATA_YAML = "/content/cadica_yolo/data.yaml"
RUN_NAME = "train_stability_v1"

# CADICA-only baseline (train_v3_amp_on)
BASELINE = {
    "val": dict(P=0.229, R=0.219, mAP50=0.117, mAP50_95=0.030),
    "test": dict(P=0.592, R=0.289, mAP50=0.308, mAP50_95=0.150),
}

STABILITY_KWARGS = dict(
    data=DATA_YAML,
    name=RUN_NAME,
    project=str(BASE),
    epochs=100,
    imgsz=640,
    batch=8,
    optimizer="AdamW",
    amp=True,
    # --- stronger stability vs baseline ---
    lr0=0.00015,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=6.0,
    weight_decay=0.0001,
    dropout=0.1,  # extra regularization; RT-DETR decoder supports this
    patience=35,
    mosaic=0.5,
    close_mosaic=20,
    mixup=0.0,
    copy_paste=0.0,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    scale=0.3,
    translate=0.05,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.01,
    hsv_s=0.4,
    hsv_v=0.3,
)

# Stage-2: cold fine-tune from stage-1 best.pt, near-zero aug, very low lr.
# Only run this if train()+eval_and_compare() shows stage-1 beats baseline
# on test, or is close but still oscillating on val.
FINETUNE_KWARGS = dict(
    data=DATA_YAML,
    name="train_stability_v1_ft",
    project=str(BASE),
    epochs=40,
    imgsz=640,
    batch=8,
    optimizer="AdamW",
    amp=True,
    lr0=0.00003,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=0,
    weight_decay=0.0001,
    dropout=0.1,
    patience=15,
    mosaic=0.0,
    close_mosaic=0,
    mixup=0.0,
    copy_paste=0.0,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    scale=0.1,
    translate=0.02,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.005,
    hsv_s=0.2,
    hsv_v=0.15,
)


def train():
    model = RTDETR("rtdetr-l.pt")
    model.train(**STABILITY_KWARGS)
    return model


def finetune_stage2():
    """Cold fine-tune stage-1 best.pt with near-zero aug + tiny lr (S3)."""
    stage1_best = BASE / RUN_NAME / "weights" / "best.pt"
    if not stage1_best.exists():
        raise FileNotFoundError(f"{stage1_best} 없음 — train() 먼저 실행")
    model = RTDETR(str(stage1_best))
    model.train(**FINETUNE_KWARGS)
    return model


def eval_and_compare(run_name: str = RUN_NAME, label: str = "stability_v1", imgsz: int = 640):
    weights = BASE / run_name / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} 없음 — train() 먼저 실행")

    model = RTDETR(str(weights))
    val = model.val(data=DATA_YAML, split="val", imgsz=imgsz)
    test = model.val(data=DATA_YAML, split="test", imgsz=imgsz)

    result = {
        "val": dict(
            P=val.box.mp, R=val.box.mr, mAP50=val.box.map50, mAP50_95=val.box.map
        ),
        "test": dict(
            P=test.box.mp, R=test.box.mr, mAP50=test.box.map50, mAP50_95=test.box.map
        ),
    }

    header = f"{'split':6} {'group':18} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}"
    print(header)
    print("-" * len(header))
    for split in ("val", "test"):
        b = BASELINE[split]
        print(
            f"{split:6} {'baseline(v3)':18} {b['P']:6.3f} {b['R']:6.3f} "
            f"{b['mAP50']:7.3f} {b['mAP50_95']:9.3f}"
        )
        r = result[split]
        print(
            f"{split:6} {label:18} {r['P']:6.3f} {r['R']:6.3f} "
            f"{r['mAP50']:7.3f} {r['mAP50_95']:9.3f}"
        )
    return result


if __name__ == "__main__":
    train()
    eval_and_compare(RUN_NAME, "stability_v1")
    # If stage-1 looks promising but still noisy, uncomment:
    # finetune_stage2()
    # eval_and_compare("train_stability_v1_ft", "stability_v1_ft")
