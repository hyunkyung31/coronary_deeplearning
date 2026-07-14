"""
RT-DETR LCA / RCA 분리 학습 + 평가 (Colab 실행용)

전제:
- build_lca_rca_datasets.py 를 먼저 실행해서
  /content/cadica_yolo_lca/data.yaml, /content/cadica_yolo_rca/data.yaml 준비 완료
- train_v3_amp_on(640, 통합 데이터) 결과가 baseline:
    Val  P=0.229 R=0.219 mAP50=0.117 mAP50-95=0.030
    Test P=0.592 R=0.289 mAP50=0.308 mAP50-95=0.150

전략:
- LCA/RCA 모두 rtdetr-l.pt(원본 pretrained)에서 새로 시작.
  (통합 학습된 last.pt/best.pt에서 이어가면 LCA 위주로 편향된 특성이
   RCA 쪽에 그대로 넘어갈 위험이 있어서, 공정한 비교를 위해 둘 다
   pretrained에서 새로 시작한다.)
- 하이퍼파라미터는 train_v3_amp_on과 동일하게 고정
  (imgsz/lr/optimizer/amp 등 변수를 한 번에 하나씩만 바꾸기 위함).
- RCA는 데이터가 더 적으므로(LCA 216 vs RCA 118 video) 과적합 신호를
  특히 주의해서 봐야 함 -> patience=20 유지, 필요하면 이후 ablation에서
  RCA만 augmentation을 더 세게 주는 것도 검토.

Colab 셀 구성 예시:
    1) build_lca_rca_datasets.py 실행
    2) 아래 train_view("LCA") / train_view("RCA") 각각 실행 (순차 or 별도 셀)
    3) eval_view("LCA") / eval_view("RCA") 로 성능 확인
    4) summarize() 로 baseline과 나란히 비교
"""

from pathlib import Path

from ultralytics import RTDETR

BASE = Path(
    "/content/drive/MyDrive/🐥new_2_project/04_DL_project/RT-DETR_result"
)

DATA_YAML = {
    "LCA": "/content/cadica_yolo_lca/data.yaml",
    "RCA": "/content/cadica_yolo_rca/data.yaml",
}

RUN_NAME = {
    "LCA": "train_lca_v1",
    "RCA": "train_rca_v1",
}

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

# 통합(train_v3_amp_on) baseline — 비교용으로 하드코딩
BASELINE = {
    "val": dict(P=0.229, R=0.219, mAP50=0.117, mAP50_95=0.030),
    "test": dict(P=0.592, R=0.289, mAP50=0.308, mAP50_95=0.150),
}

_results = {}


def train_view(view: str):
    assert view in ("LCA", "RCA")
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=DATA_YAML[view],
        name=RUN_NAME[view],
        **COMMON_TRAIN_KWARGS,
    )
    return model


def eval_view(view: str, imgsz: int = 640):
    assert view in ("LCA", "RCA")
    weights = BASE / RUN_NAME[view] / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} 없음 — train_view('{view}') 먼저 실행")

    model = RTDETR(str(weights))
    val = model.val(data=DATA_YAML[view], split="val", imgsz=imgsz)
    test = model.val(data=DATA_YAML[view], split="test", imgsz=imgsz)

    result = {
        "val": dict(P=val.box.mp, R=val.box.mr, mAP50=val.box.map50, mAP50_95=val.box.map),
        "test": dict(P=test.box.mp, R=test.box.mr, mAP50=test.box.map50, mAP50_95=test.box.map),
    }
    _results[view] = result

    print(f"[{view}] VAL  P={result['val']['P']:.3f} R={result['val']['R']:.3f} "
          f"mAP50={result['val']['mAP50']:.3f} mAP50-95={result['val']['mAP50_95']:.3f}")
    print(f"[{view}] TEST P={result['test']['P']:.3f} R={result['test']['R']:.3f} "
          f"mAP50={result['test']['mAP50']:.3f} mAP50-95={result['test']['mAP50_95']:.3f}")
    return result


def summarize():
    """LCA / RCA / 통합 baseline을 한 표로 정리해서 출력."""
    header = f"{'split':6} {'group':10} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}"
    print(header)
    print("-" * len(header))
    for split in ("val", "test"):
        b = BASELINE[split]
        print(f"{split:6} {'baseline(합)':10} {b['P']:6.3f} {b['R']:6.3f} "
              f"{b['mAP50']:7.3f} {b['mAP50_95']:9.3f}")
        for view in ("LCA", "RCA"):
            if view not in _results:
                continue
            r = _results[view][split]
            print(f"{split:6} {view:10} {r['P']:6.3f} {r['R']:6.3f} "
                  f"{r['mAP50']:7.3f} {r['mAP50_95']:9.3f}")


if __name__ == "__main__":
    for v in ("LCA", "RCA"):
        train_view(v)
    for v in ("LCA", "RCA"):
        eval_view(v)
    summarize()
