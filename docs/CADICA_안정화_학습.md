# CADICA-only 강한 안정화 학습 (RT-DETR)

> Group 2(현경). ARCADE / LCA·RCA가 약할 때, **통합 CADICA + imgsz 640**을
> 유지한 채 학습만 안정화해서 baseline(`train_v3_amp_on`)을 넘길 수 있는지 본다.

## 0. 왜 mAP가 “최신 모델인데도” 낮아 보이나

COCO/논문 데모의 RT-DETR·YOLO 숫자는 **대규모 자연 영상** 기준이다.
CADICA는 그와 조건이 다르다.

| 요인 | CADICA 현실 |
|------|-------------|
| 규모 | 환자 ~42명, 탐지 bbox 프레임 ~3.7k, patient split |
| 도메인 | X-ray angio, 저대비, 카테터/뼈/배경 노이즈 |
| 타깃 | stenosis bbox — 경계 모호, 등급 정의 애매 |
| 지표 | 팀 바 mAP50 ≈ 0.20–0.30이 현실적. COCO 0.5+를 기대하면 안 됨 |

**해석:** “RT-DETR이 CADICA에 안 맞는다”기보다,
**작은·어려운 의료 데이터에서 탐지 mAP가 원래 낮게 나오는 경우가 많다.**
같은 `common_split`에서 YOLO(5조)도 비슷한 대역이면 → **데이터/과제 난이도**.
YOLO만 훨씬 높고 RT-DETR만 깨지면 → **모델/하이퍼 쪽**을 의심.

현재 기준점 (`train_v3_amp_on`, imgsz=640):

| Split | P | R | mAP50 | mAP50-95 |
|-------|------|------|-------|----------|
| Val   | 0.229 | 0.219 | 0.117 | 0.030 |
| Test  | 0.592 | 0.289 | **0.308** | 0.150 |

이미 버린 것: imgsz 768, LCA/RCA 분리. ARCADE는 별도 실험 종료 후 판정.

## 1. ARCADE 병합 결과 (참고, 이 실험의 출발점)

| | Val mAP50 | Test mAP50 | Test P | Test R |
|--|----------:|-----------:|-------:|-------:|
| baseline | 0.117 | 0.308 | 0.592 | 0.289 |
| CADICA+ARCADE | 0.138 | 0.315 | 0.755 | 0.293 |

Test mAP50 +0.007 → **노이즈 수준, 뚜렷한 승리 아님.** ARCADE는 폐기까진
아니지만 메인으로 못 박지 않는다. 다음 카드가 이 **CADICA-only 강한
안정화**다.

## 2. “강한 안정화”가 baseline과 다른 점 (Stage 1)

한 번에 여러 노브를 세게 조인다 (데이터는 **CADICA-only** 고정, ARCADE 없음).

| 항목 | baseline | `train_stability_v1` (강) |
|------|----------|---------------------------|
| lr0 | 0.0005 | **0.00015** |
| cos_lr | False | **True** |
| warmup_epochs | 3 | **6** |
| weight_decay | 0.0005 | **0.0001** |
| dropout (decoder) | 0 | **0.1** |
| mosaic | 1.0 | **0.5** |
| close_mosaic | 10 | **20** (마지막 20 epoch mosaic off) |
| mixup / copy_paste | 0 | 0 유지 (의료 bbox에 공격적 aug 비권장) |
| degrees / shear | 0 | 0 유지 |
| scale / translate | 기본 | **scale=0.3, translate=0.05** (약하게) |
| hsv_s / hsv_v | 0.7 / 0.4 | **0.4 / 0.3** |
| patience | 20 | **35** |
| epochs | 50 | **100** |
| imgsz / batch / AdamW / amp | 640 / 8 / AdamW / True | 동일 |

의도: lr를 baseline의 **1/3 이하**로 낮추고 cosine으로 스케줄해서 진동을
줄인다. dropout으로 과적합/변동을 추가로 누른다. mosaic/색/기하를 줄여
**의료 프레임의 기하·대비를 과도하게 깨지 않게** 한다. epochs·patience를
크게 늘려 lr가 낮아진 만큼 충분히 수렴할 시간을 준다.

## 3. Stage 2 — 필요하면 콜드 파인튜닝 (`finetune_stage2`)

Stage 1 결과가 baseline에 근접하거나 이겼지만 val이 여전히 흔들리면,
`train_stability_v1/weights/best.pt`에서 이어서 **lr0=0.00003**,
**aug 거의 0**, epochs=40으로 짧게 미세조정한다. "거칠게 학습 → 정밀
튜닝" 2단계 구조.

## 4. Colab 실행

데이터: `/content/cadica_yolo/data.yaml` (통합 CADICA, ARCADE 없음)

```python
from train_eval_rtdetr_stability import train, eval_and_compare, finetune_stage2

train()
eval_and_compare("train_stability_v1", "stability_v1")

# Stage 1이 애매하면 이어서:
# finetune_stage2()
# eval_and_compare("train_stability_v1_ft", "stability_v1_ft")
```

## 5. 판정

| 결과 | 다음 |
|------|------|
| test mAP50 ≥ baseline 0.308 (확실히) | **안정화 채택**, 최종 후보 |
| baseline과 비슷(±0.02)하지만 val 진동 여전 | **Stage 2 파인튜닝** 실행 |
| Stage 2로도 개선 없음 | baseline 유지, **YOLO 공정 비교**로 이동 |
| 둘 다 확 하락 | lr를 더 낮추기(0.0001 이하) 대신, 모델 교체 검토 |

## 6. 다른 모델을 돌려봐야 하나?

**지금은 “모델 탓”으로 단정하지 말 것.** 순서:

1. ARCADE 결과 = 보류 (기록만, 메인 아님)
2. 이 **강한 안정화 Stage 1 → 필요시 Stage 2**
3. 같은 split으로 **YOLO(5조)와 표 비교** — 둘 다 ~0.2–0.3이면 CADICA 난이도 스토리로 발표 가능
4. 그다음에야 후보: YOLOv8/11-m (본인 재현), RT-DETR-m(더 작음), 또는 P2는 **박스 크기 분포 확인 후** YOLO 쪽에만

바꾸지 말 것: `common_split`, imgsz=640 기본, binary lesion, val/test = CADICA only.
