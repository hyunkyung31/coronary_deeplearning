# E02 — CADICA Fine-tuning (ADSD Pretrained Init)

> 상태: 파이프라인 준비 완료 / **최종 수치 결과 대기**  
> 관련: `EXPERIMENT_LOG.md` E2, Colab Cell 40~44  
> 목적: ADSD `best.pth`로 CADICA를 fine-tune하고 val/test에서 전이 효과를 측정

---

## 1. 목적

```
ADSD best.pth
  → CADICA Fine-tuning
  → CADICA Validation / Test Evaluation
  → Test 예측 시각화 저장
```

- Detector/Backbone/Transforms는 E01과 **동일**
- 의도된 차이는 **초기 가중치 = ADSD pretrained**
- ImageNet/COCO-only baseline(E03 성격)은 **일정 때문에 보류** — 나중에 같은 HP로 비교

---

## 2. 데이터셋

| Split | 이미지 수 |
|---|---|
| train | 13,316 |
| val | 1,396 |
| test | 1,087 |

- Dataset: `CADICADataset` + `common_split.csv` (patient-wise)
- Negative 포함 (옵션 1): gt 없는 프레임 = boxes=0
- Transforms: `ToTensor` only (augmentation 없음)
- 등급: `p0_20`~`p100` → 전부 Stenosis(label=1)

---

## 3. 모델 / Init (안전장치)

| 항목 | 값 |
|---|---|
| Architecture | Faster R-CNN ResNet50-FPN, `num_classes=2` (변경 금지) |
| Init | `checkpoints/adsd/best.pth` **명시적 로드** |
| `pretrained_backbone` | `False` (ADSD ckpt가 전체 덮어씀; COCO 재다운로드 불필요) |
| Optimizer/Scheduler | **새로 생성** (ADSD momentum/lr schedule 미이관) |

### 반드시 지켜야 할 로직
1. Stage1 종료 시점 메모리 `model`은 last(epoch 13)일 수 있음 → **best.pth 재로드**
2. `load_checkpoint(..., optimizer=None, scheduler=None)` — 모델만 로드
3. `CADICA_CKPT_DIR`에 기존 `last.pth`가 있으면 `fit(resume=True)`가 ADSD init을 덮어씀 → **사전 점검**
4. 최종 평가 전 **CADICA `best.pth`를 다시 로드** (ES로 멈춘 last ≠ best 가능)

---

## 4. 하이퍼파라미터

| 항목 | 값 | E01과 비교 |
|---|---|---|
| batch_size | 4 | 동일 |
| optimizer | SGD | 동일 |
| lr | **0.001** | E01은 0.005 (fine-tune용으로 낮춤) |
| momentum | 0.9 | 동일 |
| weight_decay | 0.0005 | 동일 |
| scheduler | StepLR(**step_size=8**, gamma=0.1) | E01은 step_size=10 |
| epochs | **20** | E01은 50 |
| EarlyStopping patience | **5** | E01은 10 |
| Best criterion | val mAP50 | 동일 |
| Transforms | ToTensor only | 동일 |
| Augmentation | 없음 | 동일 |

> Fair comparison을 위해 이후 ImageNet baseline을 돌릴 때도 **이 표의 CADICA HP를 그대로** 쓸 것.

---

## 5. 실행 셀 순서

| Cell | 내용 |
|---|---|
| 40 | `CADICA_CKPT_DIR` 기존 ckpt 점검 |
| 41 | ADSD best 로드 + CADICA optimizer/scheduler 생성 |
| 42 | `fit(... epochs=20, patience=5)` |
| 43 | CADICA best 재로드 → val/test 평가 + `final_evaluation.json` |
| 44 | Test 시각화 저장 (`logs/cadica/visualizations/`) |

---

## 6. 평가 지표 (필수 보고)

Validation + Test 각각:

| Metric | 비고 |
|---|---|
| mAP50 | Best 선정과 동일 계열 |
| mAP50-95 | torchmetrics `map` |
| Precision | score≥0.5, IoU≥0.5 |
| Recall | 동일 |
| Validation loss | 모니터링 전용 (best 기준 아님) |

평가 함수: `evaluate_detector` / `validate_loss` (E01과 **동일 설정**)

---

## 7. 산출물 경로 (예정)

```
{DRIVE_ROOT}/checkpoints/cadica/best.pth
{DRIVE_ROOT}/checkpoints/cadica/last.pth
{DRIVE_ROOT}/logs/cadica/training_log.csv
{DRIVE_ROOT}/logs/cadica/final_evaluation.json
{DRIVE_ROOT}/logs/cadica/visualizations/test_pred_*.png
```

---

## 8. 결과 (작성 시점: 미기입)

> 학습/평가가 끝나면 아래 표를 채울 것.

### Validation
| Metric | Value |
|---|---|
| mAP50 | |
| mAP50-95 | |
| Precision | |
| Recall | |
| val_loss | |
| best_epoch | |

### Test
| Metric | Value |
|---|---|
| mAP50 | |
| mAP50-95 | |
| Precision | |
| Recall | |

### 해석 / 다음
- (결과 기입 후 작성)
- 도메인 불일치 분석: `E03_domain_analysis.md`
- 전처리 실험: `E04_crop_experiment.md`

---

## 9. 하지 않을 것 (이 실험에서)

- Augmentation 추가
- Architecture / anchor / NMS / min_size 변경
- 새 split 생성
- Synthetic negative
- Grid Search / Optuna
- ImageNet baseline 동시 실행 (보류)
