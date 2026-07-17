# E01 — ADSD Pretraining (Faster R-CNN ResNet50-FPN)

> 상태: 완료  
> 관련: `EXPERIMENT_LOG.md` E1, Colab Cell 31~39  
> 목적: CADICA fine-tuning용 ADSD pretrained `best.pth` 확보 (ADSD SOTA가 목표 아님)

---

## 1. 목적

- COCO pretrained Faster R-CNN을 ADSD patient-wise train/val로 pretrain
- Best checkpoint는 **val mAP50** 기준
- ADSD 자체 성능 극대화보다 **전이용 가중치** 확보가 목표

---

## 2. 데이터셋

| Split | 이미지 수 | 출처 |
|---|---|---|
| train | 5,829 | `adsd_patient_split.csv` |
| val | 1,261 | 동일 |
| test | 1,235 | **이 단계에서 사용 안 함** |

- Dataset: `ADSDDataset`
- Transforms: `ToTensor` only (augmentation 없음)
- Negative 이미지: **0장** (데이터셋 자체의 특성)

---

## 3. 모델 / Init

| 항목 | 값 |
|---|---|
| Detector | Faster R-CNN |
| Backbone | ResNet50-FPN |
| Init | COCO (`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`) |
| Head | `FastRCNNPredictor(num_classes=2)` |
| Freeze | 없음 (backbone 전체 학습). BN은 torchvision `FrozenBatchNorm2d`만 고정 |

내부 transform (변경 안 함):
```
Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
Resize(min_size=(800,), max_size=1333)
```

---

## 4. 하이퍼파라미터

| 항목 | 값 |
|---|---|
| batch_size | 4 |
| optimizer | SGD |
| lr | 0.005 |
| momentum | 0.9 |
| weight_decay | 0.0005 |
| scheduler | StepLR(step_size=10, gamma=0.1) |
| epochs | 50 |
| EarlyStopping | patience=10, criterion=**val mAP50** |
| Best criterion | **val mAP50** |
| seed | 42 |
| score_thresh (P/R) | 0.5 |
| iou_thresh (P/R) | 0.5 |

---

## 5. 학습 결과

| 항목 | 값 |
|---|---|
| Best val mAP50 | **0.1124** |
| EarlyStopping | epoch **13**에서 종료 |
| 추정 best epoch | ≈ **3** (13 − 10) |
| LR at end | 0.0005 (epoch 10에서 1회 감소) |

### 종료 시점 (epoch 13, best가 아님)
| Metric | Value |
|---|---|
| train_loss | 0.0529 |
| val_loss | 0.3474 |
| val_mAP50 | 0.0600 |
| val_mAP50-95 | 0.0240 |
| Precision | 0.4018 |
| Recall | 0.0714 |

Resume 로그 예:
```
[resume] last.pth -> resumed from epoch 13
best_map50 = 0.1124
epochs_no_improve = 10
```

---

## 6. 산출물 경로

```
{DRIVE_ROOT}/checkpoints/adsd/best.pth
{DRIVE_ROOT}/checkpoints/adsd/last.pth
{DRIVE_ROOT}/logs/adsd/training_log.csv
{DRIVE_ROOT}/logs/adsd/events.out.tfevents.*
```

`DRIVE_ROOT` =
`/content/drive/MyDrive/팀프로젝트/DL 프로젝트/ADSD+CADICA`

---

## 7. 결과 해석

- best가 매우 이른 epoch에서 나온 뒤 개선이 멈춤 → lr이 데이터 규모 대비 높았거나 val(patient 10명) 노이즈 가능
- val mAP50 0.11은 **낮은 pretraining 품질** → CADICA 전이 기대치 제한적
- 다만 프로젝트 방침상 **ADSD 추가 튜닝은 하지 않음** (pretraining only)

### 한계 후보
1. ADSD 100% positive (negative 학습 불가)
2. Backbone 전체 unfreeze + 소규모 데이터
3. 이른 EarlyStopping / 낮은 best mAP

---

## 8. 운영 메모

- Colab 런타임이 끊기면 Drive의 `last.pth`로 resume (`fit(resume=True)`)
- `/content` 데이터는 재압축 해제 필요
- 최종 전이에는 **반드시 `best.pth`**를 쓸 것 (epoch 13 last 모델이 아님)

---

## 9. 다음

- `E02_cadica_finetuning.md` — ADSD `best.pth`로 CADICA fine-tune
- ADSD 자체 HP 탐색은 TODO P3(보류)
