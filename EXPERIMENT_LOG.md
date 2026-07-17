# EXPERIMENT_LOG.md

> Single Source of Truth — 실험 이력 (시간순)  
> 마지막 업데이트: 2026-07-17  
> 새 Agent는 이 파일만 읽어도 어떤 실험을 했고 무엇을 배웠는지 파악할 수 있어야 한다.

---

## 실험 인덱스

상세 문서는 `docs/experiments/`에 분리되어 있다.

| ID | 상세 파일 | 이름 | 상태 | 핵심 결과 |
|---|---|---|---|---|
| E0 | [E00_dataset.md](docs/experiments/E00_dataset.md) | 데이터셋 검사 / 프로토콜 확정 | 완료 | 구조·split·포맷 검증, 등급 병합 확정 |
| E1 | [E01_adsd_pretraining.md](docs/experiments/E01_adsd_pretraining.md) | ADSD Pretraining (Faster R-CNN) | 완료 | best val mAP50 = 0.1124, ES@epoch 13 |
| E2 | [E02_cadica_finetuning.md](docs/experiments/E02_cadica_finetuning.md) | CADICA Fine-tuning (ADSD init) | 파이프라인 준비됨 / 결과 대기 | — |
| E3 | [E03_domain_analysis.md](docs/experiments/E03_domain_analysis.md) | 도메인 분석 (+ ImageNet baseline은 보류) | 분석 완료 / baseline 보류 | 도메인 불일치 정량화 |
| E4 | [E04_crop_experiment.md](docs/experiments/E04_crop_experiment.md) | ADSD 전처리(상대 면적 정렬) 후 재전이 | **설계만** (미구현) | 크롭 전략 제안, 승인 대기 |
| A1 | (E03에 포함) | 파이프라인 감사 | 완료 | 코드 버그 없음, 설계상 특징 확인 |
| A2 | (E03에 포함) | ADSD vs CADICA 통계 비교 | 완료 | 도메인 불일치 정량화 |

---

## E0. 데이터셋 검사 및 프로토콜 확정

### 목적
학습 코드 작성 전에 ADSD/CADICA 실물 구조와 split CSV 정합성을 검증하고, 라벨 체계를 확정한다.

### 데이터셋
- ADSD: `/content/adsd_dataset/dataset/` (`.bmp` + Pascal VOC `.xml`)
- CADICA: `/content/cadica_dataset/selectedVideos/` (`input/*.png` + `groundtruth/*.txt`)
- Split: `adsd_patient_split.csv` (64 patients), `common_split.csv` (42 patients, 334 videos)

### 주요 발견
1. ADSD 어노테이션 = Pascal VOC XML (`xmin,ymin,xmax,ymax`, class=`Stenosis`)
2. CADICA 어노테이션 = `x y w h category` (category = `p0_20`~`p100` 협착 등급)
3. CADICA lesion 비디오라도 **프레임 단위로 groundtruth가 없는 경우가 다수** (키프레임 11,791 vs gt 파일 3,685)
4. `train_labels.csv`/`test_labels.csv`는 patient-wise가 아님(63/64 patient 누수) — **사용하지 않음**. 공식 split CSV만 사용.
5. `video_test/`, `video_val/`, `timing_*.xlsx`는 이전 벤치마크 잔존물 — 학습 미사용

### 확정 결정
- CADICA 등급 → 단일 `Stenosis` (옵션 A)
- CADICA Dataset: **옵션 1** — 모든 키프레임 포함, 박스 없으면 negative
- `metadata.xlsx` 파이프라인 미사용
- `num_classes=2` (background + Stenosis)

### 해석
이후 모든 Dataset/학습/평가는 이 결정을 전제로 한다. 프로토콜 변경은 명시적 승인 필요.

---

## E1. ADSD Pretraining (Faster R-CNN ResNet50-FPN)

### 목적
CADICA fine-tuning의 초기 가중치로 쓸 ADSD pretrained checkpoint를 만든다.  
**ADSD 자체 SOTA를 목표로 하지 않음** (pretraining only).

### 데이터셋
- ADSD train/val (patient-wise, `adsd_patient_split.csv`)
- train: 5,829 / val: 1,261 / test: 1,235 (test는 이 단계에서 사용 안 함)

### 모델
- Faster R-CNN ResNet50-FPN
- Init: COCO pretrained (`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`)
- Head: `FastRCNNPredictor(num_classes=2)`
- Backbone: **전체 unfreeze** (명시적 freeze 코드 없음)
- BN: torchvision `FrozenBatchNorm2d` (영구 고정, 우리가 건드린 것 아님)

### 하이퍼파라미터
| 항목 | 값 |
|---|---|
| batch_size | 4 |
| optimizer | SGD |
| lr | 0.005 |
| momentum | 0.9 |
| weight_decay | 0.0005 |
| scheduler | StepLR(step_size=10, gamma=0.1) |
| epochs | 50 |
| EarlyStopping | patience=10, criterion=val mAP50 |
| Best criterion | val mAP50 |
| Transforms | ToTensor only (no augmentation) |
| Internal resize | min_size=800, max_size=1333 (GeneralizedRCNNTransform) |
| seed | 42 |

### 학습 결과
| 항목 | 값 |
|---|---|
| Best val mAP50 | **0.1124** |
| EarlyStopping | epoch 13에서 종료 |
| 추정 best epoch | ≈3 (13 − 10) |
| Epoch 13 (종료 시점, best 아님) train_loss | 0.0529 |
| Epoch 13 val_loss | 0.3474 |
| Epoch 13 val_mAP50 | 0.0600 |
| Epoch 13 val_mAP50-95 | 0.0240 |
| Epoch 13 Precision | 0.4018 |
| Epoch 13 Recall | 0.0714 |
| LR schedule | epoch 10에서 0.005 → 0.0005 |

### 산출물
- `{DRIVE_ROOT}/checkpoints/adsd/best.pth`
- `{DRIVE_ROOT}/checkpoints/adsd/last.pth`
- `{DRIVE_ROOT}/logs/adsd/training_log.csv`
- TensorBoard 로그

### 결과 해석
- best가 매우 이른 epoch(≈3)에서 나온 뒤 10 epoch 동안 개선 없음 → lr이 데이터 규모 대비 다소 높았거나 val set(patient 10명) 노이즈 가능
- ADSD val mAP50 0.11 수준은 **낮은 pretraining 품질** → CADICA 전이 기대치도 제한적
- 다만 본 실험의 목적은 ADSD SOTA가 아니므로, **추가 ADSD 튜닝은 하지 않기로 결정**

### 실패/한계 (ADSD 자체 성능이 낮은 이유 후보)
1. 100% positive 데이터 (negative 0장)
2. Backbone 전체 학습 + 소규모 데이터
3. 이른 EarlyStopping / 낮은 best mAP

### 다음 개선 방향 (당시)
→ CADICA Fine-tuning으로 바로 이동 (ADSD 추가 튜닝 금지)

---

## E2. CADICA Fine-tuning (ADSD pretrained init) — 파이프라인 준비

### 목적
ADSD `best.pth`를 초기화로 사용하여 CADICA를 fine-tune하고, val/test에서 전이 효과를 측정한다.

### 설계상 안전장치 (구현에 반영됨)
1. 메모리의 마지막 `model`이 아니라 **ADSD `best.pth`를 명시적으로 로드**
2. Optimizer/scheduler는 **새로 생성** (ADSD momentum/lr schedule 미이관)
3. `CADICA_CKPT_DIR`에 기존 `last.pth`가 있으면 resume이 ADSD 로드를 덮어쓸 수 있음 → **사전 점검 셀**
4. 최종 평가 전 **CADICA `best.pth`를 다시 로드** 후 val/test 평가

### 하이퍼파라미터 (확정)
| 항목 | 값 |
|---|---|
| Init | ADSD `best.pth` (pretrained_backbone=False로 구조만 만들고 덮어쓰기) |
| batch_size | 4 |
| optimizer | SGD |
| lr | 0.001 |
| momentum | 0.9 |
| weight_decay | 0.0005 |
| scheduler | StepLR(step_size=8, gamma=0.1) |
| epochs | 20 |
| EarlyStopping | patience=5, criterion=val mAP50 |
| Best criterion | val mAP50 |
| Transforms | ADSD와 동일 (ToTensor only) |
| Architecture | ADSD와 동일 (변경 금지) |

### 평가 (설계)
- Validation + Test 자동 평가
- 지표: mAP50, mAP50-95, Precision, Recall, Validation loss
- Test 예측 시각화 저장 (GT=초록, Pred=빨강, score≥0.5)

### 상태
- Colab Cell 40~44로 코드 제공됨
- **최종 수치 결과는 이 문서 작성 시점에 아직 확정 기입되지 않음** → 실행 완료 후 이 섹션에 결과 테이블을 추가할 것

### Fair comparison에 대한 결정
- ImageNet/COCO-only baseline을 같은 HP로 돌리는 것이 연구적으로 가장 clean
- 그러나 **일정/연산 시간 제약으로 baseline은 보류** (E3)
- 우선 ADSD→CADICA만 완성

---

## E3. ImageNet/COCO-only CADICA Baseline — 보류

### 목적
초기화만 다르게 하고 나머지 HP를 E2와 완전히 동일하게 맞춰, ADSD pretraining 효과를 isolate한다.

### 설계
- Init: `pretrained_backbone=True` (COCO), ADSD 로드 없음
- HP: E2와 동일
- 저장 경로: `checkpoints/cadica_baseline_imagenet/`, `logs/cadica_baseline_imagenet/`

### 상태
**실행하지 않음.** E2 결과가 유망하고 시간이 남으면 동일 HP로 실행.

---

## A1. 파이프라인 감사 (코드 수정 없음)

### 점검 항목과 결론
| 항목 | 결과 |
|---|---|
| Trainable layers | Backbone 전체 unfreeze. FrozenBatchNorm만 비학습(torchvision 기본) |
| Optimizer | SGD, 단일 param group, ADSD lr=0.005 / CADICA lr=0.001 |
| Scheduler | epoch 끝 1회 step, resume 시 state 복원 OK |
| EarlyStopping | val mAP50, patience 로직 OK |
| Evaluation | ADSD/CADICA 동일 (`score_thresh=0.5`, `iou_thresh=0.5`, torchmetrics mAP) |
| Dataset negatives | CADICA OK. **ADSD는 원천적으로 negative 0장** |
| 코드 버그 | **없음** |

### 의심스러운 설계상 특징 (버그 아님)
1. Backbone 전체 학습
2. ADSD 100% positive

---

## A2. ADSD vs CADICA 통계 비교 (코드 수정 없음)

### 데이터 규모
| | ADSD | CADICA |
|---|---|---|
| 이미지 수 | 8,325 | 15,799 |
| 박스 수 | 8,326 | 5,835 |

### Positive / Negative
| | ADSD | CADICA |
|---|---|---|
| Positive | 8,325 (100%) | 3,685 (23.3%) |
| Negative | 0 (0%) | 12,114 (76.7%) |

### 박스 절대 크기 (px)
| | ADSD mean | CADICA mean |
|---|---|---|
| width | 46.18 | 46.04 |
| height | 39.49 | 44.88 |

→ **절대 픽셀 크기는 거의 동일**

### 상대 박스 면적 (%)
| | ADSD | CADICA |
|---|---|---|
| median | 0.285% | 0.595% |
| mean | 0.336% | 0.823% |

→ CADICA가 상대적으로 **약 2배 큼**.  
원인: ADSD 이미지가 더 큼(주로 800/1000) + 절대 박스 크기는 비슷 → 상대 면적만 작아짐.  
초기 가설 "CADICA 병변이 더 작다"는 **기각** (방향이 반대).

### 해상도
- ADSD: 512(1603), 608(343), 800(5332), 1000(1047)
- CADICA: 512×512 고정 (15,799)

### 이미지당 평균 박스
- ADSD 전체: 1.000
- CADICA 전체(neg 포함): 0.369
- CADICA positive만: 1.583

### 해석 (전이 실패 가설 종합)
1. Pos/Neg 비율 정반대 (가장 강력)
2. 상대 스케일 불일치 (절대 크기는 비슷)
3. Backbone 전체 학습 + 소규모 ADSD
4. ADSD pretraining mAP 자체 낮음
5. CADICA는 positive 내 멀티박스 비율이 더 높음

---

## E4. ADSD 전처리로 CADICA와 유사하게 만들기 — 설계만

### 목적
코드/아키텍처를 바꾸지 않고, **진짜 ADSD 데이터만**으로 상대 분포를 CADICA에 가깝게 만들어 전이력을 높일 수 있는지 검토.

### 핵심 발견 (설계 시)
- 모델 내부가 이미 `min_size=800`으로 리사이즈하므로 **원본 파일 리사이즈는 효과 없음**
- 상대 면적(%)은 균일 리사이즈에 **불변** → 리사이즈로는 상대 면적을 못 맞춤
- "진짜 데이터만 / synthetic negative 금지" 조건에서 통제 가능한 레버는 **상대 면적뿐** → **병변 중심 크롭**

### 제안 전략 (미구현)
- 각 ADSD 이미지에서 bbox 중심 정사각형 crop
- 목표 상대 면적 ≈ CADICA median (0.6%)
- CADICA에는 적용하지 않음 (ADSD 전용)
- split/라벨 체계 불변
- 예상 잔여 이미지 ≈ 거의 전량 (필터링 손실 미미)

### Pros
- 실측된 유효 차이를 정확히 타겟
- 진짜 픽셀만 사용
- 데이터 손실 적음

### Cons / Risks
- 해부학적 맥락 손실(과도한 zoom-in)
- Pos/Neg·밀도 문제는 해결 못함
- 새 실험이므로 원본 ADSD-pretrain과 비교 필요 → 추가 GPU 시간

### 상태
**사용자 승인 대기. 코드 미작성.**

---

## 이전 RT-DETR / YOLO 실험 (배경, 본 파이프라인 밖)

본 Faster R-CNN 파이프라인으로 전환하기 **이전**에 수행된 실험들 (상세 수치는 이 저장소/대화에 완전 기입되지 않음):

- RT-DETR extensive tuning
- YOLO comparison
- Image size experiments
- LCA/RCA split
- ADSD evaluation
- Swin-T classification

**결과**: Patient-wise 일반화가 나쁨 → **Faster R-CNN으로 교체** 결정.

참고 잔존물 (학습 미사용): `video_test/`, `video_val/`, `timing_*.xlsx`, 레거시 `train_labels.csv`/`test_labels.csv`

---

## 로그 작성 규칙 (앞으로)

새 실험마다 아래 템플릿으로 섹션을 추가한다.

```markdown
## E#. 실험 이름

### 목적
### 데이터셋
### 모델 / Init
### 하이퍼파라미터
### 결과 (표)
### 산출물 경로
### 결과 해석
### 실패 원인 / 한계
### 다음 개선 방향
```
