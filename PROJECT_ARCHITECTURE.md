# PROJECT_ARCHITECTURE.md

> Single Source of Truth — 코드/데이터/파이프라인 구조  
> 마지막 업데이트: 2026-07-17  
> 실행 환경: Google Colab (노트북 셀 기반). 모듈 파일로 분리되지 않은 Colab 셀 코드가 본체다.

---

## 1. 현재 프로젝트 구조

### 1.1 Google Drive (영구 저장)

```
/content/drive/MyDrive/팀프로젝트/DL 프로젝트/ADSD+CADICA/
├── dataset/
│   ├── Stenosis detection.zip      # ADSD 원본 zip
│   └── CADICA.zip                  # CADICA 원본 zip
├── split_csv/
│   ├── adsd_patient_split.csv      # ADSD patient-wise split (64명)
│   └── common_split.csv            # CADICA patient-wise split (42명, 334 video)
├── checkpoints/
│   ├── adsd/
│   │   ├── best.pth                # Stage1 best (val mAP50)
│   │   └── last.pth                # Stage1 resume용
│   └── cadica/
│       ├── best.pth                # Stage2 best (예정)
│       └── last.pth
└── logs/
    ├── adsd/
    │   ├── training_log.csv
    │   ├── events.out.tfevents.*   # TensorBoard
    │   └── (final_evaluation.json 등)
    └── cadica/
        ├── training_log.csv
        ├── final_evaluation.json
        └── visualizations/         # test 예측 시각화
```

> ImageNet baseline을 나중에 돌릴 경우 (보류):  
> `checkpoints/cadica_baseline_imagenet/`, `logs/cadica_baseline_imagenet/`

### 1.2 Colab 로컬 (`/content`, 런타임 리셋 시 삭제)

```
/content/
├── adsd_dataset/
│   ├── dataset/                    # ★ 학습 대상: *.bmp + *.xml
│   ├── video_test/                 # 미사용 (이전 벤치마크 .avi)
│   ├── video_val/                  # 미사용
│   ├── train_labels.csv            # 미사용 (patient-wise 아님)
│   ├── test_labels.csv             # 미사용
│   └── timing_*.xlsx               # 미사용
└── cadica_dataset/
    ├── selectedVideos/             # ★ 학습 대상 (common_split.csv와 대응)
    │   └── pX/vY/
    │       ├── input/*.png
    │       ├── groundtruth/*.txt   # lesion video만
    │       └── pX_vY_selectedFrames.txt
    ├── nonselectedVideos/          # 미사용
    ├── metadata.xlsx               # 미사용 (임상 메타)
    └── readme.txt
```

### 1.3 GitHub 저장소 (참고)

현재 원격 저장소에는 CADICA README / 레거시 라벨 CSV / 일부 노트북이 있을 수 있다.  
**실제 실험 본체는 Colab + Drive 경로**이며, 이 문서 4개가 SSOT다.

---

## 2. Dataset 구조

### 2.1 ADSD

| 항목 | 내용 |
|---|---|
| 이미지 | `dataset/*.bmp` (그레이스케일 → Dataset에서 RGB 변환) |
| 어노테이션 | 동일 stem의 Pascal VOC `.xml` |
| patient_id | `filename.split('_')[:2]` → `"14_XXX"` (예: `14_024_2_0042.bmp` → `14_024`) |
| Split | `adsd_patient_split.csv`의 `patient_id`/`split` |
| 클래스 | `Stenosis` → label=1 |
| Negative | **없음** (모든 이미지에 ≥1 box) |
| 해상도 | 512 / 608 / 800 / 1000 (혼재) |

**클래스**: `ADSDDataset` (대략 Cell 17)

### 2.2 CADICA

| 항목 | 내용 |
|---|---|
| 이미지 | `selectedVideos/{patient_id}/{video_id}/input/*.png` |
| 어노테이션 | `groundtruth/{stem}.txt` — 한 줄: `x y w h category` |
| 좌표 변환 | `[x,y,w,h]` → `[x, y, x+w, y+h]` (xyxy) |
| 등급 | `p0_20`~`p100` → **전부 label=1**, 원본 문자열은 `raw_categories`에 보존 |
| Split | `common_split.csv`의 `(patient_id, video_id, split)` |
| Negative | gt 파일 없으면 박스 0개 샘플로 포함 (**옵션 1**) |
| 해상도 | 전부 512×512 |

**클래스**: `CADICADataset` (대략 Cell 18)

### 2.3 Split CSV 스키마

**adsd_patient_split.csv**
```
patient_id,n_images,split
14_002,53,train
```
- train 45 / val 10 / test 9 patients
- n_images 합계 8,325 (실측과 완전 일치)

**common_split.csv**
```
patient_id,video_id,label,view,n_frames,has_groundtruth,split
p1,v2,lesion,LCA,56,True,train
```
- train 33 / val 4 / test 5 patients
- 한 환자는 하나의 split에만 속함 (누수 0)

### 2.4 금지 사항
- `train_test_split()` / 새 split 생성 금지
- `train_labels.csv` / `test_labels.csv`를 patient-wise split으로 쓰지 말 것
- `nonselectedVideos`를 학습에 쓰지 말 것 (현재 프로토콜)

---

## 3. Detection Pipeline

```
[Drive zip] → extract → Dataset(split CSV) → Transforms(ToTensor)
    → DataLoader(collate_fn) → Faster R-CNN (ResNet50-FPN)
    → fit() [train → val_loss → mAP → last/best ckpt → EarlyStopping]
    → load best.pth → evaluate_detector(val/test) → 시각화 저장
```

### Stage 흐름
1. **ADSD Pretraining**: COCO init → ADSD train/val → `checkpoints/adsd/best.pth`
2. **Load ADSD best**: 새 모델 생성 → `load_checkpoint(best.pth)` (optimizer 미로드)
3. **CADICA Fine-tuning**: 새 SGD/StepLR → CADICA train/val → `checkpoints/cadica/best.pth`
4. **CADICA Eval**: CADICA `best.pth` 재로드 → val/test metrics + visualizations

---

## 4. 핵심 함수 / 클래스 역할

> Colab 셀 번호는 대화 중 부여한 번호다. 노트북을 재구성하면 번호가 달라질 수 있으니, **함수 이름**을 기준으로 찾을 것.

| 심볼 | 대략적 Cell | 역할 |
|---|---|---|
| `DRIVE_ROOT` 등 경로 상수 | Cell 2 | Drive/로컬 경로 SSOT |
| `extract_if_needed`, `flatten_single_nested_dirs` | Cell 3~5 | zip 해제 + 중첩 폴더 정규화 |
| `ADSDDataset` | Cell 17 | ADSD VOC Dataset |
| `CADICADataset` | Cell 18 | CADICA Dataset (옵션1 negative) |
| `Compose`, `ToTensor`, `get_transforms` | Cell 28 | 외부 transform (ToTensor만) |
| `seed_worker`, `make_generator` | Cell 23 | DataLoader 재현성 |
| `collate_fn` | Cell 24 | detection용 list batch |
| `build_dataloader` | Cell 25 | DataLoader 팩토리 |
| `build_datasets_and_loaders` | Cell 26/29 | train/val/test Dataset+Loader |
| `build_model` | Cell 31 | Faster R-CNN ResNet50-FPN, num_classes=2 |
| `smoke_test` | Cell 32 | 모델 입출력 점검 |
| `evaluate_detector`, `_match_boxes` | Cell 35 | mAP50 / mAP50-95 / P / R |
| `save_checkpoint`, `load_checkpoint` | Cell 36 | ckpt I/O (`weights_only=False`) |
| `train_one_epoch` | Cell 37 | 1 epoch 학습 |
| `validate_loss` | Cell 37 | val loss (모니터링 전용) |
| `fit` | Cell 38 | 전체 루프 + EarlyStopping + logging |
| ADSD training 실행 | Cell 39 | Stage1 실행 (epochs=50, patience=10) |
| CADICA 사전점검 | Cell 40 | `CADICA_CKPT_DIR` 기존 ckpt 확인 |
| ADSD best 로드 + CADICA opt | Cell 41 | Stage3+4 준비 |
| CADICA fine-tune 실행 | Cell 42 | epochs=20, patience=5 |
| CADICA 최종 평가 | Cell 43 | best 재로드 → val/test |
| Test 시각화 | Cell 44 | GT/Pred 박스 저장 |
| 통계 수집 | Cell 46~52 | ADSD vs CADICA 분석 (파이프라인 비수정) |

### `build_model(num_classes=2, pretrained_backbone=True)`
- `fasterrcnn_resnet50_fpn(weights=...)` 로드
- `box_predictor`만 2-class로 교체
- **backbone/RPN/anchor/NMS 등 아키텍처 파라미터를 바꾸지 말 것**

### `fit(..., early_stopping_patience=None)`
순서 (매 epoch):
1. `train_one_epoch`
2. `validate_loss` (best에 사용 안 함)
3. `evaluate_detector` → val mAP50 등
4. `scheduler.step()`
5. `last.pth` 저장 (`epochs_no_improve` 포함)
6. val mAP50 개선 시 `best.pth` 저장
7. `epochs_no_improve >= patience`면 break

### `evaluate_detector(model, loader, device, score_thresh=0.5, iou_thresh=0.5)`
- mAP: `torchmetrics.detection.MeanAveragePrecision` (전체 predictions, threshold 미적용)
- Precision/Recall: `score_thresh` 이상 예측만 IoU 매칭
- **ADSD/CADICA가 동일 함수·동일 기본값 사용** (평가 설정 분리 금지)

### `load_checkpoint(path, model, optimizer=None, scheduler=None)`
- CADICA 전이 시 **반드시 optimizer/scheduler를 넘기지 말 것** (모델 가중치만 로드)
- Resume 시에는 셋 다 전달

---

## 5. 하이퍼파라미터 기본값 (현재 확정)

| | ADSD Pretrain | CADICA Fine-tune |
|---|---|---|
| Init | COCO pretrained | ADSD `best.pth` |
| lr | 0.005 | 0.001 |
| momentum | 0.9 | 0.9 |
| weight_decay | 0.0005 | 0.0005 |
| scheduler | StepLR(10, 0.1) | StepLR(8, 0.1) |
| epochs | 50 | 20 |
| EarlyStopping patience | 10 | 5 |
| batch_size | 4 | 4 |
| Best metric | val mAP50 | val mAP50 |

---

## 6. 코드 수정 시 절대 건드리면 안 되는 부분

명시적 승인 없이 아래를 변경하지 말 것.

### 프로토콜 / 과학적 타당성
1. **Detector / Backbone**: Faster R-CNN ResNet50-FPN 고정
2. **Split 파일**: `adsd_patient_split.csv`, `common_split.csv`만 사용. 새 split 생성 금지
3. **클래스 체계**: 단일 Stenosis (`num_classes=2`). 등급 다중분류로 되돌리지 말 것
4. **CADICA negative 정책**: 옵션 1 (모든 키프레임 포함) — 승인 없이 positive-only로 바꾸지 말 것
5. **Best 기준**: val mAP50
6. **최종 평가 데이터**: CADICA test (및 보고용 val). ADSD test로 프로토콜을 바꾸지 말 것
7. **Augmentation**: 현재 없음. 특히 horizontal flip은 의학 검증 없이 금지
8. **실험 비교 시 HP**: baseline vs ADSD-init 비교를 할 때는 **초기화만 다르게**, 나머지 HP 동일

### 구현 안전장치 (버그 방지용 — 함부로 제거 금지)
9. CADICA fine-tune 시작 시 **ADSD `best.pth` 명시적 로드** (메모리 last 모델 재사용 금지)
10. 전이 시 **optimizer/scheduler 상태 미이관**
11. 최종 평가 전 **해당 stage의 `best.pth` 재로드**
12. `fit()` resume이 의도치 않게 이전 CADICA `last.pth`를 덮어쓰지 않도록 **ckpt 디렉토리 사전 점검**
13. Resume를 위해 `epochs_no_improve`를 checkpoint `extra`에 저장

### 데이터 경로 / 산출물
14. 체크포인트·로그는 **Google Drive**에 저장 (`/content`만 쓰면 런타임 끊김 시 소실)
15. ADSD 전용 전처리를 도입할 경우 **CADICA에는 적용하지 말 것**

---

## 7. 평가 임계값 (현재)

| 항목 | 값 | 비고 |
|---|---|---|
| mAP score threshold | 없음 (전체 pred) | torchmetrics 표준 |
| Precision/Recall score_thresh | 0.5 | `evaluate_detector` 기본값 |
| Precision/Recall iou_thresh | 0.5 | 동일 |
| 모델 내부 score_thresh | 0.05 | torchvision 기본 |
| box_nms_thresh | 0.5 | torchvision 기본 |
| rpn_nms_thresh | 0.7 | torchvision 기본 |
| box_detections_per_img | 100 | torchvision 기본 |

아키텍처 변경 금지 지침에 따라 NMS/anchor 등도 **기본값 유지**.

---

## 8. 런타임이 끊겼을 때

1. Colab **런타임 → 모두 실행** (또는 Drive 마운트부터 파이프라인 정의 셀까지 재실행)
2. `/content` 데이터는 다시 압축 해제됨 (`extract_if_needed`가 자동 처리)
3. `fit(resume=True)`가 Drive의 `last.pth`를 보고 이어서 학습
4. 출력에 `[resume] ... epoch N부터 재개`가 보이는지 확인

---

## 9. 관련 문서

- `PROJECT_STATUS.md` — 현재 상태 / 가설
- `EXPERIMENT_LOG.md` — 실험 이력
- `TODO.md` — 할 일
