# PROJECT_STATUS.md

> Single Source of Truth — 프로젝트 현재 상태  
> 마지막 업데이트: 2026-07-17  
> 플랫폼: Google Colab + Google Drive  
> 목표: ADSD Pretraining → CADICA Fine-tuning → CADICA Test Evaluation (Faster R-CNN ResNet50-FPN)

---

## 1. 현재 프로젝트 진행 상황 (한 줄 요약)

**ADSD Pretraining은 완료**되었고(best val mAP50=0.1124, EarlyStopping@epoch 13),  
**CADICA Fine-tuning 파이프라인은 설계·코드까지 준비되었으나**,  
실험 결과 해석 과정에서 **"코드 버그가 아니라 ADSD↔CADICA 도메인 불일치"**가 핵심 문제로 확인되었다.  
현재는 파이프라인을 수정하지 않고, **ADSD를 CADICA와 더 비슷하게 만드는 전처리 전략을 설계 단계**에서 검토 중이다.

---

## 2. 완료된 작업

### Step 0 — Colab 환경 준비
- Google Drive 마운트
- ADSD zip / CADICA zip 압축 해제 (`/content/adsd_dataset`, `/content/cadica_dataset`)
- 중첩 폴더 정규화 (`flatten_single_nested_dirs`)
- Drive 경로 확정:
  ```
  /content/drive/MyDrive/팀프로젝트/DL 프로젝트/ADSD+CADICA
  ├── dataset/
  │   ├── Stenosis detection.zip
  │   └── CADICA.zip
  ├── split_csv/
  │   ├── adsd_patient_split.csv
  │   └── common_split.csv
  ├── checkpoints/{adsd, cadica}/
  └── logs/{adsd, cadica}/
  ```

### Step 1 — 데이터셋 정밀 검사
- ADSD: Pascal VOC XML + `.bmp`, 8,325장, 1:1 매칭, `adsd_patient_split.csv` 64명 완전 일치
- CADICA: `[x,y,w,h,category]` txt + `.png`, `common_split.csv` 42명/334 video 완전 일치
- CADICA 협착 등급(`p0_20`~`p100`) → **단일 `Stenosis` 클래스로 병합** 확정
- `metadata.xlsx`는 파이프라인에 미사용 확정
- CADICA lesion 비디오라도 프레임 단위로 groundtruth가 없을 수 있음 확인 (옵션 1: negative 포함)

### Step 2 — Dataset 클래스
- `ADSDDataset`, `CADICADataset` 구현
- 시각화 검증 통과 (ADSD 10장 / CADICA 10장)
- 샘플 수: ADSD train 5,829 / CADICA train 13,316

### Step 3 — DataLoader
- `collate_fn` (list 묶기), `seed_worker`, `make_generator`
- `BATCH_SIZE=4`, `NUM_WORKERS=2`
- split별 샘플 수 재검증 완료

### Step 4 — Transforms
- **ToTensor만** (augmentation 없음)
- Resize/Normalize는 `GeneralizedRCNNTransform`에 위임
- horizontal flip 제외 (LCA/RCA 해부학 방향성 우려, 의학 자문 없이 적용 금지)

### Step 5 — Faster R-CNN (ResNet50-FPN)
- `build_model()`: COCO pretrained + `FastRCNNPredictor(num_classes=2)`
- 스모크 테스트 통과 (ADSD/CADICA 배치, 박스 0개 배치 포함)
- 아키텍처/backbone **절대 변경 금지** (연구 지침)

### Step 6 — 학습 파이프라인
- `train_one_epoch`, `validate_loss`, `evaluate_detector`, `fit`
- Best 기준: **val mAP50**
- `last.pth` / `best.pth` / TensorBoard / `training_log.csv`
- Resume 지원 (`epochs_no_improve`도 체크포인트에 저장)
- EarlyStopping 지원

### Step 10 — ADSD Pretraining 실행 (완료)
- epochs=50, EarlyStopping patience=10
- Best val mAP50 = **0.1124**
- EarlyStopping으로 epoch 13에서 종료
- `best.pth` 저장됨

### 파이프라인 감사 + 통계 분석 (완료, 코드 수정 없음)
- trainable layers / optimizer / scheduler / EarlyStopping / evaluation / dataset 감사
- ADSD vs CADICA 통계 비교 (박스 크기, 상대 면적, pos/neg 비율 등)
- 결론: **코딩 버그가 아니라 도메인 불일치**

---

## 3. 현재 진행 중인 작업

- [ ] ADSD를 CADICA와 더 비슷하게 만드는 **전처리 전략 설계 검토**
  - 제안된 후보: **병변 중심 크롭(Lesion-Centered Crop)**으로 상대 면적 분포 정렬
  - **아직 구현하지 않음** — 설계 승인 대기
  - 단순 리사이즈는 효과 없음으로 분석됨 (`GeneralizedRCNNTransform`이 이미 800으로 통일 + 상대 면적은 균일 리사이즈에 불변)

---

## 4. 다음 작업

1. 전처리 전략(크롭 등)에 대한 **사용자 승인/방향 결정**
2. 승인 시 ADSD 전처리 → ADSD Pretraining 재실행 → CADICA Fine-tuning → Val/Test 평가
3. (보류) ImageNet/COCO-only CADICA baseline을 **동일 하이퍼파라미터**로 돌려서 fair comparison
4. Test set 예측 시각화 저장 (발표용) — CADICA fine-tuning 완료 후

---

## 5. 현재 가장 큰 문제

**ADSD → CADICA 전이학습 효과가 거의 없다** (또는 기대에 미치지 못한다).

원인은 코드 버그가 아니라 **두 데이터셋의 도메인 불일치**로 판단된다.

---

## 6. 현재 가설 (영향력 순위)

| 순위 | 가설 | 근거 |
|---|---|---|
| 1 | **Positive/Negative 비율 불일치** | ADSD 100% positive / CADICA 76.7% negative. ADSD에서 "항상 뭔가 찾으라"는 편향 학습 가능 |
| 2 | **상대 병변 면적(%) 불일치** | ADSD median 0.28% vs CADICA 0.60%. 절대 픽셀은 비슷하나 이미지 해상도 차이로 상대 스케일이 다름 |
| 3 | **Backbone 전체 unfreeze** | ResNet50 전체가 ADSD 5,829장에 재학습 → COCO 일반 특징이 ADSD 특유 통계로 쏠릴 위험 |
| 4 | **ADSD pretraining 품질 자체 낮음** | best val mAP50=0.1124, best epoch≈3 (너무 이른 수렴) |
| 5 | **병변 밀도 차이** | ADSD positive당 평균 1.0박스 vs CADICA positive당 1.58박스 |

**통제 가능한 레버(현재 제약 하)**: 상대 병변 면적(%) — 병변 중심 크롭으로만 조정 가능.  
**통제 불가**: positive/negative 비율, 병변 밀도(ADSD에 없는 negative/추가 병변을 만들지 않기로 함).

---

## 7. 확정된 실험 프로토콜 (변경 금지, 명시적 승인 전까지)

```
ADSD Pretraining
    ↓
Save Best Checkpoint (val mAP50)
    ↓
Load ADSD best.pth
    ↓
CADICA Fine-tuning
    ↓
CADICA Validation / Test Evaluation
```

- Detector: **Faster R-CNN**
- Backbone: **ResNet50-FPN**
- Split: `adsd_patient_split.csv`, `common_split.csv`만 사용 (**새 split 생성 금지**)
- CADICA 등급 → 단일 `Stenosis` 클래스
- CADICA: **모든 키프레임 포함** (박스 없는 프레임 = negative)
- Augmentation: **없음**
- Best 기준: **val mAP50**
- 최종 지표: mAP50, mAP50-95, Precision, Recall (+ val_loss는 모니터링만)

---

## 8. 관련 문서

- `EXPERIMENT_LOG.md` — 실험 이력 요약
- `PROJECT_ARCHITECTURE.md` — 코드/구조
- `TODO.md` — 할 일 체크리스트
- `docs/experiments/` — 실험별 상세
  - `E00_dataset.md`
  - `E01_adsd_pretraining.md`
  - `E02_cadica_finetuning.md`
  - `E03_domain_analysis.md`
  - `E04_crop_experiment.md`
