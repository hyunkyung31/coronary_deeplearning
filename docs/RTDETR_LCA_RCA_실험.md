# RT-DETR LCA / RCA 분리 학습 실험

> Group 2(현경) 탐지 파트. `common_split.csv`(view 컬럼 포함) 기준으로
> LCA/RCA를 분리해서 학습하고, 통합 baseline과 비교하기 위한 실행 기록.

## 0. 배경

- 통합(LCA+RCA 혼합) 학습 baseline: `train_v3_amp_on` (imgsz=640)

  | Split | P | R | mAP50 | mAP50-95 |
  |-------|------|------|-------|----------|
  | Val   | 0.229 | 0.219 | 0.117 | 0.030 |
  | Test  | 0.592 | 0.289 | 0.308 | 0.150 |

- imgsz 768 ablation은 640보다 낫지 않아 640 유지로 결론 (`train_imgsz768`):

  | Split | P | R | mAP50 | mAP50-95 |
  |-------|------|------|-------|----------|
  | Val   | 0.165 | 0.281 | 0.102 | 0.034 |
  | Test  | 0.418 | 0.242 | 0.221 | 0.096 |

- 다음 순서: **LCA/RCA 분리 학습**. 뷰가 다르면 혈관 구조/각도가 달라
  하나의 모델이 두 분포를 동시에 학습하면서 불안정(val mAP oscillation)
  했을 가능성이 있다는 가설.

- `common_split.csv` 기준 view 분포 (video 단위):

  | view | train | val | test | total |
  |------|-------|-----|------|-------|
  | LCA  | 169   | 28  | 19   | 216   |
  | RCA  | 102   | 8   | 8    | 118   |

  RCA는 특히 val/test가 각각 8개 video뿐이라 표본이 작다는 점을
  결과 해석 시 감안해야 한다.

## 1. 실행 순서 (Colab)

1. 통합 YOLO 데이터셋(`/content/cadica_yolo`)과 `common_split.csv`가
   이미 있는 상태에서 `scripts/build_lca_rca_datasets.py` 실행
   → `/content/cadica_yolo_lca`, `/content/cadica_yolo_rca` 생성
   (symlink 방식, 원본 이미지 복사 없음)
2. `scripts/train_eval_rtdetr_lca_rca.py`의 `train_view("LCA")`,
   `train_view("RCA")` 각각 실행 (별도 Colab 셀 권장 — 하나가 NaN
   collapse 나도 다른 쪽에 영향 없게)
3. `eval_view("LCA")`, `eval_view("RCA")`로 val/test 성능 확인
4. `summarize()`로 baseline과 나란히 비교

## 2. 설계 결정

- **두 뷰 모두 `rtdetr-l.pt`(원본 pretrained)에서 새로 시작.**
  통합 학습된 `last.pt`/`best.pt`에서 이어가면 LCA 비중이 큰(216 vs 118)
  통합 데이터의 편향이 그대로 넘어갈 수 있어 공정한 비교가 어려움.
- 하이퍼파라미터는 `train_v3_amp_on`과 동일하게 고정
  (imgsz=640, batch=8, lr0=0.0005, AdamW, amp=True, patience=20, epochs=50).
  한 번에 하나의 변수만 바꾸는 원칙 유지.
- RCA는 데이터가 더 적으므로 오버피팅/조기 수렴 신호를 특히 주의해서
  볼 것. 필요하면 이후 ablation에서 RCA만 augmentation을 강하게
  주는 것을 검토.

## 3. 결과

### 3.1 뷰 전용 모델 (동일 뷰 test로 평가)

| Split | View | P | R | mAP50 | mAP50-95 | notes |
|-------|------|------|------|-------|----------|-------|
| Val   | LCA  | — | — | **0.092** | — | `train_lca_v1` |
| Test  | LCA  | — | — | **0.463** | — | |
| Val   | RCA  | 0.557 | 0.259 | **0.243** | 0.064 | val n=81, EarlyStop@41 / best@21 |
| Test  | RCA  | 0.073 | 0.067 | **0.013** | 0.004 | 붕괴 |

### 3.2 공정 비교: 통합 baseline을 같은 뷰-test에 평가

| Model | Eval set | P | R | mAP50 | mAP50-95 |
|-------|----------|------|------|-------|----------|
| LCA-only (`train_lca_v1`) | LCA-TEST | — | — | 0.463 | — |
| Baseline (`train_v3_amp_on`) | LCA-TEST | — | — | **0.485** | 0.242 |
| RCA-only (`train_rca_v1`) | RCA-TEST | 0.073 | 0.067 | 0.013 | 0.004 |
| Baseline (`train_v3_amp_on`) | RCA-TEST | 0.188 | 0.189 | **0.060** | 0.020 |

→ **같은 test에서 통합 baseline이 뷰 전용보다 항상 우위.**

데이터 규모 (frames): LCA train/val/test = 1818 / 254 / 153,
RCA = 1259 / 81 / 120.

## 4. 결론 — **뷰 분리 폐기**

1. **LCA**: 전용 모델(0.463) < baseline(0.485). 이득 없음.
2. **RCA**: val mAP50=0.243은 좋아 보였으나 test=0.013으로 붕괴.
   val이 video 8개 / frame 81장뿐이라 early-stop 신호가 신뢰할 수 없음.
   baseline도 RCA-TEST에서 mAP50=0.060으로 약하지만, RCA-only보다 4배 이상 나음.
3. 통합 baseline이 LCA에서 강하고 RCA에서 약한 asymmetry는 남지만,
   뷰를 쪼개면 표본이 더 줄어서 오히려 악화 → **통합 640 baseline 유지**.

다음 실험: **CADICA train + ARCADE stenosis train** 병합
(`cursor/cadica-arcade-merge-a265`, val/test는 CADICA만).

## 5. 해석 가이드 (사후)

- 가설(뷰 분리 → 안정화)은 **기각**. 불안정/낮은 mAP의 주원인은
  뷰 혼합보다 **데이터 절대량·라벨 난이도** 쪽에 가깝다.
- RCA가 본질적으로 더 어렵다(baseline도 RCA-TEST 0.060 vs LCA-TEST 0.485).
  뷰 분리로는 해결되지 않음 → 외부 데이터(ARCADE) 또는 학습 안정화
  (`cos_lr`, 낮은 lr)로 이동.
