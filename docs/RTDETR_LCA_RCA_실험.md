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

## 3. 결과 (실행 후 채우기)

| Split | View | P | R | mAP50 | mAP50-95 |
|-------|------|---|---|-------|----------|
| Val   | LCA  |   |   |       |          |
| Val   | RCA  |   |   |       |          |
| Test  | LCA  |   |   |       |          |
| Test  | RCA  |   |   |       |          |

## 4. 해석 가이드

- LCA/RCA 각각의 mAP50이 통합 baseline(val 0.117 / test 0.308)보다
  **둘 다** 높아지면 → 뷰 분리가 유효했다는 신호. 다음 단계에서
  두 모델을 뷰별로 서빙하거나, 뷰 분류기 + 뷰별 탐지기 조합을 고려.
- 한쪽(특히 LCA, 데이터가 더 많은 쪽)만 개선되고 RCA는 표본 부족으로
  큰 변동을 보이면 → RCA는 데이터 증강/외부 데이터 없이는 한계가
  있다는 결론으로, 통합 모델 유지 + RCA 전용 후처리(threshold 조정 등)
  같은 대안을 검토.
- 두 뷰 모두 개선이 없거나 나빠지면 → 뷰 분리보다 다른 요인(lr, 데이터
  절대량, augmentation)이 더 중요하다는 뜻이므로 다음 순서였던 CLAHE
  ablation이나 cos_lr/lr0=0.0003 등 정제 아이디어로 넘어감.
