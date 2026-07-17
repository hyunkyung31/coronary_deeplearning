# E03 — ADSD vs CADICA Domain Analysis

> 상태: 완료 (코드/파이프라인 수정 없음)  
> 관련: `EXPERIMENT_LOG.md` A1~A2, Colab Cell 46~52  
> 목적: 전이 효과가 약한 원인이 버그인지 도메인 불일치인지 정량적으로 판별

---

## 1. 목적

파이프라인 감사 + 데이터셋 통계 비교로:

1. 코딩 버그 여부 확인
2. ADSD↔CADICA의 **통제 가능한/불가능한** 분포 차이 식별
3. 전이 실패 가설을 순위화

**결론 한 줄**: 문제는 코딩 버그가 아니라 **도메인 불일치**다.

---

## 2. 파이프라인 감사 요약 (A1)

| 항목 | 결과 |
|---|---|
| Trainable layers | Backbone 전체 unfreeze. FrozenBatchNorm만 비학습 |
| Optimizer | SGD 단일 param group (정상) |
| Scheduler | epoch 끝 1회 step, resume OK |
| EarlyStopping | val mAP50, patience 로직 OK |
| Evaluation | ADSD/CADICA 동일 (`score=0.5`, `IoU=0.5`, torchmetrics) |
| Dataset negatives | CADICA OK / **ADSD는 원천적으로 0장** |
| 코드 버그 | **없음** |

의심스러운 **설계상 특징** (버그 아님):
1. Backbone 전체 학습
2. ADSD 100% positive

---

## 3. 규모 통계

| | ADSD | CADICA |
|---|---|---|
| 이미지 수 | 8,325 | 15,799 |
| 박스 수 | 8,326 | 5,835 |

---

## 4. Positive vs Negative (가장 큰 차이)

| | ADSD | CADICA |
|---|---|---|
| Positive | 8,325 (**100%**) | 3,685 (**23.3%**) |
| Negative | 0 (**0%**) | 12,114 (**76.7%**) |

**해석**: ADSD는 “항상 병변이 있다”는 편향을 학습하기 쉽고, CADICA는 대부분 “아무것도 찾지 말아야 하는” 이미지다. 전이 시 false positive 성향으로 작용할 수 있음.

---

## 5. 박스 절대 크기 (px)

| | ADSD | CADICA |
|---|---|---|
| width mean | 46.18 | 46.04 |
| height mean | 39.49 | 44.88 |
| width median | 41 | 41 |
| height median | 36 | 39 |

**해석**: 절대 픽셀 크기는 **거의 동일**. “CADICA 병변이 픽셀 단위로 훨씬 작다”는 가설은 기각.

---

## 6. 상대 박스 면적 (% of image)

| | ADSD | CADICA |
|---|---|---|
| median | **0.285%** | **0.595%** |
| mean | 0.336% | 0.823% |
| median 비율 (ADSD/CADICA) | **0.48배** | — |

**해석**: CADICA 병변이 이미지 대비 **약 2배 더 큼**.  
원인: 절대 박스 크기는 비슷한데 ADSD 이미지가 더 큼(주로 800/1000) → 상대 면적만 작아짐.

초기 질문 “CADICA가 ADSD보다 유의하게 작은가?” → **아니요, 반대**.

> 참고: 초기 Mann-Whitney 코드가 `alternative='greater'`(ADSD>CADICA)로 잘못 설정되어 `p=1.00`이 나옴.  
> 올바른 방향은 `alternative='less'` (ADSD < CADICA). 재실행 권장.

---

## 7. 이미지 해상도

| ADSD | count |
|---|---|
| 512×512 | 1,603 |
| 608×608 | 343 |
| 800×800 | 5,332 |
| 1000×1000 | 1,047 |

CADICA: **전부 512×512** (15,799)

### 중요: 단순 리사이즈는 레버가 아님
`GeneralizedRCNNTransform`이 이미 `min_size=800`으로 통일하고, 상대 면적(%)은 균일 리사이즈에 **불변**이다.  
→ 원본 파일을 512로 맞추는 전처리는 **전이 관점에서 효과 없음**.

---

## 8. 병변 밀도

| | ADSD | CADICA |
|---|---|---|
| 전체 이미지당 평균 박스 | 1.000 | 0.369 |
| Positive만 평균 박스 | 1.000 | **1.583** |

CADICA는 양성 프레임에서 멀티박스 비율이 더 높음 (gt 기준 39.1%가 multi-box).

---

## 9. 전이 실패 가설 순위

| 순위 | 가설 | 통제 가능? |
|---|---|---|
| 1 | Pos/Neg 비율 정반대 | ❌ (ADSD에 negative 원천 부재, synthetic 금지) |
| 2 | 상대 병변 면적(%) 불일치 | ✅ 병변 중심 크롭으로만 조정 가능 |
| 3 | Backbone 전체 unfreeze | △ (아키텍처 변경은 아니지만 freeze는 별도 실험/승인 필요) |
| 4 | ADSD pretrain 품질 낮음 (mAP50=0.1124) | △ ADSD 추가 튜닝은 현재 방침상 안 함 |
| 5 | Positive 내 박스 밀도 차이 | ❌ (가짜 어노테이션 없이 불가) |

---

## 10. 다음 실험으로의 연결

- 아키텍처/디텍터/augmentation을 바꾸지 않고, **진짜 ADSD만**으로 상대 면적을 맞추는 전략 → `E04_crop_experiment.md`
- CADICA fine-tune 수치와 함께 해석 → `E02_cadica_finetuning.md`
- Fair comparison baseline → 보류 (TODO P2)

---

## 11. 하지 않은 것

- 파이프라인 코드 수정
- Synthetic negative 생성
- Detector/backbone 변경
- 새 split 생성
