# E04 — ADSD Lesion-Centered Crop (Transferability Experiment)

> 상태: **설계만 완료 / 미구현 / 사용자 승인 대기**  
> 관련: `EXPERIMENT_LOG.md` E4, `PROJECT_STATUS.md` 현재 진행 중 작업  
> 목적: ADSD pretraining 데이터를 CADICA와 더 비슷하게 만들어 전이력을 높일 수 있는지 검증

---

## 1. 목적 / 비목적

### 목적
- ADSD → CADICA **transferability** 극대화
- 진짜 ADSD 픽셀만 사용 (synthetic negative / 가짜 어노테이션 금지)
- Detector architecture 변경 금지

### 비목적
- ADSD validation mAP 최대화
- CADICA에 동일한 전처리 적용
- Pos/Neg 비율을 인위적으로 맞추기

---

## 2. 왜 이 실험이 필요한가 (E03 요약)

통제 가능한 차이 중 실질 레버:

| 차이 | 왜 레버인가 |
|---|---|
| 상대 병변 면적(%) | ADSD median 0.28% vs CADICA 0.60%. RPN의 상대 스케일 감각에 영향 |
| 원본 해상도 | **레버 아님** — 모델이 이미 800으로 리사이즈, 상대 면적은 균일 리사이즈에 불변 |
| Pos/Neg | 중요하지만 **이 실험에서 건드리지 않음** (ADSD에 negative 부재) |
| 박스 밀도 | 가짜 어노테이션 없이 불가 |

---

## 3. 제안 전처리: Lesion-Centered Crop

### 아이디어
각 ADSD 이미지에서 어노테이션 bbox를 중심으로,  
**목표 상대 면적(예: CADICA median ≈ 0.60%)**이 되도록 정사각형 crop.

```
target_rel = 0.006   # 0.60%
crop_area  = box_area / target_rel
crop_side  = sqrt(crop_area)
→ box 중심 기준 crop_side × crop_side 잘라냄
→ 이미지 경계 밖이면 clamp
→ crop 내부 bbox 좌표를 새 좌표계로 변환
```

### 규칙
| 항목 | 처리 |
|---|---|
| 필터링 | 거의 없음 (버리기보다 자르기). 예상 잔여 ≈ 거의 전량 |
| 원본 파일 리사이즈 | **하지 않음** (효과 없음) |
| 크롭 대상 | ADSD만 |
| CADICA | **절대 적용하지 않음** |
| Split | patient_id 배정 불변 |
| 라벨 | 단일 Stenosis 유지 |
| 이미 목표 비율보다 큰 병변 | 과도한 crop으로 잘릴 위험 → 원본 유지 또는 최소 crop |

멀티박스 이미지(ADSD는 거의 1박스/장): 현재 ADSD는 사실상 1박스라 단순.  
드물게 2박스면 — **설계 승인 시 처리 규칙을 명시해야 함** (예: 첫 박스 기준 / union bbox 기준). 현재 미확정.

---

## 4. 실험 설계 (승인 시)

### Arm A — 기존 (이미 완료: E01)
- 원본 ADSD pretrain → CADICA fine-tune (E02 HP)

### Arm B — Crop ADSD (신규)
1. ADSD에 lesion-centered crop 적용 (통계로 상대 면적 재확인)
2. **동일 HP**로 ADSD pretrain 재실행 (E01과 동일하게 유지 권장)
3. 새 `best.pth`로 CADICA fine-tune (**E02와 동일 HP**)
4. Test mAP50 / mAP50-95 / P / R 비교 (A vs B)

> 비교의 공정성을 위해 CADICA 쪽 HP·평가·split을 바꾸지 말 것.

---

## 5. Pros / Cons / Risks

### Pros
- 실측된 유효 차이(상대 면적)를 정확히 타겟
- 진짜 픽셀만 사용, 가짜 라벨 없음
- 필터링 대비 데이터 손실 적음
- 보고서에 재현 가능하게 기술 가능

### Cons / Risks
1. **해부학적 맥락 손실** (과도한 zoom-in) → 오히려 ADSD-특화 특징만 학습할 위험
2. Pos/Neg·밀도 문제는 **해결하지 못함** (부분 개입)
3. 경계 근처 병변은 clamp로 목표 비율 미달 가능
4. 추가 GPU 시간 (ADSD pretrain + CADICA fine-tune 재실행)
5. CADICA에 실수로 동일 전처리를 넣으면 실험이 오염됨

---

## 6. 성공 기준 (제안, 승인 필요)

최소 하나라도 만족하면 “유망”으로 볼지 사전 합의 권장:

- [ ] CADICA **Test mAP50**이 Arm A 대비 개선
- [ ] CADICA Test Precision 또는 Recall이 명확히 개선 (특히 FP 감소 여부)
- [ ] 상대 면적 분포가 CADICA에 통계적으로 근접 (전처리 검증용)

실패해도 남는 가치: “상대 면적만 맞춰서는 부족하고 Pos/Neg가 지배적”이라는 반증.

---

## 7. 구현 시 주의 (아직 코드 없음)

승인 후에만 구현:

1. ADSD 전용 Dataset/전처리 경로와 CADICA 경로를 **물리적으로 분리**
2. 크롭 전후 통계 셀을 먼저 돌려 분포 확인 → 그다음 학습
3. 체크포인트 경로를 분리  
   예: `checkpoints/adsd_crop/`, `logs/adsd_crop/`, `checkpoints/cadica_from_adsd_crop/`
4. `PROJECT_STATUS.md` / `EXPERIMENT_LOG.md` / 본 파일에 결과 기입
5. Architecture / augmentation / split은 변경하지 말 것

---

## 8. 현재 결정 요청 (사용자)

아래 중 하나로 답해주면 다음 단계 진행:

1. **승인** — 위 설계대로 구현·실험
2. **수정 후 승인** — 목표 상대 면적, 멀티박스 규칙, crop clamp 정책 등 변경
3. **기각** — 크롭 실험 안 함. E02 완료 또는 다른 방향 검토

---

## 9. 결과 (미기입)

| Arm | CADICA Test mAP50 | mAP50-95 | P | R | 비고 |
|---|---|---|---|---|---|
| A 원본 ADSD init | | | | | E02 |
| B Crop ADSD init | | | | | 본 실험 |

### 해석
- (실행 후 작성)
