# TODO.md

> Single Source of Truth — 앞으로 할 일 체크리스트  
> 마지막 업데이트: 2026-07-17  
> 우선순위: P0(즉시) > P1(다음) > P2(여유 시) > P3(보류/장기)

---

## P0 — 문서 / 인수인계

- [x] `PROJECT_STATUS.md` 작성
- [x] `EXPERIMENT_LOG.md` 작성
- [x] `PROJECT_ARCHITECTURE.md` 작성
- [x] `TODO.md` 작성
- [ ] 사용자 검토 후 부족한 내용 보완 (요청 시)

---

## P0 — 현재 의사결정 (코드 수정 전)

- [ ] ADSD→CADICA 전처리 전략(병변 중심 크롭) **승인 / 수정 / 기각** 결정
- [ ] 결정 전 구현 금지 (설계만 완료된 상태)
- [ ] (선택) Mann-Whitney 검정 방향을 `alternative='less'`로 재실행해 p-value 확정 기입

---

## P1 — CADICA Fine-tuning 파이프라인 완성 (승인된 현 프로토콜)

> ImageNet baseline 추가 없이, ADSD pretrained → CADICA만 먼저 완성

- [ ] Cell 40: `CADICA_CKPT_DIR` 기존 체크포인트 사전 점검
- [ ] Cell 41: ADSD `best.pth` 로드 + 새 optimizer/scheduler 생성
- [ ] Cell 42: CADICA fine-tuning 실행 (epochs=20, patience=5, lr=0.001, StepLR step_size=8)
- [ ] Cell 43: CADICA `best.pth` 재로드 후 Validation/Test 평가
  - [ ] mAP50
  - [ ] mAP50-95
  - [ ] Precision
  - [ ] Recall
  - [ ] Validation loss
  - [ ] `logs/cadica/final_evaluation.json` 저장
- [ ] Cell 44: Test set 예측 시각화 저장 (발표용)
- [ ] `EXPERIMENT_LOG.md`의 E2 섹션에 **최종 수치 결과 기입**

---

## P1 — 전처리 실험 (크롭 전략이 승인된 경우에만)

- [ ] ADSD lesion-centered crop 전처리 구현 (CADICA에는 적용 금지)
- [ ] 크롭 후 상대 면적 분포가 CADICA에 근접하는지 통계로 재확인
- [ ] 전처리된 ADSD로 Pretraining 재실행 (기존 HP 유지 권장, 변경 시 별도 기록)
- [ ] 새 ADSD `best.pth`로 CADICA Fine-tuning 재실행
- [ ] 원본 ADSD-pretrain 전이 결과와 비교 표 작성
- [ ] `EXPERIMENT_LOG.md`에 E4 결과 추가

---

## P2 — Fair Comparison Baseline (시간 여유 시)

- [ ] ImageNet/COCO-only CADICA baseline 실행
  - HP는 CADICA fine-tuning과 **완전 동일**
  - Init만 `pretrained_backbone=True` (ADSD 로드 없음)
  - 경로: `checkpoints/cadica_baseline_imagenet/`, `logs/cadica_baseline_imagenet/`
- [ ] ADSD-init vs ImageNet-init Test 성능 비교표 작성
- [ ] `EXPERIMENT_LOG.md`에 E3 결과 추가

---

## P2 — 발표 / 리포트 정리

- [ ] Test 시각화 중 대표 사례 선별 (성공/실패/negative FP)
- [ ] ADSD vs CADICA 도메인 불일치 통계 그림 정리 (pos/neg, relative area, resolution)
- [ ] 가설 요약 슬라이드용 bullet 정리
- [ ] 최종 성능 표 (val/test) 정리

---

## P3 — 보류 (현재 단계에서 하지 않음)

명시적 요청 전까지 **구현하지 말 것**.

- [ ] Augmentation (horizontal flip 포함) 추가
- [ ] Backbone partial freeze / differential LR
- [ ] Anchor / NMS / min_size 등 아키텍처/디텍터 설정 변경
- [ ] CADICA 다중 클래스(등급) 검출로 전환
- [ ] Synthetic negative / 가짜 어노테이션 생성
- [ ] Grid Search / Optuna 등 HP 탐색
- [ ] Detector를 RT-DETR / YOLO 등으로 재교체
- [ ] 새 train/val/test split 생성

---

## 운영 체크리스트 (매번 학습 전)

- [ ] Drive 마운트 확인
- [ ] `DRIVE_ROOT`가 `.../ADSD+CADICA`인지 확인
- [ ] zip / split CSV 4개 경로 "있음"
- [ ] `/content/adsd_dataset`, `/content/cadica_dataset` 존재 (없으면 압축 해제)
- [ ] `torchmetrics` 설치됨
- [ ] GPU 사용 가능 (`cuda`)
- [ ] 체크포인트가 Drive에 저장되는지 확인 (`checkpoints/`)
- [ ] Resume 시 `[resume] ...` 로그 확인

---

## 완료 기록 규칙

- 항목을 끝내면 `[x]`로 바꾸고, 가능하면 날짜를 옆에 적는다.
- 실험 수치는 **반드시** `EXPERIMENT_LOG.md`에도 옮긴다 (TODO만 체크하고 숫자 안 남기지 말 것).
- 프로토콜을 바꾸면 `PROJECT_STATUS.md`의 "확정된 실험 프로토콜"과 `PROJECT_ARCHITECTURE.md`의 "절대 건드리면 안 되는 부분"을 함께 수정한다.
