# CADICA + ARCADE 병합 학습 (RT-DETR)

> Group 2(현경) 탐지 파트. LCA/RCA 분리 학습이 baseline보다 낫지 않다는 결론
> 이후, 데이터 자체를 늘리기 위한 실험. 공개 데이터셋 ARCADE(stenosis)를
> train에만 보강하고, 평가는 항상 CADICA val/test로 고정한다.

## 0. 배경 — 왜 이 실험을 하나

- LCA/RCA 분리 결과 → **뷰 분리 폐기**, 통합 640 baseline 유지.

  | Model | Eval | mAP50 |
  |-------|------|-------|
  | LCA-only | LCA-TEST | 0.463 |
  | Baseline | LCA-TEST | **0.485** |
  | RCA-only | RCA-TEST | 0.013 |
  | Baseline | RCA-TEST | **0.060** |

  RCA val(0.243)은 좋아 보였으나 test에서 붕괴(0.013). val n=81(video 8)
  이라 early-stop 신호가 신뢰되지 않음. baseline도 RCA가 본질적으로 약하지만
  뷰를 쪼개면 더 나빠짐.
- CADICA는 환자 42명으로 탐지 학습 기준 데이터가 작음 → 다음 카드로
  **외부 공개 데이터 추가(ARCADE stenosis)** 시도.

## 1. Split 구성 (핵심 원칙)

| Split | 구성 |
|-------|------|
| Train | CADICA train + ARCADE stenosis train (필요하면 val도) |
| Val   | **CADICA val만** |
| Test  | **CADICA test만** |

ARCADE는 학습량 보강용이고, Group5(YOLO)와 공정 비교하려면 평가는 CADICA
val/test만 유지해야 한다. → **합치는 건 train만, 평가는 항상 CADICA.**

## 2. ARCADE 구조 (CADICA와 다른 점)

CADICA처럼 `pX/vY/groundtruth/*.txt` 구조가 아니다.

- **환자 ID 없음**: privacy 때문에 제거됨. 폴더로 환자 단위 구분 불가 → 정상.
- **bbox 위치**: 이미지 옆 txt가 아니라 **COCO JSON** 안의 `annotations[].bbox`
  (`[x, y, w, h]`, CADICA와 같은 XYWH).
- 찾아야 할 경로 (버전마다 약간 다를 수 있음):

  ```
  stenosis/          ← syntax 폴더 말고 stenosis 폴더
    train/
      images/*.png
      annotations/train.json   ← 여기
    val/
    test/
  ```

- `syntax/`는 혈관 SYNTAX 세그멘테이션용이라 탐지 메인에는 쓰지 않는다.
  `stenosis/`만 사용.
- `categories`에 SYNTAX 세그먼트(1~25)와 `stenosis` 카테고리가 함께 있을 수
  있어서, 이름에 "sten"이 포함된 카테고리만 걸러서 lesion(class 0)으로
  변환한다 (`convert_arcade_to_yolo.py`가 자동 처리, 못 찾으면 경고 출력).

## 3. 실행 순서 (Colab)

1. ARCADE zip 다운로드 & 압축 해제 → `/content/arcade/stenosis/...`
2. `scripts/convert_arcade_to_yolo.py` 실행
   → COCO bbox를 YOLO `class cx cy w h`로 변환, `arcade_` prefix로
   CADICA와 파일명 충돌 방지, **train만** 변환 (`/content/arcade_yolo/`)
3. `scripts/build_cadica_arcade_dataset.py` 실행
   → CADICA train + ARCADE train을 합쳐 `/content/cadica_arcade_yolo/`
   생성, val/test는 CADICA만 그대로 symlink
4. `scripts/train_eval_rtdetr_cadica_arcade.py`의 `train()` → `eval_and_compare()`
   실행 → CADICA-only baseline과 나란히 비교되는 표 출력

## 4. 설계 결정

- **라벨 정의 통일**: ARCADE stenosis는 SYNTAX 기준 ≥50% narrowing, CADICA는
  <20%~100%로 더 세분화된 등급. 합칠 때는 **binary lesion(bbox 있음=1)**으로
  통일한다. 중증도 세분류는 이후 별도 과제.
- **해상도**: ARCADE는 보통 512×512, CADICA는 프레임마다 다를 수 있음.
  RT-DETR/YOLO의 letterbox가 학습 시 알아서 맞춰주므로 별도 resize 전처리는
  필수가 아니다.
- **환자 단위 split**: ARCADE는 patient ID가 없어 patient-wise split이
  불가능하다. 그래서 ARCADE는 **train 보강 전용**으로만 쓰고, leakage
  방지는 CADICA의 `common_split.csv` patient split에 전적으로 맡긴다.
- **CLAHE/GMM 등 추가 필터**: 탐지 메인에는 넣지 않음(기존 실험 결론 유지).
  필요하면 Ultralytics 기본 aug 쪽에서만 약하게 적용.
- 하이퍼파라미터는 baseline(`train_v3_amp_on`)과 동일하게 고정
  (imgsz=640, batch=8, lr0=0.0005, AdamW, amp=True, patience=20, epochs=50).
  "ARCADE 추가" 한 가지 변수만 바꿔서 비교.

## 5. 결과 (실행 후 채우기)

| Split | 구성 | P | R | mAP50 | mAP50-95 |
|-------|------|---|---|-------|----------|
| Val   | CADICA-only (baseline) | 0.229 | 0.219 | 0.117 | 0.030 |
| Val   | CADICA+ARCADE |   |   |   |   |
| Test  | CADICA-only (baseline) | 0.592 | 0.289 | 0.308 | 0.150 |
| Test  | CADICA+ARCADE |   |   |   |   |

## 6. 해석 가이드

- mAP50/mAP50-95가 baseline보다 오르면 → 데이터 보강 효과 확인, 다음은
  ARCADE val도 포함(`--include-val`)해서 추가 이득이 있는지 ablation.
- Precision은 오르는데 Recall이 떨어지면(또는 반대) → ARCADE 라벨 정의
  차이(≥50% narrowing만 lesion)가 CADICA의 경미한 병변 인식에 영향을
  준 것일 수 있음. 이 경우 발표에서는 "정의 차이로 인한 트레이드오프"로
  명시.
- 개선이 없거나 나빠지면 → 도메인 시프트(병원/장비/라벨 기준 차이)가
  이득보다 크다는 뜻. 이 경우 CADICA-only baseline을 최종으로 유지하고,
  학습 안정화(`cos_lr=True`, `lr0=0.0003`) 쪽으로 이동.
