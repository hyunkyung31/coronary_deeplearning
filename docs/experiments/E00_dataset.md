# E00 — Dataset Inspection & Protocol Lock

> 상태: 완료  
> 관련: `EXPERIMENT_LOG.md` E0, `PROJECT_STATUS.md` Step 0~1  
> 목적: 학습 코드 전에 ADSD/CADICA 실물·split·라벨 체계를 검증하고 프로토콜을 고정한다.

---

## 1. 목적

- ADSD / CADICA 디렉터리·어노테이션·split CSV 정합성 검증
- patient-wise split만 사용하도록 확정
- CADICA 협착 등급 처리 방식 확정
- Dataset 구현 전에 가정하지 말고 실측으로 결정

---

## 2. 데이터 위치

### Drive
```
/content/drive/MyDrive/팀프로젝트/DL 프로젝트/ADSD+CADICA/
├── dataset/
│   ├── Stenosis detection.zip
│   └── CADICA.zip
└── split_csv/
    ├── adsd_patient_split.csv
    └── common_split.csv
```

### Colab 로컬 (압축 해제 후)
```
/content/adsd_dataset/dataset/          # *.bmp + *.xml
/content/cadica_dataset/selectedVideos/ # pX/vY/input + groundtruth
```

---

## 3. ADSD 검증 결과

| 항목 | 결과 |
|---|---|
| 어노테이션 | Pascal VOC XML (`xmin,ymin,xmax,ymax`, class=`Stenosis`) |
| 이미지 | `.bmp` 8,325장 |
| 매칭 | bmp↔xml 1:1, 누락 0 |
| patient_id | `"_".join(stem.split("_")[:2])` → `14_XXX` |
| Split CSV | 64명, train 45 / val 10 / test 9 |
| n_images | CSV 합계 8,325 = 실측 완전 일치 |
| Negative | **0장** (모든 이미지에 ≥1 box) |
| 해상도 | 512(1603), 608(343), 800(5332), 1000(1047) |
| 미사용 | `video_test/`, `video_val/`, `timing_*.xlsx`, 레거시 `train/test_labels.csv` |

**주의**: `train_labels.csv`/`test_labels.csv`는 patient-wise가 아님(63/64명 누수). **절대 split으로 쓰지 말 것.**

---

## 4. CADICA 검증 결과

| 항목 | 결과 |
|---|---|
| 어노테이션 | `x y w h category` (txt), category=`p0_20`~`p100` |
| 이미지 | `.png`, 전부 512×512 |
| Split CSV | 42명, 334 video, patient-wise 누수 0 |
| split 분포 | train 33 / val 4 / test 5 patients |
| `n_frames` vs input 개수 | 334개 video 전부 일치 |
| groundtruth | `has_groundtruth`와 폴더 존재 일치 (불일치 0) |
| lesion 비디오라도 | **프레임마다 gt가 없을 수 있음** (gt 파일 3,685 vs lesion 키프레임 11,791) |

### 등급 분포 (groundtruth 전수)
| Label | 개수 |
|---|---|
| p0_20 | 1,832 |
| p20_50 | 1,066 |
| p50_70 | 984 |
| p70_90 | 850 |
| p90_98 | 839 |
| p100 | 204 |
| p99 | 60 |
| **합계** | **5,835** |

멀티박스 프레임: 1,442 / 3,685 (39.1%)

---

## 5. 확정된 프로토콜 결정

1. **Split**: `adsd_patient_split.csv`, `common_split.csv`만 사용. 새 split 금지.
2. **CADICA 등급**: 전부 단일 `Stenosis` (label=1)로 병합. 원본 등급은 `raw_categories`로만 보존.
3. **CADICA negative 정책 (옵션 1)**: 모든 키프레임 포함. gt 없으면 boxes=0.
4. **`metadata.xlsx`**: 파이프라인 미사용.
5. **`num_classes=2`** (background + Stenosis).
6. **학습 대상 폴더**: ADSD=`dataset/`, CADICA=`selectedVideos/`만.

---

## 6. Dataset 샘플 수 (구현 후 재확인)

| | train | val | test |
|---|---|---|---|
| ADSD | 5,829 | 1,261 | 1,235 |
| CADICA | 13,316 | 1,396 | 1,087 |

CADICA train 13,316 = `common_split.csv` train `n_frames` 합계와 일치.

---

## 7. 산출물 / 다음

- 산출물: 검사 로그, 시각화 샘플 확인, 프로토콜 확정
- 다음 실험: `E01_adsd_pretraining.md`
