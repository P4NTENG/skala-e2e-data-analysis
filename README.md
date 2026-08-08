# SKALA E2E Data Analysis — NYC Yellow Taxi RatecodeID Prediction

NYC Yellow Taxi (2026-05) 데이터를 활용한 End-to-End 데이터 분석 및 RatecodeID 예측 ML 프로젝트.

## 프로젝트 구조

```
skala-e2e-data-analysis/
├── data/                          # 데이터셋 (gitignore)
│   ├── yellow_tripdata_2026-05.parquet          # 원본
│   ├── yellow_tripdata_2026-05_clean.parquet    # 정제본
│   └── yellow_tripdata_2026-05_ml.parquet       # ML용 feature subset
├── output/
│   ├── report.md                  # 종합 결과 리포트
│   ├── model_hgb.joblib           # HistGradientBoosting 모델
│   ├── model_rf.joblib            # RandomForest 모델
│   └── figures/                   # 시각화 차트 (PNG + HTML)
├── src/
│   ├── explore_distribution.py    # 초기 탐색적 EDA
│   ├── 2_visualization.py         # RatecodeID 시각화 (Seaborn + Plotly)
│   └── 4_ml_pipeline.py           # ML Pipeline (전처리 → 학습 → 평가 → 시각화)
├── DATA_CLEANING_NOTES.md         # 데이터 정제·Feature·통계 분석 문서
├── ASSIGNMENT.md                  # 과제 명세
├── PLAN.md                        # 프로젝트 계획
├── RUBRIC.md                      # 채점 기준
└── requirements.txt               # (필요 시 추가)
```

## 데이터셋

- [NYC Yellow Taxi 2026-05](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet)
- 원본: 4,090,836행 × 20컬럼
- 정제 후: 3,021,962행 (결측치 행 제거 + 이상치 필터링)

## 실행 방법

### 1. 환경 설정

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas polars numpy seaborn matplotlib plotly scipy scikit-learn joblib pyarrow kaleido
```

### 2. 데이터 다운로드

```powershell
# data/ 디렉토리에 yellow_tripdata_2026-05.parquet 다운로드
```

### 3. 시각화 실행

```powershell
python src/2_visualization.py
```

RatecodeID 분포, trip_distance/승객/시간대별 분석, 상관계수, η², t-test 요약 등 16개 차트를 `output/figures/`에 생성.

### 4. ML 파이프라인 실행

```powershell
python src/4_ml_pipeline.py
```

RandomForest + HistGradientBoosting 모델 학습, 5-fold CV 평가, classification report 출력, 모델 저장, 12개 결과 시각화 생성.

## 모델 성능

| 지표 | RandomForest | HistGradientBoosting |
|------|:---:|:---:|
| CV Balanced Accuracy | 0.7487 | **0.8207** |
| Test Balanced Accuracy | 0.7522 | **0.8293** |
| Test Weighted F1 | 0.8042 | **0.9090** |

### 클래스별 F1 (HGB)

| Standard | JFK | Unknown | Nassau | Negotiated | Newark |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.93 | 0.80 | 0.73 | 0.63 | 0.19 | 0.11 |

## Feature Importance

| 순위 | Feature | 중요도 |
|:---:|---------|:---:|
| 1 | DOLocationID | 0.431 |
| 2 | PULocationID | 0.211 |
| 3 | VendorID | 0.249 |
| 4 | passenger_count | 0.037 |
| 5 | hour | 0.031 |

## 결과 확인

- **`output/report.md`** — 전체 결과 리포트 (이미지 포함)
- **`DATA_CLEANING_NOTES.md`** — 데이터 정제·Feature·통계 분석 상세
- **`output/figures/`** — 모든 시각화 차트
