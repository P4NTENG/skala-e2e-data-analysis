# NYC Yellow Taxi — RatecodeID 예측

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)

NYC Yellow Taxi 2026년 5월 데이터를 활용한 End-to-End 데이터 분석 및 `RatecodeID` 예측 ML 프로젝트.

## 데이터셋

[NYC Yellow Taxi Trip Data (2026-05)](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet)

| | 원본 | 정제본 |
|---|:---:|:---:|
| 행 | 4,090,836 | 3,021,962 |
| 컬럼 | 20 | 11 (feature subset) |
| 타겟 | `RatecodeID` | 6-class |

## 빠른 시작

```bash
git clone https://github.com/P4NTENG/skala-e2e-data-analysis.git
cd skala-e2e-data-analysis

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# data/ 디렉토리에 데이터셋 다운로드
# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet

python src/4_ml_pipeline.py
```

## 실행 방법

| 스크립트 | 설명 |
|----------|------|
| `src/2_visualization.py` | EDA + 통계 분석 차트 생성 (Seaborn / Plotly) |
| `src/4_ml_pipeline.py` | ML 파이프라인: 전처리 → 학습 → 평가 → 시각화 |

## 디렉토리 구조

```
├── data/                    # 데이터셋 (gitignore)
├── output/
│   ├── report.md            # 종합 결과 리포트
│   ├── model_hgb.joblib     # HistGradientBoosting 모델
│   ├── model_rf.joblib      # RandomForest 모델
│   └── figures/             # 차트 (PNG + 인터랙티브 HTML)
├── src/
│   ├── explore_distribution.py
│   ├── 2_visualization.py
│   └── 4_ml_pipeline.py
├── requirements.txt
└── README.md
```

## 입력 변수 (Feature)

| 변수 | 타입 | 설명 |
|------|------|------|
| `PULocationID` | categorical (265) | 승차 위치 |
| `DOLocationID` | categorical (265) | 하차 위치 |
| `VendorID` | binary | 택시 벤더 |
| `payment_type` | categorical | 결제 수단 |
| `passenger_count` | numeric | 탑승 인원 |
| `store_and_fwd_flag` | binary | 차량 메모리 저장 여부 |
| `hour` | numeric | 승차 시간 (0-23) |
| `day_of_week` | numeric | 요일 (0-6) |
| `is_weekend` | binary | 주말 여부 |

## 모델 성능

두 모델 모두 `sklearn.pipeline.Pipeline`으로 구성:

| 모델 | CV Balanced Accuracy | Test Weighted F1 |
|------|:---:|:---:|
| RandomForest | 0.7487 | 0.8042 |
| **HistGradientBoosting** | **0.8207** | **0.9090** |

### 클래스별 F1 (HistGradientBoosting)

| Standard | JFK | Unknown | Nassau | Negotiated | Newark |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.93 | 0.80 | 0.73 | 0.63 | 0.19 | 0.11 |

## 주요 발견

- 지리 정보(`DOLocationID` + `PULocationID`)가 전체 변수 중요도의 **64%** 를 차지
- `RatecodeID=99`(Unknown)는 `VendorID=1 ∩ payment_type=1 ∩ passenger_count=1` 패턴으로 완벽히 식별 가능
- 모든 변수가 RatecodeID 그룹 간 통계적으로 유의미한 차이를 보임 (p < 0.001, Welch's t-test)
- 소수 클래스(Newark 0.4%, Negotiated 0.7%)는 데이터 부족으로 성능 저조

## 문서

- [`output/report.md`](output/report.md) — 차트가 포함된 전체 결과 리포트
- [`DATA_CLEANING_NOTES.md`](DATA_CLEANING_NOTES.md) — 데이터 정제, feature 분석, 통계 검정 상세

## 의존성

```
pandas >= 3.0
polars >= 1.0
numpy >= 2.0
seaborn >= 0.13
matplotlib >= 3.7
plotly >= 5.15
scipy >= 1.10
scikit-learn >= 1.3
joblib >= 1.3
pyarrow >= 12.0
kaleido >= 0.2
```
