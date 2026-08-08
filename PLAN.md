# 프로젝트 구성 계획

## 1. 데이터셋 선택

**NYC Yellow Taxi (2026-05)** 추천:
- `.parquet` 단일 파일로 로딩 간편
- fare, trip distance 등 ML 예측에 적합한 수치 컬럼 다수
- Pandas/Polars 모두 Parquet 네이티브 지원

## 2. 디렉토리 구조

```
skala-e2e-data-analysis/
├── data/                    # 데이터 디렉토리 (.parquet gitignore)
├── output/                  # 결과물 (report.md, model.joblib, figures/)
├── src/
│   ├── 1_data_prep.py       # Pandas/Polars 로딩 비교, 결측치·중복 처리, EDA
│   ├── 2_visualization.py   # Seaborn 정적 + Plotly 인터랙티브 차트
│   ├── 3_statistics.py      # 기술통계, 상관계수, t-test, p-value 해석
│   ├── 4_ml_pipeline.py     # Pipeline 전처리+학습, 평가, joblib 저장
│   └── 5_automation.py      # report.md 자동 생성, 전체 실행
├── requirements.txt
├── .gitignore
├── ASSIGNMENT.md
└── RUBRIC.md
```

## 3. 의존성 (`requirements.txt`)

```
pandas>=2.0
polars>=1.0
numpy>=1.24
seaborn>=0.12
matplotlib>=3.7
plotly>=5.15
scipy>=1.10
scikit-learn>=1.3
joblib>=1.3
pyarrow>=12.0
kaleido>=0.2          # Plotly 정적 이미지 저장용
```

## 4. 각 모듈별 상세 계획

### 1_data_prep.py — 데이터 준비
- Parquet 다운로드/로딩 (Pandas + Polars)
- shape, dtypes, 메모리 사용량 비교 출력
- 결측치 처리, 중복 제거
- `describe()`, `info()` 등 기본 EDA 출력

### 2_visualization.py — 시각화
- **Seaborn:** 요금 분포 히스토그램 + KDE, 거리 vs 요금 상관 산점도 중 1개 이상
- **Plotly:** 시간대별 승차량 히트맵, 요금 분포 interactive histogram 등 1개 이상
- 제목, 축 레이블 포함, `output/figures/`에 저장

### 3_statistics.py — 통계 분석
- 평균, 표준편차, 분위수 산출
- 수치 컬럼 간 상관계수 히트맵
- `scipy.stats.ttest_ind`로 두 그룹 비교 (예: weekday vs weekend 요금)
- p-value 해석 코멘트 포함

### 4_ml_pipeline.py — ML Pipeline
- 타겟: `total_amount` 또는 `trip_duration` 예측
- Pipeline 구성: `StandardScaler` + `OneHotEncoder` (범주형) + `ColumnTransformer` → `RandomForestRegressor`
- 평가 지표: RMSE, MAE, R²
- `joblib.dump()`로 모델 저장 → `output/model.joblib`

### 5_automation.py — 자동화
- 1~4 모듈을 순차 실행하는 메인 러너
- 모든 출력을 `output/report.md`로 자동 조합
- Markdown 포맷으로 텍스트 결과 + 이미지 링크 포함

## 5. GitHub 커밋 계획

| # | 커밋 메시지 | 내용 |
|---|------------|------|
| 1 | `chore: project setup with requirements and gitignore` | requirements.txt, .gitignore, 디렉토리 구조 |
| 2 | `feat: data preparation with Pandas and Polars` | 1_data_prep.py |
| 3 | `feat: visualization with Seaborn and Plotly` | 2_visualization.py + output/figures/ |
| 4 | `feat: statistical analysis with t-test` | 3_statistics.py |
| 5 | `feat: ML pipeline with sklearn Pipeline and joblib` | 4_ml_pipeline.py + output/model.joblib |
| 6 | `feat: automated report generation` | 5_automation.py + output/report.md |
| 7 | `docs: final report and code improvement notes` | 코드 품질 의견, 실행 스크린샷 |
