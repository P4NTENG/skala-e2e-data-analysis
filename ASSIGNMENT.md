# End2End 데이터 분석 프로젝트

## 사용 데이터셋

팀별로 아래 레코드셋 중 적용

- NYC Yellow Taxi : https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet

- Stack Overflow Survey 2024
  - 2024년 데이터 : https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/2024/results.csv
  - 전체 아카이브 폴더 (스키마 설명 등): https://github.com/StackExchange/Survey/tree/main/packages/archive/2024
  - 공식 안내 페이지 (연도별 링크 전부 정리) : https://survey.stackoverflow.co/

- Adult Census Income
  ```python
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
  cols = ["age","workclass","fnlwgt","education","education-num",
  "marital-status","occupation","relationship","race","sex",
  "capital-gain","capital-loss","hours-per-week","native-country","income"]
  df = pd.read_csv(url, header=None, names=cols, na_values=" ?",
  skipinitialspace=True)
  print(df.shape) # (32561, 15)
  ```

## 실습 내용

- 데이터 준비
  - 선택한 데이터셋을 Pandas와 Polars 양쪽으로 로딩하여 결과를 비교하고, 결측치·중복 처리 및 기본 EDA 수행
- 시각화Seaborn으로 정적 차트, Plotly로 인터랙티브 차트 각 1개 이상 작성(분포·상관관계·그룹 비교 중 택일)
- 통계 분석기술통계(평균·표준편차·분위수) 산출, 변수 간 상관계수 계산, scipy.stats.ttest_ind로 ttest 수행 및 p-value 해석
- ML Pipelinesklearn.pipeline.Pipeline으로 전처리 + 모델 학습 구성, 평가 지표 출력, joblib으로 모델 저장
- 자동화·발표분석 결과를 report.md로 자동 생성 팀별 발표
