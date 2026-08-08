# 데이터 정제 내역 분석

`yellow_tripdata_2026-05.parquet` (원본) → `yellow_tripdata_2026-05_clean.parquet` (정제본) 변환 과정에서 확인된 사항.

## 1. 기본 정보 비교

| 항목 | 원본 | Clean |
|------|------|-------|
| 행 수 | 4,090,836 | 3,021,962 |
| 컬럼 수 | 20 | 20 |
| 전체 결측치 | 4,776,855 | 0 |
| 제거된 행 | — | 1,068,874 (26.1%) |

## 2. 결측치(NaN) 포함 컬럼

원본에서 아래 5개 컬럼이 **동일한 955,371개 행**에서 일괄 결측:

| 컬럼 | NaN 개수 |
|------|----------|
| `passenger_count` | 955,371 |
| `RatecodeID` | 955,371 |
| `store_and_fwd_flag` | 955,371 |
| `congestion_surcharge` | 955,371 |
| `Airport_fee` | 955,371 |

→ 해당 행들은 NaN을 특정 값으로 대체하지 않고 **전체 삭제**됨.

## 3. 추가 필터링된 행 (~113,503행)

결측치 행 제거 후에도 약 113,503행이 추가 제거됨. 추정되는 필터 조건:

| 조건 | 해당 행 수 (원본 Non-Null 기준) |
|------|-------------------------------|
| `trip_distance <= 0` | 37,065 |
| `fare_amount <= 0` | 16,128 |
| `total_amount <= 0` | 15,370 |
| `passenger_count == 0` | 12,533 |
| `RatecodeID == 99` (Unknown) | 일부 제거 |

※ 조건 간 중복이 있어 합계는 113,503과 일치하지 않음.

## 4. 값 매핑 사례 (NaN이 아닌 이상치 정제)

### 4-1. `congestion_surcharge`

| 값 | 원본 (Non-Null) | Clean | 변환 |
|----|-----------------|-------|------|
| `2.5` | 2,774,931 | 2,694,788 | 유지 |
| `0.0` | 348,658 | 327,174 | 유지 |
| `-2.5` | 11,876 | 0 | → `0.0` 또는 `2.5`로 매핑 + 일부 행 제거 |

### 4-2. `Airport_fee`

| 값 | 원본 (Non-Null) | Clean | 변환 |
|----|-----------------|-------|------|
| `0.0` | 2,878,494 | 2,775,204 | 유지 |
| `2.0` | 252,429 | 246,758 | 유지 |
| `-2.0` | 2,327 | 0 | → `0.0`으로 매핑 |
| `7.0` | 1,391 | 0 | → `0.0`으로 매핑 |
| `5.0` | 404 | 0 | → `0.0`으로 매핑 |
| `1.75` | 196 | 0 | → `0.0`으로 매핑 |
| 기타 (6.75, 20.0, 27.0 등) | 소수 | 0 | → `0.0`으로 매핑 |

## 5. `RatecodeID=99`

- 원본 Non-Null에 140,897건 존재
- Clean에도 140,601건 존재 (일부만 제거됨)
- NYC TLC의 공식 코드 `99 = Unknown` 으로, NaN 대체값이 아닌 **유효한 도메인 값**

## 6. 결론

1. NaN 결측치는 특정 값으로 **매핑되지 않고 해당 행 전체가 삭제**됨
2. 추가적인 데이터 품질 필터링으로 zero/negative fare, zero distance 등 불량 행 제거
3. `congestion_surcharge`와 `Airport_fee` 컬럼의 이상치 값은 정상 범위로 **매핑/클리핑**됨
4. 최종 Clean 데이터는 3,021,962행 × 20컬럼, 결측치 없음

---

# RatecodeID 예측 Feature 분석

## 1. RatecodeID 분포 (6-class 분류)

| RC | 의미 | 건수 | 비율 |
|----|------|------|------|
| 1 | Standard rate | 2,750,151 | 91.0% |
| 2 | JFK (flat fare) | 90,510 | 3.0% |
| 3 | Newark | 11,491 | 0.4% |
| 4 | Nassau/Westchester | 8,103 | 0.3% |
| 5 | Negotiated fare | 21,106 | 0.7% |
| 99 | Unknown | 140,601 | 4.7% |

→ 심한 클래스 불균형. RC=1이 91%로 다수 클래스.

## 2. RatecodeID별 특성 패턴

| 특성 | RC=1 (Standard) | RC=2 (JFK) | RC=3 (Newark) | RC=4 (Nassau) | RC=5 (Negotiated) | RC=99 (Unknown) |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| `trip_distance` (mean) | 2.6 mi | 17.7 mi | 13.5 mi | 20.5 mi | 7.4 mi | 8.4 mi |
| `passenger_count` (mean) | 1.25 | 1.46 | 1.49 | 1.30 | 1.74 | 1.00 |
| **Top PULocationID** | 237 | **132 (JFK, 75%)** | **138 (Newark, 24%)** | 132 (JFK, 56%) | 132 (JFK) | 76 |
| **Top DOLocationID** | 236 | 132 (JFK) | **1 (EWR, 46%)** | **265 (Nassau, 88%)** | 132 (JFK) | 76 |
| `VendorID=1` 비율 | 18.3% | 13.6% | 13.2% | 10.3% | 10.1% | **99.98%** |
| `payment_type=1` 비율 | 87.2% | 86.0% | 76.1% | 76.7% | 90.2% | **99.99%** |
| `store_and_fwd_flag=Y` | 0.11% | 0.11% | 0.41% | 0.10% | 0.09% | **0%** |
| `hour` (mean) | 14.5 | 14.7 | 13.4 | 14.4 | 12.9 | **11.9** |
| `Airport_fee=2` 비율 | 6.1% | **73.8%** | 34.5% | **70.5%** | 18.8% | 0% |
| `congestion_surcharge=2.5` | 94.7% | 92.3% | **0.1% (NJ)** | 11.5% | 31.2% | 0.1% |

## 3. 추천 Feature (X)

### 포함 (강한 신호)

| Feature | 타입 | 선정 근거 |
|---------|------|-----------|
| `PULocationID` | categorical (1~265) | RC=2는 75%가 JFK(132), RC=3는 24%가 Newark(138)에서 픽업 — **지리적 위치가 가장 강력한 신호** |
| `DOLocationID` | categorical (1~265) | RC=4는 88%가 Nassau(265), RC=3는 46%가 EWR(1)로 하차 |
| `VendorID` | binary (1,2) | RC=99는 99.98%가 VendorID=1로 완전 분리에 가까움 |
| `payment_type` | categorical (1~4) | RC=99는 99.99%가 payment_type=1 |
| `passenger_count` | numeric (1~8) | RC=99는 100%가 1명, RC=5는 평균 1.74명으로 가장 높음 |

### 포함 (보조 신호)

| Feature | 타입 | 선정 근거 |
|---------|------|-----------|
| `store_and_fwd_flag` | binary (N/Y) | RC=99는 Y가 0건 — 완전 분리 특성 |
| `hour` (pickup 시간) | numeric (0~23) | RC=99는 평균 11.9시로 아침/점심 집중, 타 클래스는 오후 피크 |
| `day_of_week` | numeric (0~6) | 요일별 패턴 차이 존재 |
| `is_weekend` | binary | 주말/평일 구분 추가 가능 |

### 제외 (데이터 누수 — RatecodeID에 의해 결정되는 값)

| Feature | 제외 근거 |
|---------|-----------|
| `fare_amount`, `total_amount`, `tip_amount` | 요금 체계 자체가 RatecodeID에 의해 결정됨 |
| `Airport_fee`, `congestion_surcharge` | 공항/혼잡 할증료는 RatecodeID에 종속 |
| `extra`, `mta_tax`, `tolls_amount`, `improvement_surcharge`, `cbd_congestion_fee` | 모두 RatecodeID에 따라 부과 방식이 달라짐 |
| `tpep_dropoff_datetime` | 예측 시점에 알 수 없는 미래 정보 |

## 4. RC=99(Unknown) 특이사항

RC=99는 `VendorID=1 & payment_type=1 & passenger_count=1.0 & store_and_fwd_flag=N` 패턴이 거의 완벽하게 일치하여, 이 4개 feature만으로도 다른 클래스와 쉽게 분리됨. VendorID=2인 데이터에서는 RC=99가 거의 등장하지 않으므로, VendorID=1에서만 Unknown rate가 발생하는 데이터 수집 패턴으로 추정됨.

## 5. 모델 설계 고려사항

- **클래스 불균형**: RC=1이 91%로 `class_weight='balanced'` 또는 SMOTE 등 오버샘플링 필요
- **고차원 범주형**: `PULocationID`(265개), `DOLocationID`(265개)는 OneHotEncoding 시 530차원으로 증가 → TargetEncoder 또는 OrdinalEncoder + Tree 모델 권장
- **소수 클래스**: RC=3(0.4%), RC=4(0.3%)는 macro-F1 기준 성능이 낮을 수 있음 → stratified CV 필수
- **평가 지표**: Accuracy는 의미 없음 (91% baseline) → macro-F1, weighted-F1, confusion matrix 위주로 평가

---

# RatecodeID 통계 분석

## 1. 기술통계 (RatecodeID별)

### `trip_distance`

| RatecodeID | mean | std | min | 25% | 50% | 75% | max |
|------------|------|-----|-----|-----|-----|-----|-----|
| 1 (Standard) | 2.59 | 2.96 | 0.01 | 0.97 | 1.60 | 2.80 | 97.78 |
| 2 (JFK) | 17.66 | 3.58 | 0.01 | 16.88 | 17.70 | 18.87 | 86.60 |
| 3 (Newark) | 13.50 | 7.27 | 0.01 | 8.75 | 14.70 | 17.97 | 85.74 |
| 4 (Nassau) | 20.48 | 12.01 | 0.02 | 11.40 | 18.24 | 26.72 | 93.18 |
| 5 (Negotiated) | 7.37 | 8.45 | 0.01 | 0.55 | 5.12 | 12.09 | 99.21 |
| 99 (Unknown) | 8.40 | 5.79 | 0.01 | 3.80 | 7.40 | 11.80 | 97.80 |

- RC=2(JFK)는 std=3.58로 분산이 가장 좁음 → flat fare로 거리 대비 요금이 고정되기 때문
- RC=4(Nassau)는 std=12.01로 분산이 가장 넓음 → 다양한 장거리 목적지 포함

### `passenger_count`

| RatecodeID | mean | std | min | 50% | max |
|------------|------|-----|-----|-----|-----|
| 1 (Standard) | 1.25 | 0.63 | 1 | 1 | 6 |
| 2 (JFK) | 1.46 | 0.73 | 1 | 1 | 6 |
| 3 (Newark) | 1.49 | 0.84 | 1 | 1 | 6 |
| 4 (Nassau) | 1.30 | 0.69 | 1 | 1 | 6 |
| **5 (Negotiated)** | **1.74** | **1.08** | **1** | **1** | **8** |
| **99 (Unknown)** | **1.00** | **0.00** | **1** | **1** | **1** |

- RC=5는 유일하게 평균 1.74명, max=8 → 단체 negotiated fare
- RC=99는 std=0, 모든 행이 정확히 1명 → 구조적 데이터 패턴

### `hour` (pickup 시간)

| RatecodeID | mean | std | 특징 |
|------------|------|-----|------|
| 1 (Standard) | 14.5 | 5.7 | 오후 피크 |
| 2 (JFK) | 14.7 | 5.4 | 오후 피크 |
| 3 (Newark) | 13.4 | 5.3 | 이른 오후 |
| 4 (Nassau) | 14.4 | 6.4 | 넓은 분포 |
| 5 (Negotiated) | 12.9 | 5.8 | 이른 오후 |
| **99 (Unknown)** | **11.9** | **4.4** | **오전~정오 집중** |

## 2. 상관계수 (수치 변수 간)

|  | trip_distance | passenger_count | hour |
|--|:---:|:---:|:---:|
| **trip_distance** | 1.00 | 0.02 | -0.03 |
| **passenger_count** | — | 1.00 | 0.04 |
| **hour** | — | — | 1.00 |

→ 변수 간 선형 상관관계가 매우 약하므로, 각 변수가 독립적인 예측 신호로 작용.

### ANOVA Eta-squared (RatecodeID 그룹 간 설명력)

| 변수 | η² | 해석 |
|------|:---:|------|
| `trip_distance` | **0.453** | RatecodeID 분산의 45.3% 설명 — 가장 강력한 변수 |
| `passenger_count` | 0.015 | 1.5% 설명 |
| `hour` | 0.010 | 1.0% 설명 |

## 3. t-test (scipy.stats.ttest_ind)

Welch's t-test (등분산 가정 없음). 아래 주요 그룹 비교:

| 비교 | 변수 | t-statistic | p-value | 해석 |
|------|------|:---:|:---:|------|
| Standard vs JFK | trip_distance | **-1251.4** | <0.001 | JFK가 약 15.1mi 더 김 — 압도적 차이 |
| Standard vs Unknown | trip_distance | **-373.5** | <0.001 | Unknown이 약 5.8mi 더 김 |
| JFK vs Newark | trip_distance | **60.4** | <0.001 | JFK가 Newark보다 약 4.2mi 더 김 |
| Standard vs Negotiated | trip_distance | **-82.1** | <0.001 | Negotiated가 약 4.8mi 더 김 |
| Standard vs Unknown | passenger_count | **649.0** | <0.001 | Standard가 0.25명 더 많음 (RC=99는 항상 1명) |
| Standard vs JFK | passenger_count | **-85.4** | <0.001 | JFK가 0.21명 더 많음 |
| Standard vs Unknown | hour | **214.0** | <0.001 | Standard가 약 2.6시간 더 늦음 |
| Standard vs Negotiated | hour | **41.0** | <0.001 | Standard가 약 1.6시간 더 늦음 |
| JFK vs Newark | hour | **22.9** | <0.001 | JFK가 약 1.2시간 더 늦음 |

### p-value 해석

모든 t-test 결과에서 **p < 0.001**로 귀무가설(두 그룹 간 평균 차이 없음)을 기각.

- `trip_distance`의 t-statistic이 -1251.4로 가장 큼 → RatecodeID 간 거리 차이가 통계적으로 매우 유의미
- `passenger_count`의 Standard vs Unknown 비교에서 t=649.0 → RC=99의 100% 1명 패턴이 극단적인 차이를 만듦
- `hour` 또한 모든 비교에서 유의미한 차이를 보여, 시간대가 RatecodeID 예측에 유효한 보조 변수임을 확인

## 4. 범주형 변수 분포 (RatecodeID별)

### `VendorID`

| RatecodeID | VendorID=1 | VendorID=2 |
|------------|:---:|:---:|
| 1 (Standard) | 18.3% | 81.7% |
| 2 (JFK) | 13.6% | 86.4% |
| 3 (Newark) | 13.2% | 86.8% |
| 4 (Nassau) | 10.3% | 89.7% |
| 5 (Negotiated) | 10.1% | 89.9% |
| **99 (Unknown)** | **100.0%** | **0.0%** |

### `payment_type`

| RatecodeID | 1 (Credit) | 2 (Cash) | 3 | 4 |
|------------|:---:|:---:|:---:|:---:|
| 1 (Standard) | 87.2% | 12.2% | 0.2% | 0.4% |
| 2 (JFK) | 86.0% | 13.2% | 0.2% | 0.6% |
| 3 (Newark) | 76.1% | 22.0% | 0.8% | 1.1% |
| 4 (Nassau) | 76.7% | 21.8% | 0.4% | 1.1% |
| 5 (Negotiated) | 90.2% | 8.7% | 0.3% | 0.8% |
| **99 (Unknown)** | **100.0%** | **0.0%** | **0.0%** | **0.0%** |

- Newark/Nassau는 현금 결제 비율(22%)이 타 클래스(9~13%)보다 높음
- RC=99는 결제 수단이 100% Credit

### `is_weekend`

| RatecodeID | 평일 | 주말 |
|------------|:---:|:---:|
| 1 (Standard) | 70.3% | 29.7% |
| 5 (Negotiated) | 63.1% | 36.9% |
| **99 (Unknown)** | **83.8%** | **16.2%** |

- RC=99는 주말 비율이 16.2%로 가장 낮음 → 평일 업무 시간대 집중

## 5. 통계 분석 결론

1. `trip_distance`가 RatecodeID 분류에 가장 강력한 변수 (η²=0.45, 모든 t-test p<0.001)
2. RC=99(Unknown)는 `passenger_count=1`, `VendorID=1`, `payment_type=1`, `store_and_fwd_flag=N`이 거의 100% 일치하는 구조적 패턴 — 특정 Vendor의 데이터 수집 한계로 추정
3. 모든 분석 변수가 RatecodeID 그룹 간 **통계적으로 유의미한 차이**(p<0.001)를 보여 ML 예측 변수로서 타당성 확보
4. 수치 변수 간 상관계수가 낮아(최대 0.04), 다중공선성 문제 없이 독립적 신호로 활용 가능
