# Employee Attrition Prediction using PySpark

이 프로젝트는 PySpark를 활용해 직원 이직(Attrition)을 예측하는 머신러닝 파이프라인을 구축한 실습입니다. 데이터 전처리부터 모델 학습, 평가, 특성 중요도 분석까지 전 과정을 포함합니다.

---

## 사용 기술 및 라이브러리

- Apache Spark (PySpark)
- pandas
- matplotlib
- seaborn
- scikit-learn (로컬 테스트 참고용)
- Jupyter Notebook

---

## 데이터 설명

- **원본 파일:** IBM HR Analytics Employee Attrition & Performance Dataset  
- **레코드 수:** 1470명 직원  
- **특성 수:** 35개 컬럼  
- **타겟:** `Attrition` (Yes/No)

---

## 분석 및 전처리 흐름

1. **EDA (탐색적 분석)**  
   - 결측치 및 이상치 시각화로 탐색  
   - 수치형, 범주형 변수 분포 확인  

2. **컬럼 제거**  
   - `EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours` 등 예측에 기여하지 않는 고정값/ID 컬럼 제거

3. **범주형 인코딩**  
   - `StringIndexer`로 문자열을 인덱스로 변환  
   - `OneHotEncoder`로 희소행렬 변환 (예: `Gender_ohe`, `JobRole_ohe`, `Department_ohe` 등)

4. **레이블 인코딩**  
   - `Attrition` (Yes/No) → `label` (1/0)로 변환하여 모델 학습에 사용

5. **VectorAssembler 사용**  
   - 수치형 + 인코딩된 범주형 컬럼을 하나의 `features` 벡터로 결합

6. **데이터 분할**  
   - `train_df`, `test_df` = 80:20 비율로 분리

7. **모델 학습**  
   - `LogisticRegression`: 이직 예측에 사용되는 기본 분류 모델  
   - `RandomForestClassifier`: Feature Importance 추출과 더 높은 성능 기대

8. **성능 평가 (AUC)**  
   - `BinaryClassificationEvaluator`를 사용하여 ROC 곡선 아래 면적(AUC) 측정

9. **Feature Importance 시각화**  
   - RandomForest 모델에서 추출한 중요 변수들을 시각화  
   - 중요 상위 변수만으로 다시 Logistic Regression 재학습

---

## 성능 비교 결과

| 모델                   | AUC Score |
| -------------------- | --------- |
| Logistic Regression  | 0.8046    |
| Random Forest        | 0.8064    |
| Logistic w/ Top Vars | 0.7235    |

- Random Forest는 전체 특성을 사용할 때 가장 높은 AUC 점수를 기록했습니다.  
- Feature Importance 기반 상위 변수만으로 학습한 Logistic Regression은 해석력은 높으나 성능은 소폭 하락했습니다.

---

## 중요 특성 Top 10

1. YearsInCurrentRole  
2. YearsSinceLastPromotion  
3. YearsWithCurrManager  
4. WorkLifeBalance  
5. Department_ohe  
6. MonthlyRate  
7. RelationshipSatisfaction  
8. MonthlyIncome  
9. JobRole_ohe  
10. Gender_ohe  

---

## 실행 방법

```bash
# PySpark 환경에서 실행 권장
jupyter notebook 직원퇴사실습_EDA.ipynb
```

---

## 참고사항

- PySpark ML은 Spark 환경 내에서만 정상적으로 작동합니다.
- 특성 선택에 따라 모델 성능이 다르게 나타날 수 있으므로 Feature Engineering이 중요합니다.

---

## 출처

- Dataset: [IBM HR Analytics Employee Attrition & Performance Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)