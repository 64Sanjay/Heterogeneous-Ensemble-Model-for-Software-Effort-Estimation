# Software Effort Estimation - Experiment Report

Generated: 2026-01-17 13:15:07

---

## Dataset: NASA93

### Best Individual Model
- **Model**: ANN
- **MAE**: 373.27
- **MMRE**: 1.1821
- **PRED(0.25)**: 0.1935

### Best Ensemble Model
- **Model**: CBR_COCOMO_XGBoost
- **MAE**: 419.58
- **MMRE**: 1.6605
- **PRED(0.25)**: 0.1613

### Improvement: **12.41%** (Individual is better)

### All Individual Models

| Model | MAE | MMRE | PRED(0.25) |
|-------|-----|------|------------|
| CBR | 436.20 | 1.7148 | 0.1290 |
| COCOMO | 474.45 | 2.2044 | 0.1290 |
| ANN | 373.27 | 1.1821 | 0.1935 |
| KNN | 458.62 | 1.7344 | 0.1935 |
| XGBoost | 431.24 | 1.6913 | 0.2796 |
| SVR | 456.69 | 1.7498 | 0.1613 |
| LinearRegression | 424.11 | 2.4676 | 0.2258 |

### All Ensemble Models

| Model | MAE | MMRE | PRED(0.25) |
|-------|-----|------|------------|
| CBR_COCOMO_ANN | 438.92 | 1.5949 | 0.2258 |
| CBR_COCOMO_KNN | 456.67 | 1.7728 | 0.1613 |
| CBR_COCOMO_XGBoost | 419.58 | 1.6605 | 0.1613 |
| CBR_COCOMO_SVR | 443.48 | 1.8216 | 0.1720 |

---

## Dataset: COCOMO81

### Best Individual Model
- **Model**: XGBoost
- **MAE**: 471.10
- **MMRE**: 1.1282
- **PRED(0.25)**: 0.2222

### Best Ensemble Model
- **Model**: CBR_COCOMO_XGBoost
- **MAE**: 524.33
- **MMRE**: 1.2554
- **PRED(0.25)**: 0.2540

### Improvement: **11.30%** (Individual is better)

### All Individual Models

| Model | MAE | MMRE | PRED(0.25) |
|-------|-----|------|------------|
| CBR | 601.93 | 4.4257 | 0.1587 |
| COCOMO | 564.26 | 1.6291 | 0.0952 |
| ANN | 537.34 | 2.1583 | 0.2540 |
| KNN | 591.19 | 3.4547 | 0.0952 |
| XGBoost | 471.10 | 1.1282 | 0.2222 |
| SVR | 618.91 | 2.3097 | 0.0794 |
| LinearRegression | 1051.23 | 19.9126 | 0.0794 |

### All Ensemble Models

| Model | MAE | MMRE | PRED(0.25) |
|-------|-----|------|------------|
| CBR_COCOMO_ANN | 548.50 | 1.5396 | 0.1587 |
| CBR_COCOMO_KNN | 551.57 | 3.2427 | 0.1746 |
| CBR_COCOMO_XGBoost | 524.33 | 1.2554 | 0.2540 |
| CBR_COCOMO_SVR | 593.63 | 1.6045 | 0.0794 |

---

## Summary

| Dataset | Best Individual | MAE | Best Ensemble | MAE | Improvement |
|---------|-----------------|-----|---------------|-----|-------------|
| nasa93 | ANN | 373.27 | CBR_COCOMO_XGBoost | 419.58 | -12.41% |
| cocomo81 | XGBoost | 471.10 | CBR_COCOMO_XGBoost | 524.33 | -11.30% |