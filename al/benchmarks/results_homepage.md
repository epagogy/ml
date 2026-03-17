# Algo Landscape Benchmark

**1,314 successful trials** across 9 datasets, 11 algorithm families, 2 engines (Rust vs sklearn).
50/50 holdout methodology. Fixed seed=42.

## Engine Speed Comparison (mean fit time)

| Algorithm | Rust (ms) | sklearn (ms) | Speedup |
|-----------|----------:|-------------:|--------:|
| random_forest | 19.2 | 449.6 | **23.4x** |
| extra_trees | 12.1 | 151.2 | **12.5x** |
| svm | 40.2 | 206.9 | **5.1x** |
| gradient_boosting | 285.9 | 1273.3 | **4.5x** |
| adaboost | 176.3 | 607.6 | **3.4x** |
| decision_tree | 9.1 | 23.9 | **2.6x** |
| linear | 3.1 | 4.0 | 1.3x |
| naive_bayes | 6.6 | 8.1 | 1.2x |
| elastic_net | 4.0 | 4.9 | 1.2x |
| knn | 7.9 | 9.1 | 1.2x |
| logistic | 10.2 | 11.7 | 1.1x |

## Best Config Per Dataset (Rust engine)

### Classification

| Dataset | Algorithm | Accuracy | F1 | Time |
|---------|-----------|:--------:|:--:|-----:|
| cancer | logistic | 0.968 | 0.968 | 5.0ms |
| churn | adaboost | 0.808 | 0.796 | 233.8ms |
| fraud | random_forest | 0.991 | 0.991 | 22.4ms |
| iris | random_forest | 0.960 | 0.960 | 3.5ms |
| titanic | gradient_boosting | 0.814 | 0.812 | 31.4ms |
| wine | svm | 0.989 | 0.989 | 3.5ms |

### Regression

| Dataset | Algorithm | R2 | RMSE | Time |
|---------|-----------|:--:|-----:|-----:|
| diabetes | svm | 0.516 | 52.31 | 2.7ms |
| houses | gradient_boosting | 0.829 | 0.47 | 1002.4ms |
| tips | random_forest | 0.424 | 1.09 | 4.0ms |

## Accuracy Parity

- Classification accuracy |diff|: **median=0.002**, max=0.173
- Regression R2 |diff|: **median=0.003**, max=0.582

Max gaps are SVM (linear SMO vs libsvm) and multiclass logistic. Median parity is excellent.

## Errors

12/1326 failed (0.9%). All `adaboost+sklearn` on NaN datasets. Rust handles NaN natively; sklearn requires complete data.

## Methodology

- `ml.split(df, target, ratio=(0.49, 0.01, 0.5), seed=42)` — train on `.dev`, evaluate on `.test`
- Metrics computed directly from predictions (accuracy, weighted F1, R2, RMSE, MAE)
- Full hyperparameter grid (e.g. GBT: 3 n_estimators x 3 learning_rate x 3 max_depth = 27 configs x 2 engines)
- Hardware: Linux x86_64, AMD Ryzen 9 9900X, Python 3.12, mlw 1.1.2, sklearn 1.8
- Raw data: `landscape.db` (SQLite)
