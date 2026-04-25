# One-Page Project Report (Draft)

## Group Member Names
- Name 1
- Name 2
- Name 3

## Data Source
- CICIoT2023 dataset
- Source URL: https://www.unb.ca/cic/datasets/iotdataset-2023.html

## Approach
This project builds an end-to-end IoT intrusion detection pipeline that:
- validates and samples a large IoT traffic dataset,
- trains a logistic regression baseline and a stronger tabular model,
- serves predictions through a FastAPI API,
- simulates micro-batch inference,
- monitors feature and prediction drift plus alert-rate changes.

## Summary of Methods Used
- Data quality checks: missing values, duplicates, class distribution
- Leakage-aware splitting: time-aware/group-aware if available, stratified fallback
- Preprocessing: numeric imputation/scaling, shared train/inference artifacts
- Modeling: logistic regression baseline + random forest stronger model
- Evaluation: macro-F1, weighted-F1, per-class precision/recall, confusion matrix, benign false positive rate
- Monitoring: PSI for feature drift, JSD for predicted class mix drift, alert-rate spike detection

## Summary Result
- Do not insert fabricated metrics.
- Fill with real evaluation results generated from your run artifacts in `reports/metrics/`.
- Include at minimum:
  - baseline vs stronger macro-F1,
  - per-class recall highlights,
  - benign false-positive rate,
  - key drift observations during stream simulation.
