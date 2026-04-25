# Streaming IoT Intrusion Detection with Drift Monitoring

Production-style ML cybersecurity project for classifying IoT traffic as benign vs malicious (and attack type), then simulating deployment with stream monitoring.

## Resume-ready project description
Built an end-to-end IoT intrusion detection system using scikit-learn and FastAPI with leakage-aware splitting, imbalance-aware training, security-focused metrics (macro-F1, per-class recall, false positives on benign traffic), micro-batch simulation, and drift/alert monitoring (PSI, JSD, alert spikes).

## Why intrusion detection is different from generic classification
- **False positives are expensive**: flagging benign traffic as malicious creates alert fatigue.
- **Class imbalance is severe**: some attack classes are rare, so accuracy alone is misleading.
- **Data drift is expected**: device behavior and attack patterns evolve; monitoring is required.

## Project architecture
- **Data**: load, sample, validate, split (time/group aware when available).
- **Features**: sklearn preprocessing pipeline + label encoding.
- **Models**: Logistic Regression baseline and Random Forest stronger model.
- **Evaluation**: macro/weighted F1, per-class recall, confusion matrix, benign FPR, PR-AUC and ROC-AUC (when valid).
- **Serving**: FastAPI for single/batch inference with class probabilities.
- **Monitoring**: feature drift (PSI), prediction drift (JSD), alert spike detection.

## Repository structure
```text
app/                 # FastAPI app
configs/             # Config + logging
data/                # raw/interim/processed data
models/              # persisted model artifacts
reports/             # metrics + figures
scripts/             # pipeline runners
src/                 # modular source code
tests/               # lightweight tests
prepare_dataset.py   # one-time CICIoT2023 cleaning/sampling helper
```

## Dataset setup (CICIoT2023)
1. Download CICIoT2023 and extract the merged CSV files somewhere on your machine.
2. Run the one-time preparation script to create the cleaned project dataset:
   ```bash
   python prepare_dataset.py --input_dir "path/to/your/merged/csvs" --output "data/raw/ciciot2023_sample.csv"
   ```
   Optional controls:
   ```bash
   python prepare_dataset.py --input_dir "path/to/your/merged/csvs" --output "data/raw/ciciot2023_sample.csv" --rows_per_file 3000 --seed 42
   ```
3. The preparation script reads each CSV, detects the label column, removes empty/duplicate rows, converts infinities to missing values for downstream imputation, drops rows without labels, trims label text, removes known CICIoT2023 ID/metadata leakage columns, drops near-constant columns, and takes a reproducible stratified sample from each file.
4. Keep `configs/config.yaml` aligned with the prepared output:
   - `paths.raw_data_path` should point to the cleaned file. The default is `data/raw/ciciot2023_sample.csv`.
   - `schema.target_column` should match the detected label column printed by `prepare_dataset.py` (usually `Label`).
   - `schema.timestamp_column` / `schema.group_column` should stay `null` for the default prepared dataset, because the helper drops known time/ID metadata columns to reduce leakage.
   - Update `schema.benign_labels` if your export uses a different benign/normal label.
   - Tune `splitting.min_class_count` for the minimum class support kept before splitting.
5. Files under `data/raw/` are intentionally ignored by git, so the cleaned CICIoT2023 CSV stays local. Re-run `prepare_dataset.py` after cloning the repo on another machine before running the pipeline.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run with Docker
```bash
docker pull vitosparadox/cybersec-iot-ids:latest
docker run --rm -p 8000:8000 vitosparadox/cybersec-iot-ids:latest
```

## Run the full pipeline
```bash
python -m scripts.run_all
```

Or run step-by-step:
```bash
python -m scripts.run_data_validation
python -m scripts.run_training
python -m scripts.run_evaluation
python -m scripts.run_stream_simulation
```

## Launch FastAPI
```bash
uvicorn app.main:app --reload --port 8000
```

### API examples
Health:
```bash
curl http://127.0.0.1:8000/health
```

Single prediction:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 0.12, "feature_2": 10.3}}'
```

Batch prediction:
```bash
curl -X POST http://127.0.0.1:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"records": [{"feature_1": 0.12, "feature_2": 10.3}, {"feature_1": 0.2, "feature_2": 9.8}]}'
```

Response includes predicted label, confidence, and top class probabilities.

## Monitoring outputs
Generated under `reports/metrics/`:
- `data_validation_report.json`
- `training_summary.json`
- `baseline_test_metrics.json`
- `stronger_test_metrics.json`
- `model_comparison.csv`
- `stream_monitoring_report.json`
- `stream_predictions.csv`

## Key design choices
- **Random Forest** chosen as stronger model: robust tabular baseline with simple, reproducible training.
- **Shared preprocessing artifact** ensures train/inference consistency.
- **Leakage guardrails**: configured ID/time/group columns are excluded from model features during training.
- **Fallback split strategy**: if no timestamp/group metadata exists, stratified random split is used and documented.
- **Rare-class handling**: extremely rare labels in the sampled CICIoT2023 data are not fabricated or oversampled before evaluation. Classes below `splitting.min_class_count` are filtered before splitting, then listed with row counts in `reports/metrics/data_validation_report.json` and `reports/metrics/training_summary.json`. If the remaining sampled data still cannot support stratification, the pipeline logs a warning and falls back to a random split instead of crashing.

## Interview explanation (simple language)
I treated this like a real SOC-facing pipeline, not just a notebook. First, I validate and sample the dataset so large CICIoT files are manageable and reproducible. I split data in a way that reduces leakage risk, then train a transparent baseline and a stronger model. I evaluate with metrics that actually matter to defenders (macro-F1, per-class recall, and false alarms on benign traffic). Finally, I expose an API for deployment-style inference and simulate streaming batches with drift and alert monitoring so we can catch model degradation early.

## Limitations
- CICIoT column names can vary across exports; configuration fields must be aligned by user.
- Current preprocessing is numeric-only for robustness; categorical support can be added if needed.
- Monitoring is intentionally lightweight (PSI/JSD/thresholds), not enterprise observability tooling.

## Future improvements
- Add calibrated probabilities and threshold tuning for SOC cost trade-offs.
- Add richer time-window drift dashboards.
- Support online/continual retraining triggers.
