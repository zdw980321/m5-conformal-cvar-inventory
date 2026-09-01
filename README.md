# Calibrated Demand Intervals and CVaR-Constrained Inventory Allocation

This repository contains the replication code and frozen derived result tables for the rolling-origin M5 retail study reported in *Calibrated Demand Intervals and CVaR-Constrained Inventory Allocation: A Rolling-Origin Study on the M5 Retail Dataset*.

## Scope and data licence

The repository deliberately excludes all raw M5 data. Obtain the M5 Forecasting Accuracy files directly from Kaggle, accept the competition terms, and place the three required CSV files in `research_plan/data/raw/m5/`:

```
calendar.csv
sales_train_evaluation.csv
sell_prices.csv
```

The analysis code was run locally on the public data on 2 September 2026. The included `artifacts/enhanced/run_manifest.json` records source and raw-file hashes, software versions, fixed random seed, selection rule, and run timing. Derived result tables are retained only to audit the paper's reported numbers; they are not a substitute for the source data.

## Environment

Create a clean Python environment and install the pinned minimum dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The allocation model uses HiGHS through `highspy`. The script chooses the locally installed solver through Pyomo; no proprietary optimizer is required.

## Reproduce the study

From the repository root:

```bash
python pipeline/run_m5_enhanced.py
python pipeline/analyse_m5_enhanced.py
python pipeline/prepare_enhanced_tables.py
```

The enhanced experiment selects 600 item-store series by a pre-test, stratified volume rule, evaluates 20 rolling origins, uses a 28-day horizon, performs split conformal calibration, and selects the CVaR coefficient only in a temporally separate policy-validation window. The scripts write new artifacts under `artifacts/enhanced/`.

## Reproducibility boundaries

M5 records retail unit sales and listed prices, not actual procurement, holding, stockout, capacity, or deployed order data. The economic coefficients in the code define counterfactual inventory scenarios, so reported costs are comparative simulated outcomes rather than Walmart financial results. The raw data must not be uploaded to this repository or otherwise redistributed from it.

## Citation

Please cite the associated article once it is published. Before publication, cite this repository by its URL, release date, and the title above.
