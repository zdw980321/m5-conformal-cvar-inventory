"""Generate descriptive and forecast tables without re-solving sensitivity models."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "enhanced"
TABLES = OUT / "tables"
results = pd.read_csv(TABLES / "enhanced_policy_results.csv")
decisions = pd.read_parquet(OUT / "enhanced_decisions.parquet")
forecasts = pd.read_parquet(OUT / "enhanced_forecasts.parquet")
pd.DataFrame([{
    "selected_item_store_series": int(decisions.series_id.nunique()), "categories": int(decisions.cat_id.nunique()),
    "states": int(decisions.state_id.nunique()), "rolling_origins": int(results.origin.nunique()),
    "forecast_horizon_days": 28, "out_of_sample_series_days": int(len(forecasts)),
}]).to_csv(TABLES / "enhanced_data_design.csv", index=False)
rows=[]
for method, column in (("Seasonal-normal", "seasonal"), ("GBM quantile", "p50")):
    error=forecasts.actual-forecasts[column]
    rows.append({"method":method,"MAE":float(abs(error).mean()),"RMSE":float(np.sqrt((error**2).mean())),"mean_error":float(error.mean())})
point=pd.DataFrame(rows)
calibration=pd.read_csv(TABLES / "enhanced_calibration.csv").groupby("method",as_index=False).agg(
    empirical_coverage=("coverage","mean"),mean_interval_width=("mean_width","mean")
)
calibration.merge(point,on="method",how="outer").to_csv(TABLES / "enhanced_forecast_performance.csv",index=False)
