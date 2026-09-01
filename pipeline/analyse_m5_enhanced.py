"""Inference, stratification, and ex-post cost sensitivity for enhanced M5 run."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from run_m5_study import Economics, SCENARIO_LEVELS, aggregate_daily_quantiles, evaluate_decision, solve_allocation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "enhanced"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SEED = 20260901


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 50_000) -> tuple[float, float]:
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def scenario(test: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "Seasonal-normal":
        return aggregate_daily_quantiles(test, "seasonal_lower", "seasonal", "seasonal_upper", 0.05, 0.95)
    return aggregate_daily_quantiles(test, "cqr_lower", "p50", "cqr_upper", 0.05, 0.95)


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = pd.read_csv(TABLES / "enhanced_policy_results.csv")
    selected = pd.read_csv(TABLES / "enhanced_nested_selection.csv")
    decisions = pd.read_parquet(OUT / "enhanced_decisions.parquet")
    forecasts = pd.read_parquet(OUT / "enhanced_forecasts.parquet")
    pd.DataFrame([{
        "selected_item_store_series": int(decisions.series_id.nunique()), "categories": int(decisions.cat_id.nunique()),
        "states": int(decisions.state_id.nunique()), "rolling_origins": int(results.origin.nunique()),
        "forecast_horizon_days": 28, "out_of_sample_series_days": int(len(forecasts)),
    }]).to_csv(TABLES / "enhanced_data_design.csv", index=False)
    point_rows = []
    for method, column in (("Seasonal-normal", "seasonal"), ("GBM quantile", "p50")):
        error = forecasts.actual - forecasts[column]
        point_rows.append({"method": method, "MAE": float(abs(error).mean()), "RMSE": float(np.sqrt((error**2).mean())), "mean_error": float(error.mean())})
    point = pd.DataFrame(point_rows)
    calibration = pd.read_csv(TABLES / "enhanced_calibration.csv").groupby("method", as_index=False).agg(
        empirical_coverage=("coverage", "mean"), mean_interval_width=("mean_width", "mean")
    )
    calibration.merge(point, on="method", how="outer").to_csv(TABLES / "enhanced_forecast_performance.csv", index=False)
    baseline = results.loc[results.policy.eq("Seasonal-normal"), ["origin", "realized_total_cost"]].rename(columns={"realized_total_cost": "seasonal_cost"})
    rows = []
    for policy, part in results.groupby("policy", sort=True):
        pair = part.merge(baseline, on="origin", how="left")
        difference = pair.realized_total_cost.to_numpy() - pair.seasonal_cost.to_numpy()
        low, high = bootstrap_ci(difference, rng)
        wins = int((difference < 0).sum())
        n = len(difference)
        exact_sign = sum(float(math.comb(n, k)) for k in range(wins, n + 1)) / 2**n if wins >= n / 2 else np.nan
        rows.append({"policy": policy, "mean_total_cost": float(part.realized_total_cost.mean()),
                     "mean_fill_rate": float(part.fill_rate.mean()), "mean_difference_vs_seasonal": float(difference.mean()),
                     "mean_reduction_pct_vs_seasonal": float((100 * -difference / pair.seasonal_cost).mean()),
                     "bootstrap_95_ci_low": low, "bootstrap_95_ci_high": high,
                     "origin_wins": wins, "origins": n, "one_sided_sign_p": exact_sign})
    pd.DataFrame(rows).to_csv(TABLES / "enhanced_inference.csv", index=False)

    group = decisions.groupby(["origin", "policy", "state_id", "cat_id"], as_index=False).agg(
        actual=("actual", "sum"), fulfilled=("fulfilled", "sum"), realized_cost=("realized_cost", "sum"),
        shortage_units=("shortage_units", "sum"), overage_units=("overage_units", "sum"),
    )
    group["fill_rate"] = group.fulfilled / group.actual
    seasonal = group.loc[group.policy.eq("Seasonal-normal"), ["origin", "state_id", "cat_id", "realized_cost"]].rename(columns={"realized_cost": "seasonal_cost"})
    group = group.merge(seasonal, on=["origin", "state_id", "cat_id"], how="left")
    group["cost_reduction_pct"] = 100 * (group.seasonal_cost - group.realized_cost) / group.seasonal_cost
    stratified = group.loc[group.policy.eq("Conformal-CVaR")].groupby(["state_id", "cat_id"], as_index=False).agg(
        mean_cost_reduction_pct=("cost_reduction_pct", "mean"), mean_fill_rate=("fill_rate", "mean"),
        wins=("cost_reduction_pct", lambda x: int((x > 0).sum())), observations=("cost_reduction_pct", "size"),
    )
    stratified.to_csv(TABLES / "enhanced_stratified_inference.csv", index=False)

    # Cost sensitivity preserves each origin's historically selected CVaR weight.
    # It is explicitly labelled post-selection sensitivity, not retuning.
    sensitivity_rows = []
    for origin, test in forecasts.groupby("origin", sort=True):
        test = test.copy()
        details = decisions.loc[(decisions.origin == origin) & (decisions.policy == "Conformal-CVaR"), ["series_id", "sell_price"]].drop_duplicates("series_id")
        actual = test.groupby("series_id", as_index=False).agg(actual=("actual", "sum")).merge(details, on="series_id", how="left").sort_values("series_id")
        chosen = float(selected.loc[selected.origin.eq(origin), "selected_cvar_weight"].iloc[0])
        distributions = {name: scenario(test, name) for name in ("Seasonal-normal", "Conformal-CVaR")}
        for shortage in (4.0, 8.0, 12.0):
            econ = Economics(shortage_ratio=shortage)
            for policy, values in distributions.items():
                values = actual[["series_id"]].merge(values, on="series_id", how="left")
                scenarios = values[[f"q_{int(x*100):02d}" for x in SCENARIO_LEVELS]].to_numpy()
                weight = chosen if policy == "Conformal-CVaR" else 0.0
                q, y, diagnostic = solve_allocation(scenarios, actual.sell_price.to_numpy(), econ, weight)
                evaluated = evaluate_decision(q, y, actual.actual.to_numpy(), actual.sell_price.to_numpy(), econ)
                sensitivity_rows.append({"origin": origin, "policy": policy, "shortage_ratio": shortage,
                                         "selected_cvar_weight": weight, **diagnostic, **evaluated})
    sensitivity = pd.DataFrame(sensitivity_rows)
    base = sensitivity.loc[sensitivity.policy.eq("Seasonal-normal"), ["origin", "shortage_ratio", "realized_total_cost"]].rename(columns={"realized_total_cost": "base_cost"})
    conformal = sensitivity.loc[sensitivity.policy.eq("Conformal-CVaR")].merge(base, on=["origin", "shortage_ratio"])
    conformal["cost_reduction_pct"] = 100 * (conformal.base_cost - conformal.realized_total_cost) / conformal.base_cost
    summary = conformal.groupby("shortage_ratio", as_index=False).agg(
        mean_cost_reduction_pct=("cost_reduction_pct", "mean"),
        mean_fill_rate=("fill_rate", "mean"), wins=("cost_reduction_pct", lambda x: int((x > 0).sum())),
    )
    sensitivity.to_csv(TABLES / "enhanced_postselection_sensitivity.csv", index=False)
    summary.to_csv(TABLES / "enhanced_postselection_sensitivity_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.boxplot(data=group.loc[group.policy.isin(["Seasonal-normal", "Conformal-CVaR"])], x="cat_id", y="cost_reduction_pct", hue="policy", ax=ax)
    ax.axhline(0, color="#4A4A4A", linewidth=1.0); ax.set(xlabel="Product category", ylabel="Cost reduction versus seasonal (%)")
    ax.legend(frameon=False, title="")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_enhanced_category_heterogeneity.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.barplot(data=summary, x="shortage_ratio", y="mean_cost_reduction_pct", color="#C97A1A", ax=ax)
    ax.axhline(0, color="#4A4A4A", linewidth=1.0); ax.set(xlabel="Shortage-cost ratio", ylabel="Mean cost reduction versus seasonal (%)")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_enhanced_cost_sensitivity.png", dpi=600, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
