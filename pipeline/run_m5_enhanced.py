"""Nested rolling-origin evaluation for the strengthened M5 manuscript.

This experiment expands the panel and selects the CVaR weight inside each
origin using a disjoint historical policy-validation window. Test outcomes are
never used to select a risk weight or calibrate a prediction interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from run_m5_study import (
    Economics, FEATURES, HORIZON, SEED, aggregate_daily_quantiles,
    evaluate_decision, fit_quantile_models, read_and_select,
    recursive_forecast, solve_allocation, static_features,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "enhanced"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
N_PER_CELL = 20
ORIGINS = list(range(1353, 1886, 28))
CVaR_GRID = (0.00, 0.25, 0.50, 0.75, 1.00)
POLICIES = ("Seasonal-normal", "GBM quantile", "Conformal-CVaR")


def qhat(scores: pd.Series, coverage: float = 0.90) -> float:
    level = min(1.0, math.ceil((len(scores) + 1) * coverage) / len(scores))
    return max(0.0, float(np.quantile(scores, level, method="higher")))


def cqr(frame: pd.DataFrame, width: float) -> pd.DataFrame:
    out = frame.copy()
    out["cqr_lower"] = np.maximum(0.0, out.p10 - width)
    out["cqr_upper"] = out.p90 + width
    return out


def seasonal_intervals(frame: pd.DataFrame, residual_scale: float) -> pd.DataFrame:
    out = frame.copy()
    out["seasonal_lower"] = np.maximum(0.0, out.seasonal - 1.645 * residual_scale)
    out["seasonal_upper"] = out.seasonal + 1.645 * residual_scale
    return out


def aggregate_and_price(panel: pd.DataFrame, frame: pd.DataFrame, decision_t: int) -> pd.DataFrame:
    aggregate = frame.groupby("series_id", as_index=False).agg(actual=("actual", "sum"))
    prices = panel.loc[panel.t.eq(decision_t), ["series_id", "sell_price"]].drop_duplicates("series_id")
    return aggregate.merge(prices, on="series_id", how="left").sort_values("series_id").reset_index(drop=True)


def scenarios(frame: pd.DataFrame, aggregate: pd.DataFrame, policy: str) -> np.ndarray:
    if policy == "Seasonal-normal":
        values = aggregate_daily_quantiles(frame, "seasonal_lower", "seasonal", "seasonal_upper", 0.05, 0.95)
    elif policy == "GBM quantile":
        values = aggregate_daily_quantiles(frame, "p10", "p50", "p90", 0.10, 0.90)
    elif policy == "Conformal-CVaR":
        values = aggregate_daily_quantiles(frame, "cqr_lower", "p50", "cqr_upper", 0.05, 0.95)
    else:
        raise ValueError(policy)
    values = aggregate[["series_id"]].merge(values, on="series_id", how="left")
    return values[["q_05", "q_20", "q_50", "q_80", "q_95"]].to_numpy()


def evaluate_policy(
    policy: str,
    demand_scenarios: np.ndarray,
    aggregate: pd.DataFrame,
    econ: Economics,
    cvar_weight: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    q, y, diagnostics = solve_allocation(demand_scenarios, aggregate.sell_price.to_numpy(), econ, cvar_weight)
    evaluation = evaluate_decision(q, y, aggregate.actual.to_numpy(), aggregate.sell_price.to_numpy(), econ)
    detail = aggregate[["series_id", "actual", "sell_price"]].copy()
    detail["order_quantity"] = q
    detail["opened"] = y
    unit = np.maximum(0.05, econ.procurement_ratio * detail.sell_price.to_numpy())
    detail["fulfilled"] = np.minimum(q, detail.actual.to_numpy())
    detail["shortage_units"] = np.maximum(detail.actual.to_numpy() - q, 0.0)
    detail["overage_units"] = np.maximum(q - detail.actual.to_numpy(), 0.0)
    detail["realized_cost"] = (
        unit * q + econ.setup_ratio * unit * y + econ.holding_ratio * unit * detail.overage_units.to_numpy()
        + econ.shortage_ratio * unit * detail.shortage_units.to_numpy()
    )
    return {**diagnostics, **evaluation}, detail


def fit_forecast(panel: pd.DataFrame, train_end: int, start: int) -> pd.DataFrame:
    train = panel.loc[(panel.t <= train_end) & (panel.t >= 365)].dropna(subset=FEATURES)
    return recursive_forecast(panel, fit_quantile_models(train), start, HORIZON)


def make_figures(coverage: pd.DataFrame, results: pd.DataFrame, groups: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
    colors = {"Seasonal-normal": "#667782", "GBM quantile": "#2878B5", "Conformal-CVaR": "#C97A1A"}
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    summary = coverage.groupby("method", as_index=False).agg(coverage=("coverage", "mean"), width=("mean_width", "mean"))
    for _, row in summary.iterrows():
        ax.scatter(row.width, row.coverage, s=95, color=colors[row.method])
        ax.annotate(row.method, (row.width, row.coverage), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0.90, color="#9B1C1C", linewidth=1.1, linestyle="--")
    ax.set(xlabel="Mean daily interval width", ylabel="Empirical daily coverage", ylim=(0.68, 0.98))
    fig.tight_layout(); fig.savefig(FIGURES / "fig_enhanced_calibration.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    baseline = results.loc[results.policy.eq("Seasonal-normal"), ["origin", "realized_total_cost"]].rename(columns={"realized_total_cost": "baseline"})
    primary = results.loc[results.policy.eq("Conformal-CVaR")].merge(baseline, on="origin")
    primary["reduction_pct"] = 100 * (primary.baseline - primary.realized_total_cost) / primary.baseline
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(primary.origin, primary.reduction_pct, marker="o", linewidth=1.8, color=colors["Conformal-CVaR"])
    ax.axhline(0, color="#4A4A4A", linewidth=1.0)
    ax.set(xlabel="Forecast origin (M5 day index)", ylabel="Cost reduction versus seasonal (%)")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_enhanced_paired_reduction.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    if not groups.empty:
        summary = groups.groupby(["state_id", "policy"], as_index=False).agg(cost=("realized_cost", "mean"), fill=("fulfilled", "sum"), demand=("actual", "sum"))
        summary["fill_rate"] = summary.fill / summary.demand
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        sns.barplot(data=summary, x="state_id", y="fill_rate", hue="policy", palette=colors, ax=ax)
        ax.set(xlabel="State", ylabel="Realized fill rate", ylim=(0.80, 1.0))
        ax.legend(frameon=False, title="")
        fig.tight_layout(); fig.savefig(FIGURES / "fig_enhanced_state_service.png", dpi=600, bbox_inches="tight"); plt.close(fig)


def main(max_origins: int | None = None) -> None:
    start_time = time.time()
    for directory in (OUT, TABLES, FIGURES): directory.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    panel, metadata = read_and_select(n_per_cell=N_PER_CELL)
    panel = static_features(panel)
    selected_origins = ORIGINS[:max_origins] if max_origins else ORIGINS
    metadata.to_csv(TABLES / "enhanced_selected_series.csv", index=False)
    metadata.to_parquet(OUT / "enhanced_selected_series.parquet", index=False)
    econ = Economics()
    all_forecasts: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    result_rows: list[dict] = []
    selected_rows: list[dict] = []
    detail_rows: list[pd.DataFrame] = []

    for origin in selected_origins:
        # Window A: interval-calibration forecast, entirely before window B.
        window_a = fit_forecast(panel, origin - 112, origin - 111)
        a_scores = np.maximum(window_a.p10 - window_a.actual, window_a.actual - window_a.p90)
        qhat_a = qhat(a_scores)
        seasonal_scale_a = float(np.std(window_a.actual - window_a.seasonal, ddof=1))
        # Window B: policy-validation forecast. It selects the CVaR weight.
        window_b = cqr(fit_forecast(panel, origin - 84, origin - 83), qhat_a)
        window_b = seasonal_intervals(window_b, seasonal_scale_a)
        b_aggregate = aggregate_and_price(panel, window_b, origin - 84)
        b_cqr_scenarios = scenarios(window_b, b_aggregate, "Conformal-CVaR")
        validation_costs = []
        for weight in CVaR_GRID:
            outcome, _ = evaluate_policy("Conformal-CVaR", b_cqr_scenarios, b_aggregate, econ, weight)
            validation_costs.append((weight, outcome["realized_total_cost"], outcome["fill_rate"]))
        selected_weight, validation_cost, validation_fill = min(validation_costs, key=lambda item: item[1])
        selected_rows.append({
            "origin": origin, "selected_cvar_weight": selected_weight,
            "policy_validation_cost": validation_cost, "policy_validation_fill_rate": validation_fill,
            "qhat_window_a": qhat_a,
        })
        # Production forecast: all observations through the decision date are available.
        test = fit_forecast(panel, origin, origin + 1)
        pooled_scores = pd.concat([a_scores, np.maximum(window_b.p10 - window_b.actual, window_b.actual - window_b.p90)])
        qhat_production = qhat(pooled_scores)
        seasonal_scale = float(np.std(pd.concat([window_a.actual - window_a.seasonal, window_b.actual - window_b.seasonal]), ddof=1))
        test = cqr(test, qhat_production)
        test = seasonal_intervals(test, seasonal_scale)
        test["origin"] = origin
        all_forecasts.append(test)
        for method, lo, hi in (
            ("Seasonal-normal", "seasonal_lower", "seasonal_upper"),
            ("GBM quantile", "p10", "p90"),
            ("Conformal-CVaR", "cqr_lower", "cqr_upper"),
        ):
            coverage_rows.append({"origin": origin, "method": method,
                                  "coverage": float(((test.actual >= test[lo]) & (test.actual <= test[hi])).mean()),
                                  "mean_width": float((test[hi] - test[lo]).mean()),
                                  "qhat": qhat_production})
        aggregate = aggregate_and_price(panel, test, origin)
        for policy in POLICIES:
            weight = selected_weight if policy == "Conformal-CVaR" else 0.0
            outcome, detail = evaluate_policy(policy, scenarios(test, aggregate, policy), aggregate, econ, weight)
            result_rows.append({"origin": origin, "policy": policy, "cvar_weight": weight, **outcome})
            detail["origin"] = origin; detail["policy"] = policy; detail["cvar_weight"] = weight
            detail_rows.append(detail)
        print(f"origin={origin} complete; selected_lambda={selected_weight:.2f}; qhat={qhat_production:.3f}", flush=True)

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    results = pd.DataFrame(result_rows)
    details = pd.concat(detail_rows, ignore_index=True).merge(metadata[["series_id", "cat_id", "store_id", "state_id"]], on="series_id", how="left")
    coverage = pd.DataFrame(coverage_rows)
    selected = pd.DataFrame(selected_rows)
    forecasts.to_parquet(OUT / "enhanced_forecasts.parquet", index=False)
    details.to_parquet(OUT / "enhanced_decisions.parquet", index=False)
    results.to_csv(TABLES / "enhanced_policy_results.csv", index=False)
    coverage.to_csv(TABLES / "enhanced_calibration.csv", index=False)
    selected.to_csv(TABLES / "enhanced_nested_selection.csv", index=False)
    groups = details.groupby(["origin", "policy", "state_id", "cat_id"], as_index=False).agg(
        actual=("actual", "sum"), fulfilled=("fulfilled", "sum"), realized_cost=("realized_cost", "sum"),
        shortage_units=("shortage_units", "sum"), overage_units=("overage_units", "sum"),
    )
    groups.to_csv(TABLES / "enhanced_stratified_results.csv", index=False)
    make_figures(coverage, results, groups)
    manifest = {
        "study": "Nested rolling-origin M5 calibrated CVaR allocation", "seed": SEED,
        "n_per_category_store_cell": N_PER_CELL, "series": int(metadata.series_id.nunique()),
        "origins": selected_origins, "horizon_days": HORIZON, "cvar_grid": CVaR_GRID,
        "nested_design": "window A calibrates CQR; window B selects CVaR; production test is untouched",
        "economics": econ.__dict__, "python": sys.version, "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"series": manifest["series"], "origins": len(selected_origins), "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-origins", type=int)
    args = parser.parse_args()
    main(args.max_origins)
