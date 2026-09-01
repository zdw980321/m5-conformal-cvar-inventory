"""Reproducible rolling-origin forecasting and inventory allocation study on M5.

The script deliberately distinguishes observed M5 sales from counterfactual
inventory economics. Unit costs are scenario parameters based on selling prices;
no result is interpreted as Walmart's observed inventory performance.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from pyomo.environ import (
    Binary, ConcreteModel, Constraint, NonNegativeReals, Objective, RangeSet,
    SolverFactory, Var, minimize, value,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research_plan" / "data" / "raw" / "m5"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
TABLES = ARTIFACTS / "tables"
SEED = 20260901
N_PER_CELL = 8
HORIZON = 28
ORIGINS = list(range(1605, 1886, 28))
SCENARIO_LEVELS = np.array([0.05, 0.20, 0.50, 0.80, 0.95])
SCENARIO_WEIGHTS = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class Economics:
    procurement_ratio: float = 0.55
    holding_ratio: float = 0.15
    shortage_ratio: float = 8.00
    setup_ratio: float = 0.05
    budget_multiplier: float = 1.20
    capacity_multiplier: float = 1.40
    service_target: float = 0.90
    cvar_alpha: float = 0.80
    cvar_weight: float = 0.75


FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_56", "lag_364",
    "roll_7", "roll_28", "roll_56",
    "sell_price", "wday", "month", "event", "snap", "store_code", "cat_code",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dirs() -> None:
    for directory in (ARTIFACTS, FIGURES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)


def read_and_select(n_per_cell: int = N_PER_CELL) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_path = DATA / "sales_train_validation.csv"
    calendar_path = DATA / "calendar.csv"
    prices_path = DATA / "sell_prices.csv"
    sales = pd.read_csv(sales_path)
    d_cols = [column for column in sales.columns if column.startswith("d_")]
    # Sample membership is fixed using an early historical window, before every test origin.
    sales["selection_volume"] = sales[d_cols[:1000]].sum(axis=1)
    selected_parts = [
        part.nlargest(n_per_cell, "selection_volume")
        for _, part in sales.groupby(["cat_id", "store_id"], sort=True)
    ]
    selected = pd.concat(selected_parts, ignore_index=True)
    keep = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "selection_volume"] + d_cols
    selected = selected[keep].copy()
    selected["series_id"] = selected["item_id"] + "__" + selected["store_id"]
    metadata = selected[["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "selection_volume"]].copy()

    panel = selected.melt(
        id_vars=["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=d_cols,
        var_name="d", value_name="sales",
    )
    panel["t"] = panel["d"].str.extract(r"(\d+)").astype(int)
    calendar = pd.read_csv(calendar_path)
    calendar["t"] = calendar["d"].str.extract(r"(\d+)").astype(int)
    calendar["event"] = calendar[["event_name_1", "event_name_2"]].notna().any(axis=1).astype(int)
    panel = panel.merge(calendar[["t", "wm_yr_wk", "wday", "month", "event", "snap_CA", "snap_TX", "snap_WI"]], on="t", how="left")
    panel["snap"] = np.select(
        [panel["state_id"].eq("CA"), panel["state_id"].eq("TX"), panel["state_id"].eq("WI")],
        [panel["snap_CA"], panel["snap_TX"], panel["snap_WI"]], default=0,
    ).astype(int)
    prices = pd.read_csv(prices_path)
    pair_keys = metadata[["item_id", "store_id"]].drop_duplicates()
    prices = prices.merge(pair_keys, on=["item_id", "store_id"], how="inner")
    panel = panel.merge(prices, on=["item_id", "store_id", "wm_yr_wk"], how="left")
    panel["sell_price"] = panel.groupby("series_id")["sell_price"].transform(lambda col: col.ffill().bfill())
    panel["sell_price"] = panel["sell_price"].fillna(panel["sell_price"].median())
    panel["store_code"] = panel["store_id"].astype("category").cat.codes.astype(int)
    panel["cat_code"] = panel["cat_id"].astype("category").cat.codes.astype(int)
    panel = panel.sort_values(["series_id", "t"], ignore_index=True)
    return panel, metadata


def static_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    group = out.groupby("series_id", sort=False)["sales"]
    for lag in (1, 7, 14, 28, 56, 364):
        out[f"lag_{lag}"] = group.shift(lag)
    out["roll_7"] = group.transform(lambda col: col.shift(1).rolling(7, min_periods=7).mean())
    out["roll_28"] = group.transform(lambda col: col.shift(1).rolling(28, min_periods=28).mean())
    out["roll_56"] = group.transform(lambda col: col.shift(1).rolling(56, min_periods=56).mean())
    return out


def fit_quantile_models(train: pd.DataFrame) -> dict[float, LGBMRegressor]:
    models: dict[float, LGBMRegressor] = {}
    x = train[FEATURES]
    y = train["sales"]
    for alpha in (0.10, 0.50, 0.90):
        model = LGBMRegressor(
            objective="quantile", alpha=alpha, n_estimators=180, learning_rate=0.05,
            num_leaves=31, min_child_samples=40, subsample=0.85, colsample_bytree=0.90,
            reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1,
        )
        model.fit(x, y)
        models[alpha] = model
    return models


def recursive_forecast(
    panel: pd.DataFrame,
    models: dict[float, LGBMRegressor],
    start: int,
    horizon: int,
) -> pd.DataFrame:
    """Forecast recursively; only model predictions enter post-origin lags."""
    order = panel["series_id"].drop_duplicates().tolist()
    model_hist = {
        series: panel.loc[(panel.series_id == series) & (panel.t < start), "sales"].astype(float).tolist()
        for series in order
    }
    seasonal_hist = {series: values.copy() for series, values in model_hist.items()}
    covariates = panel.loc[(panel.t >= start) & (panel.t < start + horizon)].copy()
    records: list[dict[str, float | int | str]] = []
    for current_t in range(start, start + horizon):
        day = covariates.loc[covariates.t.eq(current_t)].set_index("series_id").loc[order].reset_index()
        feature_frame = day[["series_id", "sell_price", "wday", "month", "event", "snap", "store_code", "cat_code"]].copy()
        feature_frame["lag_1"] = [model_hist[s][-1] for s in order]
        feature_frame["lag_7"] = [model_hist[s][-7] for s in order]
        feature_frame["lag_14"] = [model_hist[s][-14] for s in order]
        feature_frame["lag_28"] = [model_hist[s][-28] for s in order]
        feature_frame["lag_56"] = [model_hist[s][-56] for s in order]
        feature_frame["lag_364"] = [model_hist[s][-364] for s in order]
        feature_frame["roll_7"] = [float(np.mean(model_hist[s][-7:])) for s in order]
        feature_frame["roll_28"] = [float(np.mean(model_hist[s][-28:])) for s in order]
        feature_frame["roll_56"] = [float(np.mean(model_hist[s][-56:])) for s in order]
        x = feature_frame[FEATURES]
        p10 = np.maximum(0, models[0.10].predict(x))
        p50 = np.maximum(0, models[0.50].predict(x))
        p90 = np.maximum(p50, models[0.90].predict(x))
        # Keep the benchmark's recursion separate from the model's median path.
        seasonal = np.maximum(0, np.array([seasonal_hist[s][-7] for s in order], dtype=float))
        actual = day["sales"].to_numpy(dtype=float)
        for idx, series in enumerate(order):
            records.append({
                "series_id": series, "t": current_t, "actual": actual[idx], "p10": p10[idx],
                "p50": p50[idx], "p90": p90[idx], "seasonal": seasonal[idx],
            })
            model_hist[series].append(float(p50[idx]))
            seasonal_hist[series].append(float(seasonal[idx]))
    return pd.DataFrame.from_records(records)


def interpolate_quantiles(lo: np.ndarray, med: np.ndarray, hi: np.ndarray) -> np.ndarray:
    values = []
    for level in SCENARIO_LEVELS:
        if level <= 0.50:
            weight = level / 0.50
            values.append(lo + weight * (med - lo))
        else:
            weight = (level - 0.50) / 0.50
            values.append(med + weight * (hi - med))
    return np.column_stack(values)


def aggregate_daily_quantiles(
    daily: pd.DataFrame,
    lower: str,
    median: str,
    upper: str,
    lower_probability: float = 0.10,
    upper_probability: float = 0.90,
) -> pd.DataFrame:
    """Convert daily quantiles into 28-day scenarios under conditional independence.

    Directly summing daily q10/q90 values assumes perfect serial dependence and
    substantially overstates an aggregate demand tail. This moment-matching
    approximation sums the conditional variances implied by the daily interval.
    """
    lower_z = STANDARD_NORMAL.inv_cdf(lower_probability)
    upper_z = STANDARD_NORMAL.inv_cdf(upper_probability)
    scale = np.maximum(0.0, (daily[upper].to_numpy() - daily[lower].to_numpy()) / (upper_z - lower_z))
    work = daily[["series_id", median]].copy()
    work["variance"] = scale**2
    aggregated = work.groupby("series_id", as_index=False).agg(mean=(median, "sum"), variance=("variance", "sum"))
    sigma = np.sqrt(aggregated.variance.to_numpy(dtype=float))
    for level in SCENARIO_LEVELS:
        aggregated[f"q_{int(level * 100):02d}"] = np.maximum(
            0.0, aggregated["mean"].to_numpy(dtype=float) + STANDARD_NORMAL.inv_cdf(float(level)) * sigma,
        )
    return aggregated.drop(columns="variance")


def solve_allocation(
    demand_scenarios: np.ndarray,
    price: np.ndarray,
    econ: Economics,
    cvar_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n_items, n_scenarios = demand_scenarios.shape
    unit_cost = np.maximum(0.05, econ.procurement_ratio * price)
    holding = econ.holding_ratio * unit_cost
    shortage = econ.shortage_ratio * unit_cost
    setup = econ.setup_ratio * unit_cost
    median_demand = demand_scenarios[:, 2]
    budget = econ.budget_multiplier * float(np.sum(unit_cost * median_demand + setup))
    capacity = econ.capacity_multiplier * float(np.sum(median_demand))
    maximum = np.maximum(2.0, demand_scenarios.max(axis=1) * 1.35 + 2.0)

    model = ConcreteModel()
    model.I = RangeSet(0, n_items - 1)
    model.S = RangeSet(0, n_scenarios - 1)
    model.q = Var(model.I, domain=NonNegativeReals)
    model.y = Var(model.I, domain=Binary)
    model.over = Var(model.I, model.S, domain=NonNegativeReals)
    model.short = Var(model.I, model.S, domain=NonNegativeReals)
    model.loss = Var(model.S, domain=NonNegativeReals)
    model.eta = Var(domain=NonNegativeReals)
    model.excess = Var(model.S, domain=NonNegativeReals)
    model.link = Constraint(model.I, rule=lambda m, i: m.q[i] <= float(maximum[i]) * m.y[i])
    model.over_def = Constraint(model.I, model.S, rule=lambda m, i, s: m.over[i, s] >= m.q[i] - float(demand_scenarios[i, s]))
    model.short_def = Constraint(model.I, model.S, rule=lambda m, i, s: m.short[i, s] >= float(demand_scenarios[i, s]) - m.q[i])
    model.loss_def = Constraint(
        model.S,
        rule=lambda m, s: m.loss[s] == sum(
            float(unit_cost[i]) * m.q[i] + float(setup[i]) * m.y[i]
            + float(holding[i]) * m.over[i, s] + float(shortage[i]) * m.short[i, s]
            for i in m.I
        ),
    )
    model.budget = Constraint(expr=sum(float(unit_cost[j]) * model.q[j] + float(setup[j]) * model.y[j] for j in model.I) <= budget)
    model.capacity = Constraint(expr=sum(model.q[i] for i in model.I) <= capacity)
    expected_demand = sum(
        float(SCENARIO_WEIGHTS[s]) * sum(float(demand_scenarios[i, s]) for i in range(n_items))
        for s in range(n_scenarios)
    )
    model.expected_fill_rate = Constraint(
        expr=sum(
            float(SCENARIO_WEIGHTS[s]) * sum(model.short[i, s] for i in model.I)
            for s in model.S
        ) <= (1 - econ.service_target) * expected_demand
    )
    model.excess_def = Constraint(model.S, rule=lambda m, s: m.excess[s] >= m.loss[s] - m.eta)
    expected = sum(float(SCENARIO_WEIGHTS[s]) * model.loss[s] for s in model.S)
    cvar = model.eta + sum(float(SCENARIO_WEIGHTS[s]) * model.excess[s] for s in model.S) / (1 - econ.cvar_alpha)
    model.objective = Objective(expr=expected + cvar_weight * cvar, sense=minimize)
    solver = SolverFactory("appsi_highs")
    result = solver.solve(model)
    if "optimal" not in str(result.solver.termination_condition).lower():
        raise RuntimeError(f"Allocation model did not solve optimally: {result.solver.termination_condition}")
    quantities = np.array([value(model.q[i]) for i in model.I])
    opened = np.array([round(value(model.y[i])) for i in model.I])
    diagnostics = {
        "objective": float(value(model.objective)), "budget": budget, "capacity": capacity,
        "budget_used": float(sum(unit_cost[i] * quantities[i] + setup[i] * opened[i] for i in range(n_items))),
        "capacity_used": float(quantities.sum()), "opened_items": float(opened.sum()),
        "expected_demand": float(expected_demand),
    }
    return quantities, opened, diagnostics


def evaluate_decision(q: np.ndarray, y: np.ndarray, demand: np.ndarray, price: np.ndarray, econ: Economics) -> dict[str, float]:
    unit_cost = np.maximum(0.05, econ.procurement_ratio * price)
    holding = econ.holding_ratio * unit_cost
    shortage = econ.shortage_ratio * unit_cost
    setup = econ.setup_ratio * unit_cost
    over = np.maximum(q - demand, 0)
    under = np.maximum(demand - q, 0)
    fulfilled = np.minimum(q, demand)
    return {
        "realized_total_cost": float(np.sum(unit_cost * q + setup * y + holding * over + shortage * under)),
        "realized_procurement_cost": float(np.sum(unit_cost * q + setup * y)),
        "realized_holding_cost": float(np.sum(holding * over)),
        "realized_shortage_cost": float(np.sum(shortage * under)),
        "fill_rate": float(fulfilled.sum() / max(demand.sum(), 1.0)),
        "shortage_units": float(under.sum()), "overage_units": float(over.sum()),
    }


def make_figures(coverage: pd.DataFrame, policy: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    palette = {
        "Seasonal-normal": "#7F8C8D", "GBM quantile": "#2878B5",
        "Conformal-CVaR": "#C97A1A", "Adaptive conformal ensemble": "#3D8B6E",
        "Hierarchical distributional conformal": "#8F4E9B",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    summary = coverage.groupby("method", as_index=False)["coverage"].mean()
    sns.barplot(data=summary, x="method", y="coverage", hue="method", palette=palette, legend=False, ax=ax)
    ax.axhline(0.90, color="#9B1C1C", lw=1.2, ls="--", label="Nominal 90%")
    ax.set(xlabel="", ylabel="Empirical interval coverage", ylim=(0, 1.0))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_interval_coverage.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    summary = policy.groupby("policy", as_index=False).agg(cost=("realized_total_cost", "mean"), fill=("fill_rate", "mean"))
    short_labels = {
        "Seasonal-normal": "Seasonal", "GBM quantile": "GBM", "Conformal-CVaR": "CQR-CVaR",
        "Adaptive conformal ensemble": "ACE", "Hierarchical distributional conformal": "HDC",
    }
    offsets = {
        "Seasonal-normal": (5, 7), "GBM quantile": (5, -12), "Conformal-CVaR": (6, 6),
        "Adaptive conformal ensemble": (6, 6), "Hierarchical distributional conformal": (6, -12),
    }
    for _, row in summary.iterrows():
        policy_name = row["policy"]
        ax.scatter(row["cost"], row["fill"], s=90, color=palette[policy_name], zorder=3)
        ax.annotate(short_labels[policy_name], (row["cost"], row["fill"]), xytext=offsets[policy_name], textcoords="offset points", fontsize=9)
    ax.set(xlabel="Mean realized counterfactual cost", ylabel="Mean fill rate", ylim=(0.88, 0.94))
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_cost_service_frontier.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.lineplot(data=policy, x="origin", y="realized_total_cost", hue="policy", marker="o", palette=palette, ax=ax)
    ax.set(xlabel="Forecast origin (M5 day index)", ylabel="Realized counterfactual cost")
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_rolling_costs.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.time()
    ensure_dirs()
    econ = Economics()
    np.random.seed(SEED)
    panel, metadata = read_and_select()
    panel = static_features(panel)
    metadata.to_csv(TABLES / "m5_selected_series.csv", index=False)
    panel.to_parquet(ARTIFACTS / "m5_selected_panel.parquet", index=False)
    all_forecasts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, float | int | str]] = []
    policy_rows: list[dict[str, float | int | str]] = []
    decision_rows: list[pd.DataFrame] = []

    for origin in ORIGINS:
        # A historical 28-day forecast provides calibration scores. The production
        # model is then refit using every observation available at the decision date.
        calibration_train = panel.loc[(panel.t <= origin - 56) & (panel.t >= 365)].dropna(subset=FEATURES)
        calibration_models = fit_quantile_models(calibration_train)
        calibration = recursive_forecast(panel, calibration_models, origin - 55, HORIZON)
        production_train = panel.loc[(panel.t <= origin) & (panel.t >= 365)].dropna(subset=FEATURES)
        production_models = fit_quantile_models(production_train)
        test = recursive_forecast(panel, production_models, origin + 1, HORIZON)
        conformity = np.maximum(calibration.p10 - calibration.actual, calibration.actual - calibration.p90)
        qhat = max(0.0, float(np.quantile(conformity, 0.90, method="higher")))
        test["cqr_lower"] = np.maximum(0, test.p10 - qhat)
        test["cqr_upper"] = test.p90 + qhat
        seasonal_scale = float(np.std(calibration.actual - calibration.seasonal, ddof=1))
        seasonal_halfwidth = 1.645 * seasonal_scale
        test["seasonal_lower"] = np.maximum(0, test.seasonal - seasonal_halfwidth)
        test["seasonal_upper"] = test.seasonal + seasonal_halfwidth
        # Each item-store weight is selected on the pre-test calibration window only.
        selector = (
            calibration.assign(
                seasonal_abs_error=np.abs(calibration.actual - calibration.seasonal),
                gbm_abs_error=np.abs(calibration.actual - calibration.p50),
            )
            .groupby("series_id", as_index=False)
            .agg(seasonal_mae=("seasonal_abs_error", "mean"), gbm_mae=("gbm_abs_error", "mean"))
        )
        selector["gbm_weight"] = np.clip(
            selector.seasonal_mae / (selector.seasonal_mae + selector.gbm_mae + 1e-8), 0.10, 0.90
        )
        calibration = calibration.merge(selector[["series_id", "gbm_weight"]], on="series_id", how="left")
        test = test.merge(selector[["series_id", "gbm_weight"]], on="series_id", how="left")
        calibration["ensemble_lower_raw"] = calibration.gbm_weight * calibration.p10 + (1 - calibration.gbm_weight) * np.maximum(0, calibration.seasonal - seasonal_halfwidth)
        calibration["ensemble_p50"] = calibration.gbm_weight * calibration.p50 + (1 - calibration.gbm_weight) * calibration.seasonal
        calibration["ensemble_upper_raw"] = calibration.gbm_weight * calibration.p90 + (1 - calibration.gbm_weight) * (calibration.seasonal + seasonal_halfwidth)
        ensemble_score = np.maximum(calibration.ensemble_lower_raw - calibration.actual, calibration.actual - calibration.ensemble_upper_raw)
        ensemble_qhat = max(0.0, float(np.quantile(ensemble_score, 0.90, method="higher")))
        test["ensemble_p50"] = test.gbm_weight * test.p50 + (1 - test.gbm_weight) * test.seasonal
        test["ensemble_lower"] = np.maximum(0, test.gbm_weight * test.p10 + (1 - test.gbm_weight) * test.seasonal_lower - ensemble_qhat)
        test["ensemble_upper"] = test.gbm_weight * test.p90 + (1 - test.gbm_weight) * test.seasonal_upper + ensemble_qhat

        # Hierarchical distributional conformal calibration: 28 within-series
        # residuals are shrunk toward the pooled residual distribution. It updates
        # the entire demand distribution, rather than only expanding an interval.
        residuals = calibration.assign(residual=calibration.actual - calibration.p50)
        global_residual_quantiles = residuals.residual.quantile(SCENARIO_LEVELS).to_numpy(dtype=float)
        by_series_residual_quantiles = (
            residuals.groupby("series_id").residual.quantile(SCENARIO_LEVELS).unstack()
            .reindex(columns=SCENARIO_LEVELS)
            .reset_index()
        )
        sample_sizes = residuals.groupby("series_id").size().rename("calibration_n").reset_index()
        hdc = by_series_residual_quantiles.merge(sample_sizes, on="series_id", how="left")
        shrinkage = hdc.calibration_n.to_numpy(dtype=float) / (hdc.calibration_n.to_numpy(dtype=float) + 56.0)
        hdc_adjustments: dict[str, np.ndarray] = {}
        for position, level in enumerate(SCENARIO_LEVELS):
            column = level
            adjustment = shrinkage * hdc[column].to_numpy(dtype=float) + (1 - shrinkage) * global_residual_quantiles[position]
            hdc_adjustments[f"hdc_{int(level * 100):02d}"] = adjustment
        hdc_frame = pd.DataFrame({"series_id": hdc.series_id, **hdc_adjustments})
        test = test.merge(hdc_frame, on="series_id", how="left")
        for level in SCENARIO_LEVELS:
            suffix = f"hdc_{int(level * 100):02d}"
            test[suffix] = np.maximum(0, test.p50 + test[suffix])
        test["origin"] = origin
        all_forecasts.append(test)
        for method, lower, upper in [
            ("Seasonal-normal", "seasonal_lower", "seasonal_upper"),
            ("GBM quantile", "p10", "p90"),
            ("Conformal-CVaR", "cqr_lower", "cqr_upper"),
            ("Adaptive conformal ensemble", "ensemble_lower", "ensemble_upper"),
            ("Hierarchical distributional conformal", "hdc_05", "hdc_95"),
        ]:
            coverage_rows.append({
                "origin": origin, "method": method,
                "coverage": float(((test.actual >= test[lower]) & (test.actual <= test[upper])).mean()),
                "mean_width": float((test[upper] - test[lower]).mean()), "qhat": qhat,
                "ensemble_qhat": ensemble_qhat,
            })

        aggregate = test.groupby("series_id", as_index=False).agg(actual=("actual", "sum"))
        price = panel.loc[panel.t.eq(origin), ["series_id", "sell_price"]].drop_duplicates("series_id")
        aggregate = aggregate.merge(price, on="series_id", how="left").sort_values("series_id").reset_index(drop=True)
        scenario_frames = {
            "Seasonal-normal": aggregate_daily_quantiles(test, "seasonal_lower", "seasonal", "seasonal_upper", 0.05, 0.95),
            "GBM quantile": aggregate_daily_quantiles(test, "p10", "p50", "p90", 0.10, 0.90),
            "Conformal-CVaR": aggregate_daily_quantiles(test, "cqr_lower", "p50", "cqr_upper", 0.05, 0.95),
            "Adaptive conformal ensemble": aggregate_daily_quantiles(test, "ensemble_lower", "ensemble_p50", "ensemble_upper", 0.05, 0.95),
            "Hierarchical distributional conformal": aggregate_daily_quantiles(test, "hdc_05", "hdc_50", "hdc_95", 0.05, 0.95),
        }
        for name, scenario_frame in scenario_frames.items():
            scenario_frame = aggregate[["series_id"]].merge(scenario_frame, on="series_id", how="left")
            scenarios = scenario_frame[[f"q_{int(level * 100):02d}" for level in SCENARIO_LEVELS]].to_numpy()
            risk_weight = econ.cvar_weight if name == "Conformal-CVaR" else 0.0
            quantities, opened, diagnostics = solve_allocation(scenarios, aggregate.sell_price.to_numpy(), econ, risk_weight)
            evaluation = evaluate_decision(quantities, opened, aggregate.actual.to_numpy(), aggregate.sell_price.to_numpy(), econ)
            policy_rows.append({"origin": origin, "policy": name, **diagnostics, **evaluation})
            decisions = aggregate[["series_id", "actual", "sell_price"]].copy()
            decisions["origin"] = origin
            decisions["policy"] = name
            decisions["order_quantity"] = quantities
            decisions["opened"] = opened
            decision_rows.append(decisions)
        print(
            f"origin={origin} complete; qhat={qhat:.3f}; "
            f"calibration_rows={len(calibration_train):,}; production_rows={len(production_train):,}"
        )

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    policy = pd.DataFrame(policy_rows)
    decisions = pd.concat(decision_rows, ignore_index=True)
    forecasts.to_parquet(ARTIFACTS / "m5_forecasts.parquet", index=False)
    decisions.to_parquet(ARTIFACTS / "m5_inventory_decisions.parquet", index=False)
    coverage.to_csv(TABLES / "table_forecast_calibration.csv", index=False)
    policy.to_csv(TABLES / "table_policy_results.csv", index=False)
    policy.groupby("policy", as_index=False).mean(numeric_only=True).to_csv(TABLES / "table_policy_summary.csv", index=False)
    make_figures(coverage, policy)
    manifest = {
        "study": "M5 rolling-origin calibrated inventory allocation", "seed": SEED,
        "origins": ORIGINS, "horizon_days": HORIZON, "series": int(metadata.series_id.nunique()),
        "selection": f"top {N_PER_CELL} series by M5 days 1-1000 volume in each category-store cell",
        "economics": econ.__dict__, "m5_files": {path.name: sha256(path) for path in sorted(DATA.glob("*.csv"))},
        "python": sys.version, "platform": platform.platform(), "elapsed_seconds": round(time.time() - started, 2),
    }
    (ARTIFACTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": manifest["elapsed_seconds"], "series": manifest["series"]}, indent=2))


if __name__ == "__main__":
    main()
