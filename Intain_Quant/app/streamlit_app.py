"""
Intain_Quant — RMBS Projection Engine Dashboard
================================================

Three behavioral analysis methods:
  1. Custom Input:          User specifies flat CPR/CDR/Severity
  2. Predefined Scenarios:  Pick from preset economic scenarios (Base, Fast Prepay, etc.)
  3. Monte Carlo:           Compute distributions from history, sample correlated paths

Run: cd Intain_Quant && streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

try:
    import altair as alt
    _HAS_ALTAIR = True
except Exception:
    alt = None
    _HAS_ALTAIR = False

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.schema import CES1_TAPE_COLUMNS
from core.config import ProjectionConfig
from core.utils import require_columns  # noqa: F401 — used by submodules

from data_prep.loader import load_ces_csv
from data_prep.tape_builder import (
    infer_as_of_date,
    build_static_loan_tape_from_history,
)
from data_prep.validators import validate_tape

from behaviors.constant import ConstantHazardModel

from distributions.sampler import DistributionParams, MonteCarloSampler
from distributions.benchmarks import (
    get_benchmark_distributions,
    blend_with_benchmarks,
)
from distributions.historical import compute_historical_distributions
from distributions.correlation import (
    get_benchmark_correlation_matrix,
    compute_correlation_matrix,
    correlation_matrix_to_dataframe,
)

from engine.runner import run_projection

from pm.metrics import compute_path_metrics
from pm.aggregator import aggregate_path_results, aggregate_cashflows_by_date
from pm.decisions import generate_decision_report

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TAPE_DIR = PROJECT_ROOT / "data" / "tapes"
TAPE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Deal name mapping (filename -> short name)
# ---------------------------------------------------------------------------
DEAL_SHORT_NAMES = {
    "J.P. MORGAN MORTGAGE TRUST 2024-CES1": "2024-CES1",
    "J.P. MORGAN MORTGAGE TRUST 2024-CES2": "2024-CES2",
    "J.P. MORGAN MORTGAGE TRUST 2025-CES1": "2025-CES1",
}
ALL_DEALS_LABEL = "All Deals (Collated)"

# ---------------------------------------------------------------------------
# Scenario presets for Predefined Scenarios method
# ---------------------------------------------------------------------------
SCENARIO_PRESETS: dict[str, dict[str, object]] = {
    "Base": {
        "cpr": 0.10, "cdr": 0.02, "severity": 0.35, "recovery_lag_months": 12,
        "description": "Normal economy, steady performance",
    },
    "Fast Prepay": {
        "cpr": 0.15, "cdr": 0.015, "severity": 0.30, "recovery_lag_months": 10,
        "description": "Rates drop, borrowers refi, fewer defaults",
    },
    "Slow Prepay / High Loss": {
        "cpr": 0.05, "cdr": 0.05, "severity": 0.50, "recovery_lag_months": 15,
        "description": "Rates rise, economy weakens",
    },
    "Severe Stress": {
        "cpr": 0.03, "cdr": 0.08, "severity": 0.60, "recovery_lag_months": 18,
        "description": "Recession — slow prepay, high defaults, deep losses",
    },
    "Crisis": {
        "cpr": 0.02, "cdr": 0.12, "severity": 0.75, "recovery_lag_months": 24,
        "description": "2008-level meltdown",
    },
}

# Asset class -> scenario dict (extensible for future asset classes)
ASSET_TYPE_SCENARIOS: dict[str, dict] = {
    "RMBS Mortgage": SCENARIO_PRESETS,
}


def _deal_short_name(csv_path: str) -> str:
    """Extract short deal name from CSV path."""
    stem = Path(csv_path).stem
    for long_name, short_name in DEAL_SHORT_NAMES.items():
        if long_name in stem:
            return short_name
    return stem


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading raw CSV...")
def _load_raw(path: str) -> pd.DataFrame:
    return load_ces_csv(path, low_memory=False)


@st.cache_data(show_spinner="Building loan tape...")
def _build_tape(raw_hash: int, raw: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Build tape with hash-based cache key (raw_hash is hash of raw df shape + path)."""
    return build_static_loan_tape_from_history(raw, as_of=as_of, keep_columns=CES1_TAPE_COLUMNS)


@st.cache_data(show_spinner="Computing historical distributions...")
def _compute_hist_dist(raw_hash: int, raw: pd.DataFrame, deal_name: str):
    """Compute historical CPR/CDR/Severity from raw deal data."""
    return compute_historical_distributions(raw, deal_name=deal_name)


def _raw_hash(path: str, shape: tuple) -> int:
    """Create a hashable key for caching based on file path and shape."""
    return hash((path, shape))


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def _fmt_balance(val):
    """Format balance with commas."""
    return f"{val:,.0f}"


def _fmt_pct(val):
    """Format percentage with 2 decimals."""
    return f"{val:.2%}"


def _fmt_years(val):
    """Format years with 2 decimals."""
    return f"{val:.2f}"


def _plot_line(df, *, x, y, title, y_title, height=280):
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or x not in df.columns or y not in df.columns:
        st.info("No data to plot.")
        return
    if not _HAS_ALTAIR:
        st.markdown(f"**{title}**")
        st.line_chart(df.set_index(x)[y])
        return
    d = df[[x, y]].copy()
    d[x] = pd.to_datetime(d[x], errors="coerce")
    chart = (
        alt.Chart(d).mark_line()
        .encode(
            x=alt.X(f"{x}:T", title="Date"),
            y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=",.0f")),
        )
        .properties(title=title, height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def _plot_multi_line(df, *, x, ys, title, y_title, height=280):
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or x not in df.columns:
        return
    if any(y not in df.columns for y in ys):
        return
    if not _HAS_ALTAIR:
        st.markdown(f"**{title}**")
        st.line_chart(df.set_index(x)[ys])
        return
    d = df[[x] + ys].copy()
    d[x] = pd.to_datetime(d[x], errors="coerce")
    long = d.melt(id_vars=[x], value_vars=ys, var_name="series", value_name="value")
    chart = (
        alt.Chart(long).mark_line()
        .encode(
            x=alt.X(f"{x}:T", title="Date"),
            y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=",.0f")),
            color=alt.Color("series:N", title="Series"),
        )
        .properties(title=title, height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def _plot_histogram(values, *, title, x_label, bins=40, height=280):
    if len(values) == 0:
        st.info("No data.")
        return
    if not _HAS_ALTAIR:
        st.markdown(f"**{title}**")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(values, bins=bins, edgecolor="white", alpha=0.8)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        ax.set_title(title)
        st.pyplot(fig)
        return
    df_hist = pd.DataFrame({"value": values})
    chart = (
        alt.Chart(df_hist).mark_bar(opacity=0.8)
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=bins), title=x_label),
            y=alt.Y("count()", title="Paths"),
        )
        .properties(title=title, height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def _plot_band_chart(expected_df, band_df, *, x, y_expected, title, y_title, height=280):
    if not _HAS_ALTAIR or expected_df is None or band_df is None:
        return
    if len(expected_df) == 0 or len(band_df) == 0:
        return
    exp = expected_df[[x, y_expected]].copy()
    exp[x] = pd.to_datetime(exp[x], errors="coerce")
    band = band_df.copy()
    band[x] = pd.to_datetime(band[x], errors="coerce")
    merged = exp.merge(band, on=x, how="inner")
    band_cols = [c for c in merged.columns if c.startswith("p")]
    if len(band_cols) < 2:
        return
    low_col = band_cols[0]
    high_col = band_cols[-1]
    area = (
        alt.Chart(merged).mark_area(opacity=0.2, color="steelblue")
        .encode(
            x=alt.X(f"{x}:T", title="Date"),
            y=alt.Y(f"{low_col}:Q", title=y_title),
            y2=f"{high_col}:Q",
        )
    )
    line = (
        alt.Chart(merged).mark_line(color="steelblue", strokeWidth=2)
        .encode(x=alt.X(f"{x}:T"), y=alt.Y(f"{y_expected}:Q"))
    )
    chart = (area + line).properties(title=title, height=height)
    st.altair_chart(chart, use_container_width=True)


def _plot_scatter_wal_loss(path_metrics, raw_results):
    if not _HAS_ALTAIR or len(path_metrics) == 0:
        return
    scatter_df = path_metrics.copy()
    scatter_df["cum_loss_pct_display"] = scatter_df["cum_loss_pct"] * 100
    if "sampled_assumptions" in raw_results:
        sa = raw_results["sampled_assumptions"]
        scatter_df = scatter_df.merge(sa, on="path_id", how="left")
        color_enc = alt.Color("cdr:Q", title="CDR", scale=alt.Scale(scheme="reds"))
        tooltip_fields = ["path_id", "wal_years", "cum_loss_pct_display", "cpr", "cdr", "severity"]
    else:
        color_enc = alt.value("steelblue")
        tooltip_fields = ["path_id", "wal_years", "cum_loss_pct_display"]
    chart = (
        alt.Chart(scatter_df).mark_circle(size=30, opacity=0.6)
        .encode(
            x=alt.X("wal_years:Q", title="WAL (years)"),
            y=alt.Y("cum_loss_pct_display:Q", title="Cumulative Loss (%)"),
            color=color_enc,
            tooltip=tooltip_fields,
        )
        .properties(title="WAL vs Loss — Each Dot Is One Path", height=400)
    )
    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# Engine helper — run projection and return results
# ---------------------------------------------------------------------------
def _run_engine(tape, cfg, *, behavior_model=None, sampled_paths=None, show_progress=True):
    """Run engine and return (cashflows, raw_results, path_metrics)."""
    if show_progress and sampled_paths is not None:
        progress_bar = st.progress(0, text="Running projection...")
    else:
        progress_bar = None

    cashflows, raw_results = run_projection(
        tape.reset_index(drop=True), cfg,
        behavior_model=behavior_model,
        sampled_paths=sampled_paths,
    )

    if progress_bar is not None:
        progress_bar.progress(100, text="Engine complete!")

    path_metrics = compute_path_metrics(
        cashflows,
        as_of_date=cfg.as_of_date,
        original_balance=raw_results["original_balance"],
    )
    return cashflows, raw_results, path_metrics


# ---------------------------------------------------------------------------
# Shared PM output display — used by all 3 analysis methods
# ---------------------------------------------------------------------------
def _display_pm_outputs(cashflows, raw_results, path_metrics, deal_name, custom_mode=False):
    """Display consistent PM outputs for any analysis method.

    When custom_mode=True (single deterministic run), show only three charts
    and the full projected cashflow table — no KPIs, percentile tables,
    decision reports, or histograms.
    """
    n_paths = int(path_metrics["path_id"].nunique()) if "path_id" in path_metrics.columns else len(path_metrics)

    # --- Aggregation ---
    agg = aggregate_path_results(path_metrics)
    cf_agg = aggregate_cashflows_by_date(cashflows)
    expected_cf = cf_agg.get("expected_by_date")
    band_cf = cf_agg.get("percentile_bands")

    if custom_mode:
        # --- Custom mode: only 3 charts + full cashflow table ---
        if expected_cf is not None and len(expected_cf) > 0:
            _plot_line(expected_cf, x="date", y="total_cashflow",
                       title="Expected Monthly Cashflow", y_title="Cashflow ($)")

            left, right = st.columns(2)
            with left:
                _plot_line(expected_cf, x="date", y="loss",
                           title="Expected Monthly Loss", y_title="Loss ($)")
            with right:
                _plot_line(expected_cf, x="date", y="end_balance",
                           title="Expected Balance", y_title="Balance ($)")

        with st.expander("Full Projected Cashflow Table", expanded=False):
            if expected_cf is not None:
                st.dataframe(expected_cf, use_container_width=True)
            else:
                numeric_cols = [c for c in ["begin_balance", "end_balance", "interest",
                    "scheduled_principal", "prepayment", "default_principal", "loss",
                    "recovery", "principal", "total_cashflow"] if c in cashflows.columns]
                fallback_cf = cashflows.groupby("date", as_index=False)[numeric_cols].mean()
                st.dataframe(fallback_cf, use_container_width=True)
        return

    # --- Full mode (Predefined Scenarios / Monte Carlo) ---
    report = generate_decision_report(path_metrics, deal_name=deal_name, yield_offered=None)

    # --- 1. KPI row ---
    mean_wal = float(path_metrics["wal_years"].mean())
    mean_loss = float(path_metrics["cum_loss_pct"].mean())
    mean_life = float(path_metrics["expected_life_years"].mean())
    orig_bal = raw_results["original_balance"]
    p95_loss = float(report.p95_loss_pct) if report.p95_loss_pct is not None else mean_loss

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mean WAL", f"{mean_wal:.2f} yr")
    k2.metric("Cumulative Loss", _fmt_pct(mean_loss))
    k3.metric("Expected Life", f"{mean_life:.2f} yr")
    k4.metric("Original Balance", _fmt_balance(orig_bal))
    k5.metric("P95 Loss", _fmt_pct(p95_loss))

    # --- 2. Risk flags ---
    if report.flags:
        for flag in report.flags:
            st.warning(flag)

    # --- 3. Percentile table ---
    st.markdown("**Percentile Table**")
    st.dataframe(agg["summary_table"].round(4), use_container_width=True, hide_index=True)

    # --- 4. Decision report + interpretation ---
    st.markdown("**Decision Report**")
    rpt_col1, rpt_col2 = st.columns([2, 1])
    with rpt_col1:
        st.dataframe(report.to_dataframe(), use_container_width=True, hide_index=True)
    with rpt_col2:
        st.markdown("**Interpretation**")
        st.markdown(
            f"- In **{(1 - report.prob_loss_exceeds_5pct):.0%}** of simulated futures, "
            f"cumulative loss stays below 5%.\n"
            f"- The worst 1% of scenarios show loss exceeding **{report.p99_loss_pct:.2%}**.\n"
            f"- WAL ranges from **{report.mean_wal - report.wal_spread/2:.1f}** to "
            f"**{report.mean_wal + report.wal_spread/2:.1f}** years (5th-95th pctl)."
        )

    # --- 5. Expected cashflow chart ---
    if expected_cf is not None and len(expected_cf) > 0:
        _plot_line(expected_cf, x="date", y="total_cashflow",
                   title="Expected Monthly Cashflow", y_title="Cashflow ($)")

    # --- 6. Balance + Loss side-by-side ---
    if expected_cf is not None and len(expected_cf) > 0:
        left, right = st.columns(2)
        with left:
            _plot_line(expected_cf, x="date", y="end_balance",
                       title="Expected Balance", y_title="Balance ($)")
        with right:
            _plot_line(expected_cf, x="date", y="loss",
                       title="Expected Monthly Loss", y_title="Loss ($)")

    # --- 7. Distribution histograms ---
    st.markdown("**Result Distributions**")
    d1, d2, d3 = st.columns(3)
    with d1:
        _plot_histogram(agg["wal_distribution"], title="WAL Distribution", x_label="WAL (years)", bins=35)
    with d2:
        _plot_histogram(agg["loss_distribution"] * 100, title="Loss Distribution", x_label="Loss (%)", bins=35)
    with d3:
        _plot_histogram(agg["life_distribution"], title="Expected Life", x_label="Life (years)", bins=35)

    # --- 8. WAL vs Loss scatter (multi-path only) ---
    if n_paths > 1:
        st.markdown("**WAL vs Loss Scatter (each dot = one path)**")
        _plot_scatter_wal_loss(path_metrics, raw_results)

    # --- 9. Cashflow confidence bands (multi-path only) ---
    if n_paths > 1 and expected_cf is not None and band_cf is not None:
        _plot_band_chart(
            expected_cf, band_cf,
            x="date", y_expected="total_cashflow",
            title="Total Cashflow (Expected + P05-P95 Band)",
            y_title="Cashflow ($)",
        )

    # --- 10. Expandable cashflow table ---
    with st.expander("Expected cashflow table (first 50)", expanded=False):
        if expected_cf is not None:
            st.dataframe(expected_cf.head(50), use_container_width=True)
        else:
            # Fallback: compute from raw cashflows
            numeric_cols = [c for c in ["begin_balance", "end_balance", "interest",
                "scheduled_principal", "prepayment", "default_principal", "loss",
                "recovery", "principal", "total_cashflow"] if c in cashflows.columns]
            fallback_cf = cashflows.groupby("date", as_index=False)[numeric_cols].mean()
            st.dataframe(fallback_cf.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Intain Quant", layout="wide")
st.title("Intain Quant")
st.caption(
    "Collateral Projection Engine: "
    
)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Deal Selection
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Deal Selection")

    csv_files = sorted([p for p in DATA_DIR.glob("*.csv")]) if DATA_DIR.exists() else []
    if not csv_files:
        st.error(f"No CSVs found in {DATA_DIR}")
        st.stop()

    # Build deal options
    deal_options = [_deal_short_name(str(p)) for p in csv_files] + [ALL_DEALS_LABEL]
    deal_paths = {_deal_short_name(str(p)): str(p) for p in csv_files}

    selected_deal = st.selectbox("Select Deal", options=deal_options, index=0)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA — Single Deal or Collated
# ═══════════════════════════════════════════════════════════════════════════
if selected_deal == ALL_DEALS_LABEL:
    # Load all deals, build tapes, collate
    all_tapes = []
    all_raws = []
    deal_info_rows = []
    max_as_of = pd.Timestamp("1970-01-01")

    for csv_path in csv_files:
        dname = _deal_short_name(str(csv_path))
        raw_i = _load_raw(str(csv_path))
        as_of_i = infer_as_of_date(raw_i)
        h = _raw_hash(str(csv_path), raw_i.shape)
        tape_i = _build_tape(h, raw_i, as_of_i)

        # Prefix Loan IDs with deal name to avoid collisions
        tape_i = tape_i.copy()
        tape_i["Loan ID"] = dname.replace("-", "") + "_" + tape_i["Loan ID"].astype(str)
        tape_i["deal_id"] = dname

        all_tapes.append(tape_i)
        all_raws.append(raw_i)
        deal_info_rows.append({"Deal": dname, "Loans": len(tape_i), "As-of Date": str(as_of_i.date())})
        if as_of_i > max_as_of:
            max_as_of = as_of_i

    tape = pd.concat(all_tapes, ignore_index=True)
    as_of = max_as_of
    deal_name_display = ALL_DEALS_LABEL
    raw_for_hist = pd.concat(all_raws, ignore_index=True)
    raw_for_hist_name = "All Deals"
else:
    csv_path = deal_paths[selected_deal]
    raw = _load_raw(csv_path)
    as_of = infer_as_of_date(raw)
    h = _raw_hash(csv_path, raw.shape)
    tape = _build_tape(h, raw, as_of)
    deal_name_display = selected_deal
    raw_for_hist = raw
    raw_for_hist_name = selected_deal
    deal_info_rows = None

# Derive projection horizon from tape
_max_term = tape["Original Term"].max()
projection_months = int(min(_max_term, 480)) if pd.notna(_max_term) else 360

# ═══════════════════════════════════════════════════════════════════════════
# DEAL INFO
# ═══════════════════════════════════════════════════════════════════════════
st.subheader(f"Deal: {deal_name_display}")

if deal_info_rows is not None:
    # Collated — show per-deal counts
    info_df = pd.DataFrame(deal_info_rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Loans", f"{len(tape):,}")
    c2.metric("Projection As-of Date", str(as_of.date()))
    c3.metric("Deals Loaded", str(len(deal_info_rows)))
    st.dataframe(info_df, use_container_width=True, hide_index=True)
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Loans in Tape", f"{len(tape):,}")
    c2.metric("As-of Date", str(as_of.date()))
    

# Validation
vr = validate_tape(tape)
if not vr.is_valid:
    st.error("Tape validation failed:\n" + vr.summary())

# ═══════════════════════════════════════════════════════════════════════════
# BASE CASE — Contractual Cashflows (CPR=0, CDR=0, Severity=0)
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Base Case (Contractual Cashflows)")
st.caption("This shows contractual cashflows assuming zero defaults and zero prepayments")

# Run base case automatically when deal changes
base_key = f"base_case_{selected_deal}"
if base_key not in st.session_state:
    base_model = ConstantHazardModel(cpr_annual=0.0, cdr_annual=0.0, severity=0.0)
    base_cfg = ProjectionConfig(
        as_of_date=pd.Timestamp(as_of),
        projection_months=int(projection_months),
        n_paths=1,
        seed=42,
        recovery_lag_months=0,
    )
    with st.spinner("Computing base case (contractual cashflows)..."):
        try:
            base_cf, base_raw, _ = _run_engine(
                tape, base_cfg, behavior_model=base_model, show_progress=False
            )
            st.session_state[base_key] = (base_cf, base_raw)
        except Exception as e:
            st.error(f"Base case computation failed: {e}")
            st.session_state[base_key] = None


base_data = st.session_state.get(base_key)
if base_data is not None:
    base_cf, base_raw = base_data
    # Build monthly pool-level table
    base_monthly = base_cf[base_cf["path_id"] == 0][[
        "date", "begin_balance", "interest", "scheduled_principal", "end_balance"
    ]].copy()
    base_monthly.columns = ["Date", "Begin Balance", "Interest", "Scheduled Principal", "End Balance"]
    base_monthly["Date"] = pd.to_datetime(base_monthly["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Chart
    _plot_line(
        base_cf[base_cf["path_id"] == 0], x="date", y="end_balance",
        title="Balance Amortization (Contractual)", y_title="Balance ($)", height=300
    )

    with st.expander("Base case cashflow table (Engine output)", expanded=False):
        display_base = base_monthly.copy()
        for c in ["Begin Balance", "Interest", "Scheduled Principal", "End Balance"]:
            display_base[c] = display_base[c].apply(lambda v: f"{v:,.2f}")
        st.dataframe(display_base, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# BEHAVIORAL ASSUMPTIONS — 3 Methods
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Probable Scenarios Analysis")

method = st.radio(
    "Analysis Method",
    options=["Custom Input", "Predefined Scenarios", "Monte Carlo"],
    horizontal=True,
)

# -----------------------------------------------------------------------
# METHOD 1: Custom Input
# -----------------------------------------------------------------------
if method == "Custom Input":
    col1, col2 = st.columns(2)
    with col1:
        m_cpr = st.slider("CPR (annual %)", 0.0, 60.0, 10.0, 0.5, format="%.1f%%") / 100.0
        m_cdr = st.slider("CDR (annual %)", 0.0, 30.0, 2.0, 0.5, format="%.1f%%") / 100.0
    with col2:
        m_sev = st.slider("Severity (%)", 0.0, 100.0, 35.0, 1.0, format="%.0f%%") / 100.0
        m_lag = st.slider("Recovery Lag (months)", 0, 60, 12, 1)

    run_manual = st.button("Run Projection", type="primary", use_container_width=True)

    if run_manual:
        model = ConstantHazardModel(
            cpr_annual=m_cpr, cdr_annual=m_cdr, severity=m_sev
        )
        cfg = ProjectionConfig(
            as_of_date=pd.Timestamp(as_of),
            projection_months=int(projection_months),
            n_paths=1,
            seed=42,
            recovery_lag_months=int(m_lag),
        )
        with st.spinner(f"Running projection with CPR={m_cpr:.1%}, CDR={m_cdr:.1%}, Severity={m_sev:.0%}..."):
            cashflows, raw_results, path_metrics = _run_engine(
                tape, cfg, behavior_model=model, show_progress=False
            )

        st.success(f"Projection complete: {raw_results['horizon']} months")
        _display_pm_outputs(cashflows, raw_results, path_metrics, deal_name_display, custom_mode=True)

# -----------------------------------------------------------------------
# METHOD 2: Predefined Scenarios
# -----------------------------------------------------------------------
elif method == "Predefined Scenarios":
    asset_type = st.selectbox("Asset Class", list(ASSET_TYPE_SCENARIOS.keys()) + ["Auto ABS", "CLO", "Student Loan"])
    if asset_type not in ASSET_TYPE_SCENARIOS:
        st.info(f"Scenarios for {asset_type}: Coming soon")
        st.stop()

    scenarios = ASSET_TYPE_SCENARIOS[asset_type]

    # Scenario selection table with checkboxes
    st.markdown("**Select Scenarios to Run**")
    preset_tbl = pd.DataFrame([
        {"Select": True, "Scenario": n, "CPR": f"{float(v['cpr'])*100:.1f}%",
         "CDR": f"{float(v['cdr'])*100:.1f}%", "Severity": f"{float(v['severity'])*100:.0f}%",
         "Recovery Lag": f"{int(v['recovery_lag_months'])}mo", "Description": v["description"]}
        for n, v in scenarios.items()
    ])
    edited_tbl = st.data_editor(
        preset_tbl,
        hide_index=True,
        use_container_width=True,
        key="scenario_editor",
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=True),
        },
        disabled=["Scenario", "CPR", "CDR", "Severity", "Recovery Lag", "Description"],
    )

    selected_names = edited_tbl.loc[edited_tbl["Select"], "Scenario"].tolist()

    run_selected = st.button(
        "Run Selected Scenarios", type="primary", use_container_width=True,
        disabled=len(selected_names) == 0,
    )

    if run_selected:
        if not selected_names:
            st.warning("Please select at least one scenario.")
        elif len(selected_names) == 1:
            sc_name = selected_names[0]
            sc = scenarios[sc_name]
            model = ConstantHazardModel(
                cpr_annual=float(sc["cpr"]),
                cdr_annual=float(sc["cdr"]),
                severity=float(sc["severity"]),
            )
            cfg = ProjectionConfig(
                as_of_date=pd.Timestamp(as_of),
                projection_months=int(projection_months),
                n_paths=1,
                seed=42,
                recovery_lag_months=int(sc["recovery_lag_months"]),
            )
            with st.spinner(f"Running '{sc_name}' scenario..."):
                cashflows, raw_results, path_metrics = _run_engine(
                    tape, cfg, behavior_model=model, show_progress=False
                )
            st.success(f"'{sc_name}' complete: {raw_results['horizon']} months")
            _display_pm_outputs(cashflows, raw_results, path_metrics, deal_name_display, custom_mode=True)
        else:
            st.markdown("### Selected Scenarios Comparison")
            comparison_rows = []
            raw_wal = {}
            raw_loss = {}
            progress = st.progress(0, text="Running selected scenarios...")
            total = len(selected_names)

            for idx, sc_name in enumerate(selected_names):
                sc = scenarios[sc_name]
                model = ConstantHazardModel(
                    cpr_annual=float(sc["cpr"]),
                    cdr_annual=float(sc["cdr"]),
                    severity=float(sc["severity"]),
                )
                cfg = ProjectionConfig(
                    as_of_date=pd.Timestamp(as_of),
                    projection_months=int(projection_months),
                    n_paths=1,
                    seed=42,
                    recovery_lag_months=int(sc["recovery_lag_months"]),
                )
                _, _, pm = _run_engine(tape, cfg, behavior_model=model, show_progress=False)

                mean_wal = float(pm["wal_years"].mean())
                mean_loss = float(pm["cum_loss_pct"].mean())
                raw_wal[sc_name] = mean_wal
                raw_loss[sc_name] = mean_loss

                comparison_rows.append({
                    "Scenario": sc_name,
                    "WAL (yr)": f"{mean_wal:.2f}",
                    "Loss (%)": f"{mean_loss:.2%}",
                })
                progress.progress((idx + 1) / total, text=f"Completed: {sc_name}")

            progress.empty()
            st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

            # --- Sensitivity summary ---
            base_name = "Base" if "Base" in raw_wal else selected_names[0]
            base_wal = raw_wal[base_name]
            base_loss = raw_loss[base_name]

            sensitivity_lines = []
            for sc_name in selected_names:
                if sc_name == base_name:
                    continue
                wal_chg = ((raw_wal[sc_name] - base_wal) / base_wal * 100) if base_wal > 0 else 0
                loss_chg = ((raw_loss[sc_name] - base_loss) / base_loss * 100) if base_loss > 0 else 0
                wal_dir = "extending" if wal_chg > 0 else "contracting"
                loss_dir = "increasing" if loss_chg > 0 else "decreasing"
                sensitivity_lines.append(
                    f"- **{sc_name}** vs {base_name}: "
                    f"WAL {wal_dir} by {abs(wal_chg):.1f}%, "
                    f"Loss {loss_dir} by {abs(loss_chg):.1f}%"
                )

            if sensitivity_lines:
                # Identify the highest-CDR scenario among selected
                worst_sc = max(
                    [n for n in selected_names if n != base_name],
                    key=lambda n: raw_loss[n],
                    default=None,
                )
                header = ""
                if worst_sc and base_wal > 0:
                    wal_pct = abs((raw_wal[worst_sc] - base_wal) / base_wal * 100)
                    header = (
                        f"The portfolio shows high sensitivity to default rates, "
                        f"with WAL {('extending' if raw_wal[worst_sc] > base_wal else 'contracting')} "
                        f"by {wal_pct:.1f}% in the {worst_sc} scenario compared to {base_name}."
                    )
                st.info(
                    f"**Sensitivity Analysis**\n\n{header}\n\n" + "\n".join(sensitivity_lines)
                )

# -----------------------------------------------------------------------
# METHOD 3: Monte Carlo
# -----------------------------------------------------------------------
elif method == "Monte Carlo":

    # ---------------------------------------------------------------
    # Step 1: Compute Historical Distributions
    # ---------------------------------------------------------------
    st.markdown("#### Step 1: Historical Distribution Analysis")

    compute_hist = st.button("Compute Historical Distributions", type="secondary")

    hist_key = f"hist_dist_{selected_deal}"
    if compute_hist:
        try:
            rh = _raw_hash(selected_deal, raw_for_hist.shape)
            hist_dists = _compute_hist_dist(rh, raw_for_hist, raw_for_hist_name)
            benchmarks = get_benchmark_distributions("ces")

            # Compute correlation from data
            corr_matrix = compute_correlation_matrix(hist_dists)

            # Blend with benchmarks
            blended = {}
            for var in ["cpr", "cdr", "severity"]:
                hd = hist_dists[var]
                bm = benchmarks[var]
                b_mean, b_std = blend_with_benchmarks(
                    hd.mean, hd.std, hd.n_observations, bm
                )
                blended[var] = {"mean": b_mean, "std": b_std, "n_obs": hd.n_observations}

            st.session_state[hist_key] = {
                "hist_dists": hist_dists,
                "benchmarks": benchmarks,
                "blended": blended,
                "corr_matrix": corr_matrix,
            }
        except Exception as e:
            st.error(f"Historical computation failed: {e}")
            st.warning("Falling back to industry benchmarks.")
            benchmarks = get_benchmark_distributions("ces")
            st.session_state[hist_key] = {
                "hist_dists": None,
                "benchmarks": benchmarks,
                "blended": {
                    "cpr": {"mean": benchmarks["cpr"].mean, "std": benchmarks["cpr"].std, "n_obs": 0},
                    "cdr": {"mean": benchmarks["cdr"].mean, "std": benchmarks["cdr"].std, "n_obs": 0},
                    "severity": {"mean": benchmarks["severity"].mean, "std": benchmarks["severity"].std, "n_obs": 0},
                },
                "corr_matrix": get_benchmark_correlation_matrix(),
            }

    hist_data = st.session_state.get(hist_key)
    if hist_data is not None:
        hist_dists = hist_data["hist_dists"]
        benchmarks = hist_data["benchmarks"]
        blended = hist_data["blended"]
        corr_matrix = hist_data["corr_matrix"]

        # Show comparison table: Historical vs Benchmark vs Blended
        comp_rows = []
        for var, label in [("cpr", "CPR"), ("cdr", "CDR"), ("severity", "Severity")]:
            bm = benchmarks[var]
            bl = blended[var]
            row = {"Variable": label, "Benchmark Mean": f"{bm.mean:.4f}", "Benchmark Std": f"{bm.std:.4f}"}
            if hist_dists is not None:
                hd = hist_dists[var]
                if not math.isnan(hd.mean):
                    row["Historical Mean"] = f"{hd.mean:.4f}"
                    row["Historical Std"] = f"{hd.std:.4f}"
                    row["N Obs"] = str(hd.n_observations)
                else:
                    row["Historical Mean"] = "Insufficient data"
                    row["Historical Std"] = "-"
                    row["N Obs"] = str(hd.n_observations)
            else:
                row["Historical Mean"] = "N/A"
                row["Historical Std"] = "N/A"
                row["N Obs"] = "0"
            row["Blended Mean"] = f"{bl['mean']:.4f}"
            row["Blended Std"] = f"{bl['std']:.4f}"
            comp_rows.append(row)

        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        with st.expander("Correlation Matrix", expanded=False):
            st.dataframe(
                correlation_matrix_to_dataframe(corr_matrix).style.format("{:.2f}"),
                use_container_width=True,
            )

        # ---------------------------------------------------------------
        # Step 2: Distribution parameter sliders
        # ---------------------------------------------------------------
        st.markdown("#### Step 2: Distribution Parameters (adjust if needed)")

        col1, col2 = st.columns(2)
        with col1:
            mc_cpr_mean = st.slider("CPR Mean", 0.01, 0.30,
                float(blended["cpr"]["mean"]), 0.005, format="%.3f", key="mc_cpr_mean")
            mc_cpr_std = st.slider("CPR Std Dev", 0.005, 0.10,
                float(blended["cpr"]["std"]), 0.005, format="%.3f", key="mc_cpr_std")
            mc_sev_mean = st.slider("Severity Mean", 0.10, 0.80,
                float(blended["severity"]["mean"]), 0.01, format="%.2f", key="mc_sev_mean")
            mc_sev_std = st.slider("Severity Std Dev", 0.02, 0.25,
                float(blended["severity"]["std"]), 0.01, format="%.2f", key="mc_sev_std")
        with col2:
            mc_cdr_mean = st.slider("CDR Mean", 0.005, 0.12,
                float(blended["cdr"]["mean"]), 0.005, format="%.3f", key="mc_cdr_mean")
            mc_cdr_std = st.slider("CDR Std Dev", 0.005, 0.06,
                float(blended["cdr"]["std"]), 0.005, format="%.3f", key="mc_cdr_std")
            mc_lag_mean = st.slider("Recovery Lag Mean (months)", 3, 30, 12, 1, key="mc_lag_mean")
            mc_lag_std = st.slider("Recovery Lag Std Dev", 1, 10, 4, 1, key="mc_lag_std")

        mc_paths = st.slider("MC Paths", 10, 2000, 500, 50)

        # ---------------------------------------------------------------
        # Step 3: Run Monte Carlo
        # ---------------------------------------------------------------
        st.markdown("#### Step 3: Run Monte Carlo")

        run_mc = st.button("Run Monte Carlo", type="primary", use_container_width=True)

        if run_mc:
            # Build distribution params
            dist_params = DistributionParams(
                cpr_mean=mc_cpr_mean, cpr_std=mc_cpr_std,
                cdr_mean=mc_cdr_mean, cdr_std=mc_cdr_std,
                severity_mean=mc_sev_mean, severity_std=mc_sev_std,
                recovery_lag_mean=float(mc_lag_mean), recovery_lag_std=float(mc_lag_std),
                correlation_matrix=corr_matrix,
            )

            # Sample
            with st.spinner(f"Sampling {mc_paths} correlated paths..."):
                sampler = MonteCarloSampler(dist_params, n_paths=int(mc_paths), seed=42)
                sampled_paths = sampler.sample()

            # Show sampled assumption histograms
            st.markdown("**Sampled Assumption Distributions**")
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                _plot_histogram(sampled_paths.cpr * 100, title="CPR", x_label="CPR (%)", bins=30, height=200)
            with h2:
                _plot_histogram(sampled_paths.cdr * 100, title="CDR", x_label="CDR (%)", bins=30, height=200)
            with h3:
                _plot_histogram(sampled_paths.severity * 100, title="Severity", x_label="Severity (%)", bins=30, height=200)
            with h4:
                _plot_histogram(sampled_paths.recovery_lag, title="Recovery Lag", x_label="Months", bins=15, height=200)

            # Run engine
            cfg = ProjectionConfig(
                as_of_date=pd.Timestamp(as_of),
                projection_months=int(projection_months),
                n_paths=int(mc_paths),
                seed=42,
            )

            progress_bar = st.progress(0, text="Running cashflow engine...")
            cashflows, raw_results = run_projection(
                tape.reset_index(drop=True), cfg, sampled_paths=sampled_paths
            )
            progress_bar.progress(80, text="Computing metrics...")

            path_metrics = compute_path_metrics(
                cashflows,
                as_of_date=cfg.as_of_date,
                original_balance=raw_results["original_balance"],
            )
            progress_bar.progress(100, text="Complete!")
            progress_bar.empty()

            # Store results
            st.session_state["mc_results"] = {
                "cashflows": cashflows,
                "raw_results": raw_results,
                "path_metrics": path_metrics,
                "sampled_paths": sampled_paths,
                "dist_params": dist_params,
            }

    # ---------------------------------------------------------------
    # MC PM Outputs (display after engine completes)
    # ---------------------------------------------------------------
    mc_results = st.session_state.get("mc_results")
    if mc_results is not None:
        path_metrics = mc_results["path_metrics"]
        cashflows = mc_results["cashflows"]
        raw_results = mc_results["raw_results"]

        st.divider()
        st.markdown("#### PM Outputs")
        _display_pm_outputs(cashflows, raw_results, path_metrics, deal_name_display)

    elif hist_data is not None:
        st.info("Click 'Run Monte Carlo' above to generate projections.")
    else:
        st.info("Click 'Compute Historical Distributions' to start the analysis pipeline.")
