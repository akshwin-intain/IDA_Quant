# Intain_Quant — Architecture

## Corrected Flow

```
data/raw/*.csv (665-col monthly history)
    │
    ├──→ data_prep/loader.py          (read CSVs)
    ├──→ data_prep/tape_builder.py    (collapse → one row per loan)
    │         ↓
    │    data/tapes/consolidated.csv  (static loan tape)
    │
    ├──→ distributions/historical.py  (compute actual CPR/CDR/Severity time series)
    ├──→ distributions/benchmarks.py  (industry fallback values)
    ├──→ distributions/correlation.py (correlation matrix)
    │         ↓
    │    DistributionParams (mean, std, min, max, correlations)
    │         ↓
    └──→ distributions/sampler.py     (Monte Carlo: generate N × 4 assumption table)
              ↓
         SampledPaths (1000 rows of CPR, CDR, Severity, RecoveryLag)
              ↓
         engine/runner.py             (loop 1000 paths)
              │
              ├── behaviors/scenario.py  (per-path: convert assumptions → hazard arrays)
              ├── engine/cashflow.py     (deterministic math: scheduled payment, amort)
              └── engine/events.py       (random draws: which loans default/prepay)
              ↓
         1,000 sets of monthly cashflows
              ↓
         pm/metrics.py                (WAL, Loss%, Expected Life — PER PATH)
              ↓
         pm/aggregator.py             (percentile tables across 1,000 paths)
              ↓
         pm/decisions.py              (probability statements, flags)
              ↓
         app/streamlit_app.py         (display to PM)
```

## Module Responsibilities

| Package | Purpose | Changes frequently? |
|---------|---------|-------------------|
| core/ | Schema, config, utilities | Rarely |
| data_prep/ | Load, clean, validate data | When new deals arrive |
| distributions/ | Estimate + sample assumptions | When models improve |
| behaviors/ | Convert assumptions → hazards | When adding loan-level models |
| engine/ | Cashflow math + MC runner | Almost never (math is math) |
| pm/ | Metrics, aggregation, decisions | When PMs want new outputs |
| models/ | Statistical + ML models | Active development (Phase 2-3) |
| app/ | Streamlit UI | Frequently |

## Two Layers of Randomness

1. **Assumption-level** (sampler.py): WHICH economic future? Different CPR/CDR per path.
2. **Event-level** (events.py): WHICH loans get hit? Different random draws per loan.

Both are necessary. Layer 1 gives you scenario variation. Layer 2 gives you idiosyncratic risk.
