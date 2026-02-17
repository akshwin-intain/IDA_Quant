# Collateral Projection Engine – Design Overview (Aligned to Current Implementation)

## 1. Objective (What the code does today)

**Objective**
Given a (static) loan tape and scenario-level behavioral assumptions (**CPR**, **CDR**, **Severity/LGD**, **Recovery lag**), simulate monthly collateral cashflows and produce portfolio-level PM outputs used for decision support:

- **Life**: Weighted Average Life (WAL)
- **Credit**: cumulative loss (% of original balance), loss timing
- **Risk**: changes in WAL and loss under stressed assumptions (re-run comparison)

**One-line summary**
Loan tape + constant hazard assumptions → Monte Carlo event simulation → monthly cashflows → PM decision metrics.

> Note on terminology: the implementation is **not purely deterministic** end-to-end.  
> Each path uses **stochastic default/prepay events** (Monte Carlo), and then applies **deterministic cashflow accounting** within the month.

---

## 2. Inputs (As implemented)

### 2.1 Static loan tape (facts)

The engine enforces a schema, but it only *requires* these fields to run (see `_prepare_tape_for_engine` in `modelling/engine.py`):

- Loan ID
- First Payment Date
- Maturity Date
- Original Term
- Original Principal Balance
- Current Principal Balance
- Current Interest Rate

Other tape columns may exist (property, FICO, LTV, etc.) but are **not used** by the current baseline behavior model.

### 2.2 Behavioral assumptions (scenario inputs)

Current baseline behavior is **ConstantHazardModel** (same rate for all loans and months):

- **CPR (annual)** in \([0,1]\)
- **CDR (annual)** in \([0,1]\)
- **Severity / LGD** in \([0,1]\)

These are converted internally into monthly hazards:

- CPR → SMM (monthly prepay hazard)
- CDR → MDR (monthly default hazard)

### 2.3 Projection configuration

From `ProjectionConfig` (`modelling/config.py`):

- `projection_months`: maximum simulated horizon per loan (cap)
- `n_paths`: number of Monte Carlo paths (scenarios)
- `seed`: RNG seed (fixed to 1 in `streamlit_app.py` for reproducibility)
- `recovery_lag_months`: delay between default and recovery cash receipt
- `event_priority`: tie-break ordering if default and prepay happen in the same month (rare)

---

## 3. Engine outputs vs PM outputs (As implemented)

### 3.1 Engine outputs (raw, auditable)

**Purpose**: show the mechanics and accounting.

The engine returns a `cashflows` DataFrame with monthly cashflows **per path** (portfolio aggregation by default in Streamlit):

- begin_balance
- interest
- scheduled_principal
- prepayment
- default_principal
- loss (recognized at default time: `LGD * default_principal`)
- recovery (paid after `recovery_lag_months`)
- end_balance
- principal (scheduled + prepay + recovery)
- total_cashflow

> The engine can also return **loan-level** rows if `aggregate="loan"` is used.

### 3.2 PM outputs (decision-ready)

**Purpose**: support comparison and judgment of risk.

The engine builds PM outputs from cashflows:

**(A) Across-path summaries (tails / expectation)**

- Expected total cashflow by date (mean across paths)
- (Available in code) percentiles by date across paths
- (Available in code) distribution of path totals

**(B) Life / credit / risk metrics**
Computed in `build_pm_risk_metrics` (`modelling/pm.py`):

- **WAL (years)**: computed from *expected principal by date* (timing-weighted principal)
- **Cumulative loss (% of original balance)**: cumulative expected loss divided by original balance
- **Loss timing**: expected loss by date and cumulative expected loss by date

In the Streamlit UI (`streamlit_app.py`), these are displayed as:

- WAL metric
- cumulative loss % metric
- expected principal by month chart/table
- expected loss & cumulative loss charts/tables

---

## 4. End-to-end projection flow (As implemented)

### Step 1: Prepare tape

- Select/enforce required columns
- Coerce numeric/date types
- Fill missing current balance and rate defaults

### Step 2: Compute scheduled cashflows (deterministic accounting)

For each loan and month, compute:

- monthly rate = annual_rate / 12
- level payment (fully-amortizing, simplified)
- interest = begin_balance * monthly_rate
- scheduled_principal = payment − interest (capped at balance)

### Step 3: Simulate events (Monte Carlo)

For each loan/month/path:

- draw randoms → decide **default** with MDR hazard
- else decide **prepay** with SMM hazard
- else continue amortizing

If default occurs:

- loss at default time = `LGD * default_principal`
- recovery cash amount = `(1 − LGD) * default_principal`
- recovery is paid at `t + recovery_lag_months`

### Step 4: Aggregate to portfolio outputs

Streamlit runs `aggregate="portfolio"`:

- sums cashflow components across loans for each path and month

### Step 5: Produce PM outputs

From the portfolio cashflows:

- expected principal / loss by date (mean across paths)
- WAL from timing-weighted expected principal
- cumulative loss % of original balance

### Step 6: Stress / sensitivity (implemented via re-run)

The Streamlit app supports an **optional stress scenario**:

- re-run the engine with stressed CPR/CDR/LGD
- report **ΔWAL** and **Δcumulative loss %**
- overlay expected loss timing (baseline vs stress)

---

## 5. Practical interpretation for PM use

### Life

- **WAL**: “expected life” proxy — higher WAL means cash comes back later (often more extension risk).

### Credit

- **Cumulative loss %**: total expected credit loss as a fraction of original balance.
- **Loss timing**: *when* losses occur matters for liquidity, leverage, and tranche risk.

### Risk / Sensitivities

- **ΔWAL under stressed CPR**: prepay slows → WAL rises (extension risk); prepay speeds → WAL falls (contraction risk).
- **Loss under stressed CDR/LGD**: defaults/severity up → higher cumulative loss and earlier loss timing.




