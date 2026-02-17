# Intain Quant - Monte Carlo Collateral Engine

A comprehensive Monte Carlo simulation framework for analyzing residential mortgage portfolios with stochastic prepayment, default, and loss severity modeling.

## Overview

This project provides a production-grade collateral engine that:

- **Processes loan tapes** from monthly deal reports into standardized formats
- **Computes historical distributions** of CPR (Conditional Prepayment Rate), CDR (Conditional Default Rate), and Loss Severity
- **Generates correlated Monte Carlo scenarios** (1,000+ paths) with realistic assumption dependencies
- **Projects deterministic cashflows** for each path under different behavior models
- **Aggregates results** into percentile distributions for Portfolio Manager decision-making
- **Provides interactive dashboards** for scenario analysis and stress testing

## Project Structure

```
Intain_Quant/
├── core/              # Core schemas, configuration, utilities
├── data_prep/         # Data loading, tape building, validation
├── distributions/     # Historical analysis, benchmarks, correlation
├── behaviors/         # Behavior models (constant, scenario, statistical)
├── engine/            # Cashflow projection engine
├── pm/                # Portfolio Manager metrics & decisions
├── models/            # Statistical models for CPR/CDR/Severity
├── app/               # Streamlit dashboard application
├── notebooks/         # Jupyter notebooks for analysis
└── tests/             # Unit and integration tests
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Intain_Quant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or with development tools:
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Data

Place your deal CSV files in `data/raw/`:
```
data/raw/JP_MORGAN_2024_CES1.csv
data/raw/JP_MORGAN_2024_CES2.csv
data/raw/JP_MORGAN_2025_CES1.csv
```

### 2. Build Loan Tape

```python
from data_prep.tape_builder import build_consolidated_tape

tape = build_consolidated_tape(
    deal_paths=["data/raw/JP_MORGAN_2024_CES1.csv"],
    as_of_date="2024-12-31"
)
tape.to_csv("data/tapes/consolidated_loan_tape.csv", index=False)
```

### 3. Run Monte Carlo Simulation

```python
from core.config import ProjectionConfig
from engine.runner import MonteCarloRunner

config = ProjectionConfig(
    as_of_date="2024-12-31",
    n_paths=1000,
    horizon_months=360,
    seed=42
)

runner = MonteCarloRunner(config)
results = runner.run(tape)
```

### 4. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501` to explore:
- Loan tape quality and statistics
- Historical CPR/CDR/Severity distributions
- Monte Carlo scenario results
- Portfolio Manager decision views

## Key Features

### Behavior Models

1. **Constant Hazard Model**: Baseline using fixed CPR/CDR/Severity
2. **Scenario Model**: Stochastic paths from correlated distributions
3. **Statistical Model**: Logistic regression on FICO/LTV/DTI (Phase 2)
4. **ML Model**: XGBoost/Neural Network enhancements (Phase 3)

### Portfolio Metrics

- Weighted Average Life (WAL)
- Expected Cumulative Loss
- Prepayment timing distribution
- Default timing distribution
- Scenario stress flags

### Monte Carlo Framework

- Correlated assumption sampling (Cholesky decomposition)
- Path-level event simulation
- Percentile aggregation (P10, P50, P90)
- Historical vs. benchmark distribution comparison

## Data Requirements

### Input Files

- Monthly deal reports with loan-level history
- Standard fields: `loan_id`, `current_upb`, `interest_rate`, `maturity_date`, `fico`, `ltv`, `dti`, `prepay_flag`, `default_flag`, `loss_amount`

### Output Files

- `consolidated_loan_tape.csv`: One-row-per-loan static tape
- `mc_results.parquet`: Monte Carlo cashflow projections
- `pm_metrics.csv`: Aggregated portfolio metrics

## Testing

Run the test suite:
```bash
pytest tests/
```

With coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

## Development

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run formatters:
```bash
black .
isort .
```

Run linters:
```bash
flake8 .
mypy .
```

### Notebooks

Jupyter notebooks for exploratory analysis:
- `01_data_exploration.ipynb`: Initial data quality checks
- `02_historical_distributions.ipynb`: CPR/CDR/Severity analysis
- `03_engine_validation.ipynb`: Cashflow engine validation
- `04_model_development.ipynb`: Statistical model development

## Documentation

- [Architecture](docs/ARCHITECTURE.md): System design and module flow
- [Collateral Engine Design](docs/COLLATERAL_ENGINE_DESIGN.md): Technical specifications
- [Data Dictionary](docs/DATA_DICTIONARY.md): Field mappings and definitions

## License

MIT License - see LICENSE file for details

## Contact

For questions or support, contact the Intain Analytics Team at analytics@intainft.com
