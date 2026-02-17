"""
Statistical and ML models for behavioral assumption estimation.

Siddhartha's 5 models:
  Model 1: CPR estimation (cpr_model.py)        — survival/logistic regression
  Model 2: CDR/PD estimation (cdr_model.py)      — logistic regression
  Model 3: Severity estimation (severity_model.py) — linear/quantile regression
  Model 4: Monte Carlo (distributions/sampler.py)  — already implemented
  Model 5: ML enhancement (ml_model.py)           — XGBoost/neural nets

Models 1-3 feed into the distributions layer (better mean/std estimates).
Model 5 captures non-linear patterns and interactions.

STATUS: Stubs only. Implementation is Phase 2/3.
"""
