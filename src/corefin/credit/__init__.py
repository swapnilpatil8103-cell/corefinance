"""Credit-Loss Forecasting Engine (Project #7): forecasts US bank credit
losses by loan category under macro scenarios -- NCOs, NPLs, provisions
and the loan-loss allowance over a 9-quarter horizon. Built on the shared
`timeline.Timeline` (quarterly), the `checks.framework` integrity-check
primitives, and the same "pure numpy functions + a small orchestrator"
pattern used throughout the rest of corefin. Its output (`CreditLossProjection`)
feeds Project #2 (Bank M&A CET1 & Accretion Simulator).

Requires the optional `corefin[fig]` extra (requests, pyarrow, scikit-learn,
statsmodels) -- kept out of the core install since the LBO/optimizer/
simulate track doesn't need them.
"""
