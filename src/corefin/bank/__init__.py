"""Bank M&A CET1 & Accretion Simulator (Project #2), Stage 2: a standalone
single-bank statement model -- balance sheet, income statement, and
regulatory capital (CET1/RWA/leverage) -- for ONE bank under ONE scenario,
consuming the Credit-Loss Forecasting Engine's `CreditLossProjection`
(`corefin.credit.interface`) for loan balances/NCOs/provisions/allowance
rather than reimplementing any of that.

Deal mechanics (purchase accounting, pro forma combination) live in
`corefin.ma`, not here -- this module only produces the two standalone
(acquirer, target) projections that `corefin.ma` combines.

Requires the optional `corefin[fig]` extra (shared with `corefin.credit`).
"""
