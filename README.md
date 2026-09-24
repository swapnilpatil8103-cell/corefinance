# corefin

A vectorized core library for quantitative finance / investment banking
projects, built as the shared foundation for:

1. **Financing Structure Optimizer** — picks the debt/equity mix that
   maximizes sponsor returns subject to leverage/coverage/covenant
   constraints, calling this model many times inside an optimizer.
2. **Sponsor LBO Monte Carlo Engine** — runs thousands of operating
   scenarios to produce distributions of IRR, MOIC and covenant breaches.
3. **Credit-Loss Forecasting Engine** (later, bank-specific).
4. **Bank M&A CET1 & Accretion Simulator** (later, bank-specific).

Projects 3 and 4 need a bank statement model, which is structurally
different from a corporate model. It isn't built here, but the shared
`timeline`, `assumptions`/`scenarios`, and `checks` layers are designed so
it can be added as a sibling module — see [Where the bank model plugs
in](#where-the-bank-model-plugs-in) below.

## Install & run

```bash
uv sync
uv run pytest                       # fast tests (excludes the slow marker)
uv run pytest -m slow               # the 10k-scenario performance test
uv run ruff check . && uv run ruff format --check .
uv run corefin run --config configs/example_midmarket.yaml
```

The `run` command prints a summary (sources & uses, IRR/MOIC distribution
across scenarios, covenant breach probabilities) and exports one scenario's
full statements, debt schedule and credit metrics to `output.xlsx`
(`--output` to change the path, `--scenario` to pick which one).

## Architecture

```
src/corefin/
  timeline.py             Period grid: n_periods, period lengths, projection flags
  assumptions/            Pydantic config schema + YAML loader
  scenarios/               DriverSet (vectorized inputs) + deterministic/stochastic generators
  statements/              Income statement, balance sheet, cash flow, orchestration
  debt/                    Tranche config, amortization/fee sizing, waterfall, circularity solver
  metrics/                 Credit metrics, covenant evaluation
  transaction/              Sources & uses, exit valuation, IRR/MOIC
  checks/                   Reusable integrity-check framework + concrete checks
  io/                       Excel export
  cli.py                    `corefin run` / `corefin version`
```

**Every time-series quantity is a numpy array of shape `(n_scenarios,
n_periods)`.** A single deterministic case is `n_scenarios = 1`. There are
no Python loops over scenarios anywhere; the only loop is over periods
inside the debt module, because the debt schedule and its interest
circularity are genuinely path-dependent (each period's ending balance
depends on the prior period's).

### Data flow

```
YAML config --pydantic--> RootConfig
RootConfig + Timeline --> DriverSet (scenarios/generator.py)
DriverSet --> revenue/EBITDA/D&A/capex/NWC (statements/corporate_model.py)
  --> debt/circularity.run_debt_schedule (loops periods; vectorized fixed-point per period)
  --> income statement, balance sheet, cash flow statement
  --> metrics/credit_metrics, metrics/covenants
  --> transaction/sources_uses (entry), transaction/returns (exit, IRR/MOIC)
```

`run_corporate_model_no_debt` exists mainly as a way to validate the
statement plumbing (revenue → EBITDA → NOL/tax → NI → BS → CFS) in
isolation, without the debt module's complexity. `run_corporate_model_with_debt`
is the real entry point for any project with leverage.

## Sign & unit conventions

Documented once, in `assumptions/schema.py`, and repeated here:

- All dollar amounts are **$mm**.
- All rates, margins, growth figures and percentages are **decimal
  fractions** (`0.20`, not `20` or `"20%"`).
- All multiples (entry/exit, leverage) are **bare floats** (`5.5x` → `5.5`).
- Cash inflows to the company are positive; interest expense, fees, and
  coverage-metric denominators are positive costs.
- In the cash flow statement, `debt_draws` is a positive inflow,
  `debt_repayments` is already negative-signed (an outflow), and
  `dividends` is a positive outflow (sign-flipped internally).

## The debt module and interest circularity

Interest on the **average balance** for period *t* depends on period *t*'s
ending balance, which depends on cash available for debt service, which
depends on net income, which depends on interest for period *t*. This is
circular, but only **within** a period — the prior period's ending balance
is already fixed by the time period *t* starts. `debt/circularity.py`
solves this with a vectorized fixed-point iteration per period (guess
interest → run the waterfall → recompute interest from the resulting
balances → repeat until the change is below `circularity_tolerance`, or
raise `CircularityNotConvergedError` after `circularity_max_iterations`).
Set `waterfall.interest_mode: beginning_balance` to skip the circularity
entirely (non-circular, useful for speed or debugging) — it converges in
exactly two passes since there's nothing to iterate on.

The waterfall itself (`debt/waterfall.py`) runs, per period, in this order:
minimum cash balance → mandatory amortization (capped at the outstanding
balance) → revolver draw if there's a shortfall (capped at the undrawn
commitment; a shortfall that still can't be covered is *flagged*, not
silently allowed) → revolver paydown → optional cash sweep by priority
(`sweep_priority`, lower = swept first) at a configurable `sweep_pct`.

## Adding a new tranche type

1. Add the new value to `TrancheType` in `assumptions/schema.py`.
2. If it needs new fields (e.g. a step-up coupon), add them to
   `TrancheConfig` with a `model_validator` enforcing any cross-field
   rules (see the existing fixed/floating-rate and PIK/revolver checks for
   the pattern).
3. If it needs special seniority treatment for leverage metrics, add it to
   (or leave it out of) `SENIOR_TRANCHE_TYPES` in `metrics/credit_metrics.py`.
4. The waterfall (`debt/waterfall.py`), schedule sizing (`debt/schedule.py`)
   and circularity solver (`debt/circularity.py`) are all written generically
   against `TrancheConfig` fields (`is_revolver`, `pik_fraction`,
   `cash_sweep_eligible`, `sweep_priority`, ...) — no changes needed there
   unless the new type needs genuinely new waterfall *behavior* (e.g. a
   payment-in-kind toggle that can flip mid-life), in which case add a field
   and branch on it in `circularity._recompute_interest` / `waterfall.run_period_waterfall`.

## A known simplification worth knowing about

`OpeningBalanceSheet` cannot see the tranche list (it's validated
independently by pydantic), so it can't enforce "assets = liabilities +
equity" on its own — that check depends on whether you're about to run
`run_corporate_model_no_debt` (which always treats debt as zero) or
`run_corporate_model_with_debt` (where each tranche is already funded at
face value as of period 0's start, so `equity_mm` must be sized to cover
that too). This is exactly what a real deal's Sources & Uses does
automatically; until the Financing Structure Optimizer or Monte Carlo
project wires `transaction/sources_uses.py`'s output directly into an
opening balance sheet, treat `OpeningBalanceSheet` as "whatever numbers are
consistent with the model you're about to run" and let
`checks.check_balance_sheet_balances` tell you at runtime if they aren't.
`configs/example_midmarket.yaml` shows a debt-inclusive example that
balances (verify the identity: `cash + nwc + ppe + goodwill +
deferred_financing_costs == other_liabilities + equity + sum(non-revolver
tranche sizes)`).

## Where the bank model plugs in

A bank statement model (Credit-Loss Forecasting, CET1/Accretion) is
structurally different — no revenue/COGS/EBITDA, no LBO debt waterfall;
instead a balance sheet driven by loan/deposit growth, NIM, provisions and
regulatory capital ratios. It reuses:

- `timeline.Timeline` — the same period grid.
- `scenarios.drivers.DriverSet` / `scenarios.generator` — the same
  deterministic/stochastic scenario interface, extended with bank-specific
  drivers (NIM, provision rate, RWA density, deposit growth, ...) instead of
  (or alongside) the corporate drivers.
- `checks.framework` — the same `CheckResult`/`check_close_to_zero`/
  `check_bounds` primitives, with new bank-specific checks (e.g. CET1 ratio
  bounds, loan-loss-reserve roll-forward) living in a new
  `checks/bank_checks.py`.

It does **not** reuse `debt/`, `statements/income_statement.py`, or
`transaction/` — those are corporate/LBO-specific. It would add a sibling
`statements/bank_statement.py` and a bank-specific `metrics/` module,
following the same "pure numpy functions + a small orchestrator" pattern
used throughout `statements/corporate_model.py`.

## Testing conventions

- `pytest.mark.slow` is excluded by default (`addopts = "-m 'not slow'"` in
  `pyproject.toml`); run it explicitly with `pytest -m slow`.
- Every check in `checks/` returns a `CheckResult` reporting *which*
  scenario/period pairs failed, not just pass/fail — `run_checks([...])`
  raises with that detail if anything failed.
- Config fixtures live in `tests/conftest.py` (a minimal schema-validation
  fixture) and `tests/test_debt_integration.py::debt_config_dict` (a
  self-consistent, debt-inclusive fixture reused by the metrics, covenants
  and transaction tests).
