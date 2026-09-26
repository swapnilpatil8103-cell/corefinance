"""Named deterministic stress scenarios: a one-off shock applied to the
deterministic (zero-vol) base case and run through the exact same model
pipeline as the Monte Carlo engine, so results are directly comparable.

Three composable shock types (`StressShockConfig` in assumptions/schema.py):
recession (a peak-year growth/margin hit that tapers back to zero over a
configurable recovery period), rate shock (a permanent, held base_rate
shift), and multiple compression (an exit-multiple shift). A named
scenario can combine any subset of these -- "combined downside" sets all
three at once.

Exit-multiple/rate interaction: if `scenario.advanced.exit_multiple_link`
is configured (Stage 2's structural link between the exit multiple and
exit-year fundamentals -- see scenarios/generator.py), a stress scenario
that includes a rate shock also applies that link's `beta_rate`
sensitivity to the exit multiple (`beta_rate * shock.rate_shock.bps`),
stacking additively with any explicit `multiple_compression` in the same
scenario. Without a rate shock component, or without the link configured,
a stress scenario's exit multiple is unaffected by this -- the link only
fires for the rate change the scenario actually tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.schema import (
    ExitMultipleLinkConfig,
    NamedStressScenarioConfig,
    RootConfig,
    StressShockConfig,
)
from corefin.metrics.covenants import evaluate_covenants
from corefin.metrics.credit_metrics import compute_credit_metrics
from corefin.optimize.structure import CandidateStructure
from corefin.scenarios.drivers import DriverSet
from corefin.scenarios.generator import deterministic_drivers
from corefin.simulate.distress import compute_distress_flags
from corefin.simulate.liquidity import compute_liquidity
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline
from corefin.transaction.returns import compute_exit_and_returns


def apply_stress_shock(
    base: DriverSet,
    timeline: Timeline,
    shock: StressShockConfig,
    exit_multiple_link: ExitMultipleLinkConfig | None = None,
) -> DriverSet:
    """Applies `shock`'s configured components on top of `base` (typically
    the deterministic zero-vol case). Any component left unset (None)
    leaves that part of the path untouched. `exit_multiple_link` is the
    root config's scenario.advanced.exit_multiple_link, if any -- see the
    module docstring for how it interacts with a rate shock."""
    revenue_growth = base.revenue_growth.copy()
    ebitda_margin = base.ebitda_margin.copy()
    base_rate = base.base_rate.copy()
    exit_multiple = base.exit_multiple.copy()

    if shock.recession is not None:
        r = shock.recession
        for t in range(timeline.n_periods):
            offset = t - r.start_year_index
            if offset < 0:
                continue
            fraction = (
                1.0 - offset / r.recovery_years if r.recovery_years > 0 else float(offset == 0)
            )
            fraction = max(0.0, fraction)
            if fraction <= 0.0:
                continue
            revenue_growth[:, t] += r.revenue_growth_hit * fraction
            ebitda_margin[:, t] += r.ebitda_margin_hit * fraction

    if shock.rate_shock is not None:
        rs = shock.rate_shock
        base_rate[:, rs.start_year_index :] += rs.bps
        if exit_multiple_link is not None:
            exit_multiple = exit_multiple + exit_multiple_link.beta_rate * rs.bps

    if shock.multiple_compression is not None:
        exit_multiple = exit_multiple + shock.multiple_compression.delta

    return DriverSet(
        revenue_growth=revenue_growth,
        ebitda_margin=ebitda_margin,
        capex_pct_revenue=base.capex_pct_revenue,
        nwc_pct_revenue=base.nwc_pct_revenue,
        base_rate=base_rate,
        exit_multiple=exit_multiple,
    )


@dataclass(frozen=True)
class StressScenarioResult:
    name: str
    irr: float
    moic: float
    min_liquidity_mm: float  # cash + undrawn revolver capacity
    min_cash_mm: float  # cash alone, kept separate since the two mean different things
    max_revolver_draw_pct: float
    covenant_breach_by_year: dict[str, np.ndarray]
    distress_by_year: np.ndarray
    distress_overall: bool


def run_stress_scenario(
    root_config: RootConfig,
    candidate: CandidateStructure,
    timeline: Timeline,
    scenario_config: NamedStressScenarioConfig,
) -> StressScenarioResult:
    """Runs one named stress scenario against `candidate` (typically the
    optimizer's recommended structure, or the input config's own structure
    -- see optimize/structure.py, reused rather than reimplemented here)."""
    base = deterministic_drivers(root_config, timeline)
    advanced = root_config.scenario.advanced
    exit_multiple_link = advanced.exit_multiple_link if advanced is not None else None
    shocked = apply_stress_shock(base, timeline, scenario_config.shocks, exit_multiple_link)

    result = run_corporate_model_with_debt(
        root_config.company,
        candidate.opening_balance_sheet,
        candidate.tranches,
        root_config.waterfall,
        timeline,
        shocked,
    )
    credit_metrics = compute_credit_metrics(
        result.debt_schedule,
        candidate.tranches,
        result.income_statement.ebitda,
        result.cash_flow_statement.capex,
        result.balance_sheet.cash,
        result.balance_sheet.total_debt,
    )
    covenant_results = evaluate_covenants(
        root_config.covenants, credit_metrics, timeline, result.debt_schedule, candidate.tranches
    )
    exit_result = compute_exit_and_returns(
        candidate.sources_and_uses.sponsor_equity_mm,
        root_config.transaction.exit_year_index,
        shocked.exit_multiple,
        result.income_statement.ebitda,
        result.balance_sheet.total_debt,
        result.balance_sheet.cash,
        result.cash_flow_statement.dividends,
    )
    distress = compute_distress_flags(result.debt_schedule, credit_metrics)[0]
    liquidity = compute_liquidity(
        result.balance_sheet.cash, result.debt_schedule, candidate.tranches
    )

    revolver = next(t for t in candidate.tranches if t.is_revolver)
    revolver_draw_pct = (
        result.debt_schedule.ending_balance[revolver.name][0] / revolver.size_mm
        if revolver.size_mm > 0
        else np.zeros(timeline.n_periods)
    )

    return StressScenarioResult(
        name=scenario_config.name,
        irr=float(exit_result.irr[0]),
        moic=float(exit_result.moic[0]),
        min_liquidity_mm=float(liquidity[0].min()),
        min_cash_mm=float(result.balance_sheet.cash[0].min()),
        max_revolver_draw_pct=float(revolver_draw_pct.max()),
        covenant_breach_by_year={cov.name: cov.breach[0] for cov in covenant_results},
        distress_by_year=distress,
        distress_overall=bool(np.any(distress)),
    )


def run_stress_scenarios(
    root_config: RootConfig,
    candidate: CandidateStructure,
    timeline: Timeline,
    scenario_configs: list[NamedStressScenarioConfig],
) -> list[StressScenarioResult]:
    return [run_stress_scenario(root_config, candidate, timeline, sc) for sc in scenario_configs]
