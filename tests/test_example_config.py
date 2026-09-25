"""End-to-end regression test for configs/example_midmarket.yaml.

This is the config the CLI's README walkthrough and manual smoke tests use;
it's easy for a config edit (like the covenant recalibration) to silently
break the balance-sheet/debt-schedule integrity of the example without any
other test catching it, since no other test loads this file directly.
"""

from pathlib import Path

from corefin.assumptions.loader import load_config
from corefin.checks.corporate_checks import check_balance_sheet_balances, check_cash_ties_to_cfs
from corefin.checks.debt_checks import (
    check_debt_rollforward,
    check_revolver_bounds,
    check_sweep_priority_respected,
)
from corefin.checks.framework import run_checks
from corefin.scenarios.generator import generate_stochastic_drivers
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline

EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "example_midmarket.yaml"


def test_example_config_runs_end_to_end_and_passes_all_integrity_checks():
    config = load_config(EXAMPLE_CONFIG_PATH)
    timeline = Timeline.annual(
        n_periods=config.timeline.n_periods,
        n_historical=config.timeline.n_historical,
        start_year=config.timeline.start_year,
    )
    drivers = generate_stochastic_drivers(config, timeline, seed=config.scenario.random_seed)
    result = run_corporate_model_with_debt(
        config.company,
        config.opening_balance_sheet,
        config.tranches,
        config.waterfall,
        timeline,
        drivers,
    )

    run_checks(
        [
            check_balance_sheet_balances(result.balance_sheet),
            check_cash_ties_to_cfs(
                result.balance_sheet,
                result.cash_flow_statement,
                config.opening_balance_sheet.cash_mm,
            ),
            check_debt_rollforward(result.debt_schedule),
        ]
    )
    revolver = next(t for t in config.tranches if t.is_revolver)
    assert check_revolver_bounds(result.debt_schedule, revolver).passed
    assert check_sweep_priority_respected(result.debt_schedule, config.tranches).passed
