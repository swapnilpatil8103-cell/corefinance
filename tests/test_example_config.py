"""End-to-end regression test for configs/example_midmarket.yaml.

This is the config the CLI's README walkthrough and manual smoke tests use;
it's easy for a config edit (like the covenant recalibration) to silently
break the balance-sheet/debt-schedule integrity of the example without any
other test catching it, since no other test loads this file directly.
"""

from pathlib import Path

import pytest

from corefin.assumptions.loader import load_config
from corefin.checks.corporate_checks import check_balance_sheet_balances, check_cash_ties_to_cfs
from corefin.checks.debt_checks import (
    check_debt_rollforward,
    check_revolver_bounds,
    check_sweep_priority_respected,
)
from corefin.checks.framework import run_checks
from corefin.optimize.search import run_optimization
from corefin.scenarios.generator import generate_stochastic_drivers
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline

EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "example_midmarket.yaml"
EXAMPLE_STRESS_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "example_midmarket_stress.yaml"
)


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


@pytest.mark.slow
def test_example_config_optimizer_end_to_end_passes_integrity_checks_for_every_candidate():
    """Runs the full optimizer (grid + refine + confirm) on the example
    config, same as `corefin optimize`, and checks every evaluated
    candidate's balance sheet and debt schedule -- not just the recommended
    one -- to catch a structural bug that only a particular decision-value
    combination would trigger."""
    config = load_config(EXAMPLE_CONFIG_PATH)
    timeline = Timeline.annual(
        n_periods=config.timeline.n_periods,
        n_historical=config.timeline.n_historical,
        start_year=config.timeline.start_year,
    )
    result = run_optimization(config, timeline)

    evaluations = list(result.grid_result.evaluations)
    if result.refinement_result is not None:
        evaluations += result.refinement_result.evaluations
    evaluations.append(result.confirmation)

    checked = 0
    for evaluation in evaluations:
        if evaluation.result is None:
            continue
        checked += 1
        run_checks(
            [
                check_balance_sheet_balances(evaluation.result.balance_sheet),
                check_cash_ties_to_cfs(
                    evaluation.result.balance_sheet,
                    evaluation.result.cash_flow_statement,
                    config.opening_balance_sheet.cash_mm,
                ),
                check_debt_rollforward(evaluation.result.debt_schedule),
            ]
        )
        revolver = next(t for t in evaluation.candidate.tranches if t.is_revolver)
        assert check_revolver_bounds(evaluation.result.debt_schedule, revolver).passed
        assert check_sweep_priority_respected(
            evaluation.result.debt_schedule, evaluation.candidate.tranches
        ).passed
    assert checked > 0


def test_example_midmarket_yaml_has_no_advanced_scenario_generation():
    """example_midmarket.yaml must keep using the simple generator (the one
    corefin run/optimize's output is pinned to) -- the richer generator
    lives only in example_midmarket_stress.yaml."""
    config = load_config(EXAMPLE_CONFIG_PATH)
    assert config.scenario.advanced is None


def test_example_stress_config_runs_end_to_end_in_advanced_mode():
    """The stress variant must actually exercise scenario.advanced (not
    silently fall back to simple mode) and still pass every balance sheet
    and debt integrity check."""
    config = load_config(EXAMPLE_STRESS_CONFIG_PATH)
    assert config.scenario.advanced is not None

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


def test_example_configs_are_identical_except_for_scenario_advanced_and_header_comment():
    """example_midmarket_stress.yaml is meant to be "the same deal plus an
    advanced block" -- everything else (tranches, covenants, optimizer,
    simulate) should round-trip identically."""
    base = load_config(EXAMPLE_CONFIG_PATH)
    stress = load_config(EXAMPLE_STRESS_CONFIG_PATH)

    assert stress.company == base.company
    assert stress.opening_balance_sheet == base.opening_balance_sheet
    assert stress.transaction == base.transaction
    assert stress.tranches == base.tranches
    assert stress.covenants == base.covenants
    assert stress.waterfall == base.waterfall
    assert stress.optimizer == base.optimizer
    assert stress.simulate == base.simulate
    # Same scenario settings apart from the advanced block itself.
    assert stress.scenario.model_copy(update={"advanced": None}) == base.scenario
