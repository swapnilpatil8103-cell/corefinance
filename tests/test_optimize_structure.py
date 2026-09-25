from pathlib import Path

import numpy as np
import pytest
import yaml

from corefin.assumptions.loader import load_config
from corefin.assumptions.schema import RootConfig
from corefin.checks.corporate_checks import check_balance_sheet_balances
from corefin.checks.framework import run_checks
from corefin.optimize.structure import (
    build_opening_balance_sheet,
    build_structure,
    compute_closing_leverage,
    compute_entry_ebitda_mm,
    size_tranches,
)
from corefin.scenarios.generator import deterministic_drivers
from corefin.statements.corporate_model import run_corporate_model_with_debt
from corefin.timeline import Timeline
from tests.test_debt_integration import debt_config_dict

EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "example_midmarket.yaml"


def _optimizer_config_dict(n_periods: int = 6) -> dict:
    data = debt_config_dict(n_periods=n_periods)
    data["optimizer"] = {
        "decision_variables": [
            {"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 4.0},
            {"tranche_name": "Notes", "min_multiple": 0.5, "max_multiple": 2.0},
        ],
    }
    return data


def test_build_structure_requires_optimizer_config():
    config = RootConfig.model_validate(debt_config_dict(n_periods=4))
    assert config.optimizer is None
    with pytest.raises(ValueError, match="optimizer"):
        build_structure(config, {"TLB": 2.0})


def test_unknown_decision_variable_tranche_rejected_at_config_load():
    data = _optimizer_config_dict()
    data["optimizer"]["decision_variables"][0]["tranche_name"] = "DoesNotExist"
    with pytest.raises(Exception, match="unknown tranche"):
        RootConfig.model_validate(data)


def test_size_tranches_only_changes_decision_variables():
    config = RootConfig.model_validate(_optimizer_config_dict())
    entry_ebitda_mm = compute_entry_ebitda_mm(config)
    sized = size_tranches(config.tranches, {"TLB": 3.0}, entry_ebitda_mm)
    by_name = {t.name: t for t in sized}
    assert by_name["TLB"].size_mm == pytest.approx(3.0 * entry_ebitda_mm)
    # Revolver and Notes weren't decision variables in this call -> unchanged.
    original_by_name = {t.name: t for t in config.tranches}
    assert by_name["Revolver"].size_mm == original_by_name["Revolver"].size_mm
    assert by_name["Notes"].size_mm == original_by_name["Notes"].size_mm


def test_closing_leverage_excludes_revolver_and_matches_hand_calc():
    config = RootConfig.model_validate(_optimizer_config_dict())
    entry_ebitda_mm = compute_entry_ebitda_mm(config)
    sized = size_tranches(config.tranches, {"TLB": 3.0, "Notes": 1.0}, entry_ebitda_mm)
    leverage = compute_closing_leverage(sized, entry_ebitda_mm)
    assert leverage.total_leverage == pytest.approx(4.0)
    assert leverage.senior_leverage == pytest.approx(4.0)  # Notes is senior in this fixture
    assert leverage.total_debt_sources_mm == pytest.approx(4.0 * entry_ebitda_mm)


def test_equity_is_always_the_sources_and_uses_plug():
    config = RootConfig.model_validate(_optimizer_config_dict())
    for tlb_multiple in (1.0, 2.5, 4.0):
        candidate = build_structure(config, {"TLB": tlb_multiple, "Notes": 1.0})
        assert candidate.opening_balance_sheet.equity_mm == pytest.approx(
            candidate.sources_and_uses.sponsor_equity_mm
        )
        assert candidate.sources_and_uses.sponsor_equity_mm == pytest.approx(
            candidate.sources_and_uses.total_uses_mm
            - candidate.sources_and_uses.total_debt_sources_mm
        )


def test_goodwill_is_invariant_to_the_debt_equity_split():
    """Equity + total debt sources always sums to total uses regardless of the
    split, so the goodwill plug shouldn't move as leverage moves -- only the
    fixed purchase price/fees/balance-sheet facts determine it."""
    config = RootConfig.model_validate(_optimizer_config_dict())
    goodwills = [
        build_structure(config, {"TLB": m, "Notes": 1.0}).opening_balance_sheet.goodwill_mm
        for m in (1.0, 2.0, 3.0, 4.0)
    ]
    assert all(g == pytest.approx(goodwills[0]) for g in goodwills)


def test_balance_sheet_balances_for_a_range_of_candidates():
    config = RootConfig.model_validate(_optimizer_config_dict(n_periods=5))
    timeline = Timeline.annual(n_periods=5)
    for tlb_multiple, notes_multiple in [(1.0, 0.5), (2.5, 1.0), (4.0, 2.0)]:
        candidate = build_structure(config, {"TLB": tlb_multiple, "Notes": notes_multiple})
        drivers = deterministic_drivers(config, timeline)
        result = run_corporate_model_with_debt(
            config.company,
            candidate.opening_balance_sheet,
            candidate.tranches,
            config.waterfall,
            timeline,
            drivers,
        )
        run_checks([check_balance_sheet_balances(result.balance_sheet)])


def test_builder_reproduces_the_example_opening_balance_sheet_exactly():
    with EXAMPLE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["optimizer"] = {
        "decision_variables": [
            {"tranche_name": "TLB", "min_multiple": 2.0, "max_multiple": 5.0},
            {"tranche_name": "SeniorNotes", "min_multiple": 0.5, "max_multiple": 3.0},
        ],
    }
    config = RootConfig.model_validate(data)
    original_config = load_config(EXAMPLE_CONFIG_PATH)

    # TLB=375mm and SeniorNotes=175mm on a 100mm entry EBITDA -> 3.75x / 1.75x.
    candidate = build_structure(config, {"TLB": 3.75, "SeniorNotes": 1.75})

    assert candidate.opening_balance_sheet == original_config.opening_balance_sheet


def test_build_opening_balance_sheet_helper_matches_full_builder():
    config = RootConfig.model_validate(_optimizer_config_dict())
    candidate = build_structure(config, {"TLB": 2.0, "Notes": 1.0})
    rebuilt = build_opening_balance_sheet(config, candidate.sources_and_uses)
    assert rebuilt == candidate.opening_balance_sheet


def test_decision_values_stored_on_candidate():
    config = RootConfig.model_validate(_optimizer_config_dict())
    values = {"TLB": 2.5, "Notes": 1.5}
    candidate = build_structure(config, values)
    assert candidate.decision_values == values


def test_higher_leverage_decision_values_increase_sponsor_equity_never_negative():
    # Sanity guard: as leverage decreases (less debt), equity check should
    # increase (more cash needed from the sponsor), and never go negative
    # within these bounds.
    config = RootConfig.model_validate(_optimizer_config_dict())
    equities = []
    for m in (1.0, 2.0, 3.0, 4.0):
        candidate = build_structure(config, {"TLB": m, "Notes": 1.0})
        equities.append(candidate.sources_and_uses.sponsor_equity_mm)
    assert np.all(np.diff(equities) < 0)  # more TLB -> less equity needed
    assert all(e >= 0 for e in equities)
