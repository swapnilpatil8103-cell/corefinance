from pathlib import Path

import pytest

from corefin.assumptions.loader import load_config
from corefin.assumptions.schema import RateType, RootConfig, TrancheConfig, TrancheType
from corefin.optimize.pricing import (
    apply_pricing_grid,
    build_priced_structure,
    check_market_capacity,
    pricing_sanity_warnings,
)
from corefin.optimize.structure import build_structure, compute_entry_ebitda_mm, size_tranches
from corefin.scenarios.generator import deterministic_drivers
from corefin.timeline import Timeline
from tests.test_debt_integration import debt_config_dict

EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "example_midmarket.yaml"


def _config_with_pricing(**pricing_overrides) -> RootConfig:
    data = debt_config_dict(n_periods=5)
    data["optimizer"] = {
        "decision_variables": [
            {"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 6.0},
            {"tranche_name": "Notes", "min_multiple": 0.5, "max_multiple": 3.0},
        ],
        "pricing": {
            "tranches": [
                {
                    "tranche_name": "TLB",
                    "basis": "total_leverage",
                    "leverage_threshold": 4.0,
                    "spread_bps_per_turn": 0.0025,
                    "upfront_fee_pct_per_turn": 0.0,
                    "oid_pct_per_turn": 0.0,
                },
                {
                    "tranche_name": "Notes",
                    "basis": "total_leverage",
                    "leverage_threshold": 4.0,
                    "spread_bps_per_turn": 0.005,  # fixed_rate tranche -> bumps fixed_rate
                },
            ],
            **pricing_overrides,
        },
    }
    return RootConfig.model_validate(data)


def test_no_bump_at_or_below_threshold():
    config = _config_with_pricing()
    entry_ebitda_mm = compute_entry_ebitda_mm(config)
    # TLB=3.0x + Notes=1.0x = 4.0x total leverage, exactly at the threshold.
    sized = size_tranches(config.tranches, {"TLB": 3.0, "Notes": 1.0}, entry_ebitda_mm)
    candidate = build_structure(config, {"TLB": 3.0, "Notes": 1.0})
    priced = apply_pricing_grid(sized, candidate.leverage, config.optimizer.pricing)
    by_name = {t.name: t for t in priced}
    original_by_name = {t.name: t for t in config.tranches}
    assert by_name["TLB"].spread == pytest.approx(original_by_name["TLB"].spread)
    assert by_name["Notes"].fixed_rate == pytest.approx(original_by_name["Notes"].fixed_rate)


def test_linear_bump_above_threshold():
    config = _config_with_pricing()
    entry_ebitda_mm = compute_entry_ebitda_mm(config)
    # TLB=4.0x + Notes=2.0x = 6.0x total leverage -> 2.0 turns above the 4.0x threshold.
    sized = size_tranches(config.tranches, {"TLB": 4.0, "Notes": 2.0}, entry_ebitda_mm)
    candidate = build_structure(config, {"TLB": 4.0, "Notes": 2.0})
    assert candidate.leverage.total_leverage == pytest.approx(6.0)
    priced = apply_pricing_grid(sized, candidate.leverage, config.optimizer.pricing)
    by_name = {t.name: t for t in priced}
    original_by_name = {t.name: t for t in config.tranches}

    turns_above = 2.0
    expected_tlb_spread = original_by_name["TLB"].spread + 0.0025 * turns_above
    expected_notes_fixed_rate = original_by_name["Notes"].fixed_rate + 0.005 * turns_above
    assert by_name["TLB"].spread == pytest.approx(expected_tlb_spread)
    assert by_name["Notes"].fixed_rate == pytest.approx(expected_notes_fixed_rate)


def test_bump_scales_linearly_with_turns():
    config = _config_with_pricing()
    entry_ebitda_mm = compute_entry_ebitda_mm(config)
    spreads = []
    for tlb_multiple in (4.0, 5.0, 6.0):  # 0, 1, 2 turns above the 4.0x threshold
        sized = size_tranches(config.tranches, {"TLB": tlb_multiple, "Notes": 0.0}, entry_ebitda_mm)
        candidate = build_structure(config, {"TLB": tlb_multiple, "Notes": 0.0})
        priced = apply_pricing_grid(sized, candidate.leverage, config.optimizer.pricing)
        spreads.append(next(t for t in priced if t.name == "TLB").spread)
    diffs = [spreads[i + 1] - spreads[i] for i in range(len(spreads) - 1)]
    assert diffs[0] == pytest.approx(diffs[1])
    assert diffs[0] == pytest.approx(0.0025)  # exactly one turn's worth of bps per turn


def test_untracked_tranche_never_repriced():
    config = _config_with_pricing()
    entry_ebitda_mm = compute_entry_ebitda_mm(config)
    sized = size_tranches(config.tranches, {"TLB": 6.0, "Notes": 3.0}, entry_ebitda_mm)
    candidate = build_structure(config, {"TLB": 6.0, "Notes": 3.0})
    priced = apply_pricing_grid(sized, candidate.leverage, config.optimizer.pricing)
    revolver = next(t for t in priced if t.name == "Revolver")
    original_revolver = next(t for t in config.tranches if t.name == "Revolver")
    assert revolver.spread == pytest.approx(original_revolver.spread)


def test_pricing_fee_bump_changes_financing_fees_and_goodwill():
    config = _config_with_pricing(
        tranches=[
            {
                "tranche_name": "TLB",
                "basis": "total_leverage",
                "leverage_threshold": 4.0,
                "upfront_fee_pct_per_turn": 0.01,
            }
        ]
    )
    below = build_priced_structure(config, {"TLB": 3.0, "Notes": 1.0})  # 4.0x, at threshold
    above = build_priced_structure(config, {"TLB": 5.0, "Notes": 1.0})  # 6.0x, 2 turns above

    assert (
        above.sources_and_uses.financing_fees_and_oid_mm
        > below.sources_and_uses.financing_fees_and_oid_mm
    )
    # Goodwill is the plug; a bigger financing fee (more DFC) with the same
    # cash/nwc/ppe/other_liab must be offset by a correspondingly bigger
    # goodwill plug to keep the balance sheet balanced.
    assert above.opening_balance_sheet.deferred_financing_costs_mm > (
        below.opening_balance_sheet.deferred_financing_costs_mm
    )


def test_build_priced_structure_without_pricing_config_matches_unpriced():
    data = debt_config_dict(n_periods=4)
    data["optimizer"] = {
        "decision_variables": [{"tranche_name": "TLB", "min_multiple": 1.0, "max_multiple": 4.0}],
    }
    config = RootConfig.model_validate(data)
    assert config.optimizer.pricing.tranches == []
    unpriced = build_structure(config, {"TLB": 3.0})
    priced = build_priced_structure(config, {"TLB": 3.0})
    assert priced == unpriced


def test_market_capacity_per_tranche_violation():
    config = _config_with_pricing(
        tranches=[
            {
                "tranche_name": "TLB",
                "basis": "total_leverage",
                "leverage_threshold": 4.0,
                "spread_bps_per_turn": 0.0025,
                "market_capacity_mm": 300.0,
            },
            {
                "tranche_name": "Notes",
                "basis": "total_leverage",
                "leverage_threshold": 4.0,
                "spread_bps_per_turn": 0.005,
            },
        ]
    )
    within_capacity = build_priced_structure(config, {"TLB": 2.0, "Notes": 1.0})  # 200mm
    over_capacity = build_priced_structure(config, {"TLB": 4.0, "Notes": 1.0})  # 400mm

    assert check_market_capacity(within_capacity, config.optimizer.pricing).feasible
    result = check_market_capacity(over_capacity, config.optimizer.pricing)
    assert not result.feasible
    assert any("TLB" in v for v in result.violations)


def test_total_market_capacity_violation():
    config = _config_with_pricing(total_market_capacity_mm=350.0)
    within = build_priced_structure(config, {"TLB": 2.0, "Notes": 1.0})  # 300mm total
    over = build_priced_structure(config, {"TLB": 3.0, "Notes": 1.0})  # 400mm total

    assert check_market_capacity(within, config.optimizer.pricing).feasible
    result = check_market_capacity(over, config.optimizer.pricing)
    assert not result.feasible
    assert any("total debt" in v for v in result.violations)


def test_no_capacity_limits_means_always_feasible():
    config = _config_with_pricing()
    candidate = build_priced_structure(config, {"TLB": 6.0, "Notes": 3.0})
    assert check_market_capacity(candidate, config.optimizer.pricing).feasible


def test_secured_leverage_basis_steps_up_at_the_right_threshold():
    # TLB (term_loan_b) is secured by default, Notes (senior_notes) is not --
    # secured leverage here is TLB alone, so only TLB's own multiple should
    # move it across the 3.0x threshold, regardless of how much Notes is
    # layered on top.
    config = _config_with_pricing(
        tranches=[
            {
                "tranche_name": "TLB",
                "basis": "secured_leverage",
                "leverage_threshold": 3.0,
                "spread_bps_per_turn": 0.004,
            }
        ]
    )
    original_spread = next(t for t in config.tranches if t.name == "TLB").spread

    at_threshold = build_priced_structure(config, {"TLB": 3.0, "Notes": 2.5})
    tlb_at_threshold = next(t for t in at_threshold.tranches if t.name == "TLB")
    assert tlb_at_threshold.spread == pytest.approx(original_spread)

    # Notes alone pushing total leverage above 3.0x must NOT bump TLB, since
    # TLB's pricing basis is secured (TLB-only) leverage, not total.
    notes_only_above = build_priced_structure(config, {"TLB": 3.0, "Notes": 2.9})
    tlb_notes_above = next(t for t in notes_only_above.tranches if t.name == "TLB")
    assert tlb_notes_above.spread == pytest.approx(original_spread)

    above_threshold = build_priced_structure(config, {"TLB": 4.0, "Notes": 0.0})
    tlb_above = next(t for t in above_threshold.tranches if t.name == "TLB")
    assert tlb_above.spread == pytest.approx(original_spread + 0.004 * 1.0)


def _secured_tlb(spread: float = 0.05, rate_floor: float = 0.01) -> TrancheConfig:
    return TrancheConfig(
        name="TLB",
        tranche_type=TrancheType.TERM_LOAN_B,
        size_mm=100.0,
        rate_type=RateType.FLOATING,
        spread=spread,
        rate_floor=rate_floor,
    )


def _unsecured_notes(**overrides) -> TrancheConfig:
    defaults = dict(
        name="Notes",
        tranche_type=TrancheType.SENIOR_NOTES,
        size_mm=50.0,
        rate_type=RateType.FIXED,
        fixed_rate=0.12,
    )
    defaults.update(overrides)
    return TrancheConfig(**defaults)


def test_pricing_warning_fires_when_unsecured_is_not_priced_above_secured():
    # TLB all-in at base_rate=0.045: max(0.045, 0.01) + 0.05 = 0.095 -- notes
    # at 0.08 fixed is cheaper, which is backwards for unsecured debt.
    tlb = _secured_tlb()
    notes = _unsecured_notes(fixed_rate=0.08)
    warnings = pricing_sanity_warnings([tlb, notes], base_rate=0.045)
    assert any("Notes" in w and "TLB" in w and "priced at or below" in w for w in warnings)


def test_pricing_warning_silent_when_unsecured_is_priced_above_secured():
    tlb = _secured_tlb()
    notes = _unsecured_notes(fixed_rate=0.12)  # above TLB's 9.5% all-in
    warnings = pricing_sanity_warnings([tlb, notes], base_rate=0.045)
    assert warnings == []


def test_pricing_warning_fires_for_sweep_eligible_unsecured_tranche():
    tlb = _secured_tlb()
    notes = _unsecured_notes(fixed_rate=0.12, cash_sweep_eligible=True, sweep_priority=2)
    warnings = pricing_sanity_warnings([tlb, notes], base_rate=0.045)
    assert any("cash-sweep eligible" in w for w in warnings)
    # Properly priced above secured -- no separate rate warning.
    assert not any("priced at or below" in w for w in warnings)


def test_example_config_produces_no_pricing_warnings():
    config = load_config(EXAMPLE_CONFIG_PATH)
    timeline = Timeline.annual(
        n_periods=config.timeline.n_periods,
        n_historical=config.timeline.n_historical,
        start_year=config.timeline.start_year,
    )
    base_drivers = deterministic_drivers(config, timeline)
    warnings = pricing_sanity_warnings(config.tranches, float(base_drivers.base_rate[0, 0]))
    assert warnings == []
