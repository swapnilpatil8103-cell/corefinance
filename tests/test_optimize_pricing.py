import pytest

from corefin.assumptions.schema import RootConfig
from corefin.optimize.pricing import (
    apply_pricing_grid,
    build_priced_structure,
    check_market_capacity,
)
from corefin.optimize.structure import build_structure, compute_entry_ebitda_mm, size_tranches
from tests.test_debt_integration import debt_config_dict


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
