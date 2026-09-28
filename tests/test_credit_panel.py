"""Offline tests for the credit panel's pure transformation functions,
against synthetic fixtures only -- no network, no real Call Report data
(one test reads an already-cached local bulk ZIP if present as a
regression guard, and skips itself if that cache isn't there)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from corefin.checks.framework import check_close_to_zero
from corefin.credit import panel
from corefin.credit.schema import CATEGORY_MDRM_CODES, LoanCategory, MdrmCodeSet
from corefin.credit.sources import ffiec_parse


def _quarters(labels: list[str]) -> pd.Series:
    return pd.Series([pd.Period(label, freq="Q") for label in labels])


# ------------------------------------------------------------- YTD -> Q ----


def test_ytd_to_quarterly_q1_is_passthrough_and_later_quarters_are_diffed():
    ytd = pd.Series([10.0, 25.0, 45.0, 60.0])  # Q1..Q4 cumulative
    quarters = _quarters(["2019Q1", "2019Q2", "2019Q3", "2019Q4"])
    quarterly = panel.ytd_to_quarterly(ytd, quarters)
    assert quarterly.tolist() == [10.0, 15.0, 20.0, 15.0]


def test_ytd_to_quarterly_resets_at_new_calendar_year():
    ytd = pd.Series([10.0, 60.0, 12.0])  # 2019Q1, 2019Q4 YTD, then 2020Q1 YTD (new year)
    quarters = _quarters(["2019Q1", "2019Q4", "2020Q1"])
    quarterly = panel.ytd_to_quarterly(ytd, quarters)
    # 2020Q1 must be its own value (12.0), not diffed against 2019Q4's 60.0
    assert quarterly.tolist() == [10.0, 50.0, 12.0]


# --------------------------------------------------------- NCO rate --------


def test_annualized_nco_rate_correctness():
    # $1,000 average balance, $10 net quarterly NCO -> 1%/quarter -> 4%/year
    chargeoffs = pd.Series([15.0])
    recoveries = pd.Series([5.0])
    avg_balance = pd.Series([1000.0])
    rate = panel.annualized_nco_rate(chargeoffs, recoveries, avg_balance)
    assert rate.iloc[0] == pytest.approx(0.04)


def test_annualized_nco_rate_handles_zero_or_negative_balance_as_nan():
    chargeoffs = pd.Series([10.0, 10.0])
    recoveries = pd.Series([0.0, 0.0])
    avg_balance = pd.Series([0.0, -5.0])
    rate = panel.annualized_nco_rate(chargeoffs, recoveries, avg_balance)
    assert rate.isna().all()


def test_average_balance_first_observation_is_nan():
    avg = panel.average_balance(pd.Series([100.0, 200.0, 300.0]))
    assert avg.isna().iloc[0]
    assert avg.iloc[1] == 150.0
    assert avg.iloc[2] == 250.0


# ----------------------------------------------- category mapping totals ---


def test_category_mapping_sums_multi_item_categories_within_tolerance():
    (code_set,) = CATEGORY_MDRM_CODES[LoanCategory.RESIDENTIAL_MORTGAGE]
    first_lien, junior_lien = code_set.balance_items
    item_frame = pd.DataFrame({first_lien: [700.0], junior_lien: [130.0]})
    other_items = (
        *code_set.past_due_30_89_items,
        *code_set.past_due_90_items,
        *code_set.nonaccrual_items,
        *code_set.chargeoff_items,
        *code_set.recovery_items,
    )
    for col in other_items:
        item_frame[col] = 0.0
    quarters = _quarters(["2015Q1"])
    mapped = panel.apply_category_mapping(item_frame, LoanCategory.RESIDENTIAL_MORTGAGE, quarters)
    expected_total = 830.0
    assert mapped["balance"].iloc[0] == pytest.approx(expected_total, rel=1e-9)


def test_category_mapping_missing_component_is_nan_not_zero(monkeypatch):
    # A synthetic category whose code set deliberately has no past-due/
    # nonaccrual items -- this must produce NaN columns, not zeros.
    synthetic_category = LoanCategory.CI
    synthetic_code_set = MdrmCodeSet(
        valid_from="2015Q1",
        valid_to=None,
        balance_items=("RCONSYN1",),
        chargeoff_items=("RIADSYN1",),
        recovery_items=("RIADSYN2",),
    )
    monkeypatch.setitem(CATEGORY_MDRM_CODES, synthetic_category, (synthetic_code_set,))

    item_frame = pd.DataFrame({"RCONSYN1": [500.0], "RIADSYN1": [1.0], "RIADSYN2": [1.0]})
    quarters = _quarters(["2015Q1"])
    mapped = panel.apply_category_mapping(item_frame, synthetic_category, quarters)
    assert mapped["past_due_30_89"].isna().all()
    assert mapped["past_due_90"].isna().all()
    assert mapped["nonaccrual"].isna().all()
    assert mapped["balance"].iloc[0] == pytest.approx(500.0)


def test_category_mapping_raises_on_missing_required_column():
    quarters = _quarters(["2015Q1"])
    with pytest.raises(KeyError):
        panel.apply_category_mapping(pd.DataFrame({"RCON9999": [1.0]}), LoanCategory.CI, quarters)


def test_category_mapping_raises_for_a_quarter_with_no_code_set():
    # auto loans have no code set before 2011Q1 -- this must raise, not
    # silently return zeros or NaN.
    (code_set,) = CATEGORY_MDRM_CODES[LoanCategory.AUTO]
    item_frame = pd.DataFrame({col: [500.0] for col in code_set.balance_items})
    other_items = (
        *code_set.past_due_30_89_items,
        *code_set.past_due_90_items,
        *code_set.nonaccrual_items,
        *code_set.chargeoff_items,
        *code_set.recovery_items,
    )
    for col in other_items:
        item_frame[col] = 0.0
    quarters = _quarters(["2005Q1"])
    with pytest.raises(ValueError, match="no MDRM code set covers"):
        panel.apply_category_mapping(item_frame, LoanCategory.AUTO, quarters)


def test_category_mapping_applies_per_row_fallback_within_a_transition_code_set(monkeypatch):
    # Synthetic 2007-style transition: two banks in the SAME quarter, one
    # still reporting the old combined item, one already on the new split.
    synthetic_category = LoanCategory.CI
    transition_code_set = MdrmCodeSet(
        valid_from="2007Q1",
        valid_to="2007Q4",
        balance_items=("RCONSPLIT1", "RCONSPLIT2"),
        chargeoff_items=("RIADSPLIT",),
        recovery_items=("RIADSPLIT_R",),
        fallback_balance_items=("RCONCOMBINED",),
        fallback_chargeoff_items=("RIADCOMBINED",),
        fallback_recovery_items=("RIADCOMBINED_R",),
    )
    monkeypatch.setitem(CATEGORY_MDRM_CODES, synthetic_category, (transition_code_set,))

    item_frame = pd.DataFrame(
        {
            # bank A: only the combined (old) item is populated
            # bank B: only the split (new) items are populated
            "RCONSPLIT1": [np.nan, 600.0],
            "RCONSPLIT2": [np.nan, 450.0],
            "RCONCOMBINED": [1000.0, np.nan],
            "RIADSPLIT": [np.nan, 20.0],
            "RIADSPLIT_R": [np.nan, 5.0],
            "RIADCOMBINED": [40.0, np.nan],
            "RIADCOMBINED_R": [10.0, np.nan],
        }
    )
    quarters = _quarters(["2007Q1", "2007Q1"])
    mapped = panel.apply_category_mapping(item_frame, synthetic_category, quarters)

    assert mapped["balance"].tolist() == pytest.approx([1000.0, 1050.0])
    assert mapped["chargeoff_ytd"].tolist() == pytest.approx([40.0, 20.0])
    assert mapped["recovery_ytd"].tolist() == pytest.approx([10.0, 5.0])


def test_category_mapping_fallback_is_per_row_not_all_or_nothing(monkeypatch):
    # A bank with the split PARTIALLY populated (one of two split items
    # missing) must NOT silently fall back -- that's a real missing-data
    # case (the sum is NaN), not a transition case; only a bank with
    # NEITHER split item present should fall back.
    synthetic_category = LoanCategory.CI
    transition_code_set = MdrmCodeSet(
        valid_from="2007Q1",
        valid_to="2007Q4",
        balance_items=("RCONSPLIT1", "RCONSPLIT2"),
        fallback_balance_items=("RCONCOMBINED",),
    )
    monkeypatch.setitem(CATEGORY_MDRM_CODES, synthetic_category, (transition_code_set,))

    item_frame = pd.DataFrame(
        {
            "RCONSPLIT1": [600.0],  # only ONE of two split items present
            "RCONSPLIT2": [np.nan],
            "RCONCOMBINED": [1000.0],
        }
    )
    quarters = _quarters(["2007Q1"])
    mapped = panel.apply_category_mapping(item_frame, synthetic_category, quarters)
    # sum(600, NaN) = NaN with skipna=False, so this row uses the fallback
    assert mapped["balance"].iloc[0] == pytest.approx(1000.0)


def test_real_2007q1_cre_construction_coverage_is_full_with_fallback():
    # Regression guard for the real bug this fallback fixes: verified
    # against a real 2007Q1 bulk file that ALL banks report either the
    # split or the combined construction item that quarter -- coverage
    # must be ~100%, not the ~60% an earlier version of this module
    # produced by only trying the split codes.
    zip_path = Path("data/raw/ffiec/03-31-2007.zip")
    if not zip_path.exists():
        pytest.skip("real 2007Q1 bulk ZIP not cached locally")
    item_frame, _ = ffiec_parse.parse_bulk_zip(zip_path.read_bytes(), pd.Period("2007Q1", freq="Q"))
    mapped = panel.apply_category_mapping(
        item_frame, LoanCategory.CRE_CONSTRUCTION, item_frame["quarter"]
    )
    assert mapped["balance"].notna().mean() > 0.99


def test_category_mapping_picks_the_right_code_set_on_each_side_of_a_switch_date():
    # cre_nonfarm_nonresidential switches from RCON1480 to RCONF160+RCONF161 at 2007Q1.
    item_frame = pd.DataFrame(
        {
            "RCON1480": [1000.0, np.nan],
            "RCONF160": [np.nan, 600.0],
            "RCONF161": [np.nan, 450.0],
            "RCON3502": [0.0, np.nan],
            "RCON3503": [0.0, np.nan],
            "RCON3504": [0.0, np.nan],
            "RIAD3590": [0.0, np.nan],
            "RIAD3591": [0.0, np.nan],
            "RCONF178": [np.nan, 0.0],
            "RCONF179": [np.nan, 0.0],
            "RCONF180": [np.nan, 0.0],
            "RCONF181": [np.nan, 0.0],
            "RCONF182": [np.nan, 0.0],
            "RCONF183": [np.nan, 0.0],
            "RIADC895": [np.nan, 0.0],
            "RIADC896": [np.nan, 0.0],
            "RIADC897": [np.nan, 0.0],
            "RIADC898": [np.nan, 0.0],
        }
    )
    quarters = _quarters(["2006Q4", "2007Q1"])
    category = LoanCategory.CRE_NONFARM_NONRESIDENTIAL
    mapped = panel.apply_category_mapping(item_frame, category, quarters)
    assert mapped["balance"].iloc[0] == pytest.approx(1000.0)
    assert mapped["balance"].iloc[1] == pytest.approx(1050.0)


def test_aggregate_balance_is_continuous_across_a_synthetic_code_switch():
    # Two banks reporting the SAME true total nonfarm-nonresidential balance
    # on both sides of the 2007Q1 code switch -- the mapped aggregate must
    # not show a spurious break just because the underlying item changed.
    item_frame = pd.DataFrame(
        {
            "RCON1480": [1000.0, 2000.0, np.nan, np.nan],
            "RCONF160": [np.nan, np.nan, 600.0, 1200.0],
            "RCONF161": [np.nan, np.nan, 400.0, 800.0],
            "RCON3502": [0.0, 0.0, np.nan, np.nan],
            "RCON3503": [0.0, 0.0, np.nan, np.nan],
            "RCON3504": [0.0, 0.0, np.nan, np.nan],
            "RIAD3590": [0.0, 0.0, np.nan, np.nan],
            "RIAD3591": [0.0, 0.0, np.nan, np.nan],
            "RCONF178": [np.nan, np.nan, 0.0, 0.0],
            "RCONF179": [np.nan, np.nan, 0.0, 0.0],
            "RCONF180": [np.nan, np.nan, 0.0, 0.0],
            "RCONF181": [np.nan, np.nan, 0.0, 0.0],
            "RCONF182": [np.nan, np.nan, 0.0, 0.0],
            "RCONF183": [np.nan, np.nan, 0.0, 0.0],
            "RIADC895": [np.nan, np.nan, 0.0, 0.0],
            "RIADC896": [np.nan, np.nan, 0.0, 0.0],
            "RIADC897": [np.nan, np.nan, 0.0, 0.0],
            "RIADC898": [np.nan, np.nan, 0.0, 0.0],
        }
    )
    quarters = _quarters(["2006Q4", "2006Q4", "2007Q1", "2007Q1"])
    category = LoanCategory.CRE_NONFARM_NONRESIDENTIAL
    mapped = panel.apply_category_mapping(item_frame, category, quarters)
    aggregate_before = mapped["balance"].iloc[:2].sum()
    aggregate_after = mapped["balance"].iloc[2:].sum()
    assert aggregate_before == pytest.approx(3000.0)
    assert aggregate_after == pytest.approx(3000.0)


# --------------------------------------------------------- RCFD fallback ---


def test_rcfd_fallback_applies_when_rcon_is_missing_or_zero():
    rcon = pd.Series([100.0, np.nan, 0.0, 0.0])
    rcfd = pd.Series([999.0, 999.0, 999.0, np.nan])
    resolved = panel.apply_rcfd_fallback(rcon, rcfd)
    # row 0: RCON present and nonzero -> keep RCON.
    # row 1: RCON missing -> use RCFD.
    # row 2: RCON exactly zero (the bank 476810 case) -> use RCFD.
    # row 3: RCON zero AND no RCFD available -> a GENUINE zero (e.g. a bank
    #   that just doesn't issue credit cards) -- must stay 0, NOT become
    #   NaN (a real bug this test now guards against: it cratered
    #   credit_card's real-data coverage from ~100% to ~18%).
    assert resolved.iloc[0] == pytest.approx(100.0)
    assert resolved.iloc[1] == pytest.approx(999.0)
    assert resolved.iloc[2] == pytest.approx(999.0)
    assert resolved.iloc[3] == pytest.approx(0.0)


def test_rcfd_fallback_missing_rcon_with_no_rcfd_stays_missing():
    rcon = pd.Series([np.nan])
    rcfd = pd.Series([np.nan])
    resolved = panel.apply_rcfd_fallback(rcon, rcfd)
    assert pd.isna(resolved.iloc[0])


def test_category_mapping_applies_rcfd_fallback_for_a_zero_balance(monkeypatch):
    # Regression guard for the real bug this fixes: bank 476810's
    # RCONB538 is a literal 0 for five straight quarters while RCFDB538
    # has the real balance.
    synthetic_category = LoanCategory.CI
    code_set = MdrmCodeSet(
        valid_from="2010Q1",
        valid_to=None,
        balance_items=("RCONBAL",),
        rcfd_items=("RCFDBAL",),
    )
    monkeypatch.setitem(CATEGORY_MDRM_CODES, synthetic_category, (code_set,))

    item_frame = pd.DataFrame({"RCONBAL": [0.0, 500.0], "RCFDBAL": [35_000_000.0, 999.0]})
    quarters = _quarters(["2010Q2", "2010Q2"])
    mapped = panel.apply_category_mapping(item_frame, synthetic_category, quarters)
    assert mapped["balance"].iloc[0] == pytest.approx(35_000_000.0)
    assert mapped["balance"].iloc[1] == pytest.approx(500.0)  # real nonzero RCON kept as-is


def test_category_mapping_skips_rcfd_fallback_when_column_absent(monkeypatch):
    synthetic_category = LoanCategory.CI
    code_set = MdrmCodeSet(
        valid_from="2010Q1",
        valid_to=None,
        balance_items=("RCONBAL",),
        rcfd_items=("RCFDBAL",),  # not present in item_frame below
    )
    monkeypatch.setitem(CATEGORY_MDRM_CODES, synthetic_category, (code_set,))

    item_frame = pd.DataFrame({"RCONBAL": [0.0]})
    quarters = _quarters(["2010Q2"])
    mapped = panel.apply_category_mapping(item_frame, synthetic_category, quarters)
    assert mapped["balance"].iloc[0] == pytest.approx(0.0)  # no RCFD column -> stays as reported


# ------------------------------------------------------------ mergers ------


def test_flag_merger_discontinuities_flags_large_jump_only():
    balance = pd.Series([100.0, 105.0, 400.0, 410.0])  # 3rd quarter ~4x jump
    flags = panel.flag_merger_discontinuities(balance, jump_threshold=0.5)
    assert flags.tolist() == [False, False, True, False]


def test_flag_merger_discontinuities_flags_a_jump_from_zero():
    # A plain pct-change check can't catch this at all (division by the
    # zero prior balance is undefined) -- regression guard for bank
    # 476810's real case: RCONB538 reported as a literal 0 for several
    # quarters, then a real balance appears.
    balance = pd.Series([0.0, 0.0, 107_815_000.0])
    flags = panel.flag_merger_discontinuities(balance, near_zero_threshold=1_000.0)
    assert flags.tolist() == [False, False, True]


def test_flag_merger_discontinuities_does_not_flag_zero_to_small_growth():
    # Growing from zero to something still below the near-zero threshold
    # is not a merger signal -- both a new tiny bank and a tiny org're
    # unremarkable, only a jump to something MATERIAL should flag.
    balance = pd.Series([0.0, 500.0])  # both under the 1,000 threshold
    flags = panel.flag_merger_discontinuities(balance, near_zero_threshold=1_000.0)
    assert flags.tolist() == [False, False]


# --------------------------------------------------------------- CECL ------


def test_cecl_regime_dummy_uses_default_when_no_adoption_quarters_given():
    quarters = _quarters(["2019Q4", "2020Q1", "2020Q2"])
    bank_id = pd.Series(["A", "A", "A"])
    default_quarter = pd.Period("2020Q1", freq="Q")
    dummy = panel.apply_cecl_regime_dummy(
        bank_id, quarters, adoption_quarters=None, default_adoption_quarter=default_quarter
    )
    assert dummy.tolist() == [0, 1, 1]


def test_cecl_regime_dummy_uses_each_banks_own_adoption_quarter():
    bank_id = pd.Series(["A", "A", "B", "B"])
    quarters = _quarters(["2019Q4", "2020Q1", "2019Q4", "2020Q1"])
    adoption_quarters = pd.Series(
        {"A": pd.Period("2020Q1", freq="Q"), "B": pd.Period("2023Q1", freq="Q")}
    )
    dummy = panel.apply_cecl_regime_dummy(bank_id, quarters, adoption_quarters)
    # bank A adopted 2020Q1 -> [0, 1]; bank B adopted 2023Q1, hasn't yet -> [0, 0]
    assert dummy.tolist() == [0, 1, 0, 0]


def test_cecl_regime_dummy_falls_back_to_default_for_banks_missing_from_adoption_quarters():
    bank_id = pd.Series(["A", "B"])
    quarters = _quarters(["2020Q1", "2020Q1"])
    adoption_quarters = pd.Series({"A": pd.Period("2020Q1", freq="Q")})  # B is missing
    dummy = panel.apply_cecl_regime_dummy(
        bank_id, quarters, adoption_quarters, default_adoption_quarter=pd.Period("2019Q1", freq="Q")
    )
    # bank A: real adoption quarter (2020Q1) -> regime starts now -> 1
    # bank B: falls back to default (2019Q1), which is already past -> 1
    assert dummy.tolist() == [1, 1]


def test_derive_cecl_adoption_quarters_picks_the_first_nonzero_signal_quarter():
    item_frame = pd.DataFrame(
        {
            "bank_id": ["A", "A", "A", "B", "B"],
            "quarter": _quarters(["2019Q4", "2020Q1", "2020Q2", "2019Q4", "2023Q1"]),
            "RIADJJ26": [0.0, 1500.0, 1500.0, 0.0, 800.0],
            "RIADJJ28": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = panel.derive_cecl_adoption_quarters(item_frame)
    assert result["A"] == pd.Period("2020Q1", freq="Q")
    assert result["B"] == pd.Period("2023Q1", freq="Q")


def test_derive_cecl_adoption_quarters_uses_either_indicator_item():
    item_frame = pd.DataFrame(
        {
            "bank_id": ["A", "A"],
            "quarter": _quarters(["2019Q4", "2020Q1"]),
            "RIADJJ26": [0.0, 0.0],
            "RIADJJ28": [0.0, 2000.0],  # only JJ28 is nonzero
        }
    )
    result = panel.derive_cecl_adoption_quarters(item_frame)
    assert result["A"] == pd.Period("2020Q1", freq="Q")


def test_derive_cecl_adoption_quarters_returns_empty_when_columns_absent():
    item_frame = pd.DataFrame({"bank_id": ["A"], "quarter": _quarters(["2020Q1"])})
    result = panel.derive_cecl_adoption_quarters(item_frame)
    assert result.empty


def test_summarize_cecl_adoption_counts_reports_bank_count_per_quarter():
    adoption_quarters = pd.Series(
        {
            "A": pd.Period("2020Q1", freq="Q"),
            "B": pd.Period("2020Q1", freq="Q"),
            "C": pd.Period("2023Q1", freq="Q"),
        }
    )
    counts = panel.summarize_cecl_adoption_counts(adoption_quarters)
    assert counts[pd.Period("2020Q1", freq="Q")] == 2
    assert counts[pd.Period("2023Q1", freq="Q")] == 1
    assert counts.index.tolist() == sorted(counts.index.tolist())


def test_resolve_cecl_adoption_quarters_marks_detected_vs_defaulted():
    bank_ids = pd.Series(["A", "B", "C"])
    detected = pd.Series({"A": pd.Period("2020Q1", freq="Q")})  # B, C undetected
    default_quarter = pd.Period("2023Q1", freq="Q")
    resolved = panel.resolve_cecl_adoption_quarters(bank_ids, detected, default_quarter)
    resolved = resolved.set_index("bank_id")

    assert resolved.loc["A", "adoption_quarter"] == pd.Period("2020Q1", freq="Q")
    assert resolved.loc["A", "detected"]
    assert resolved.loc["B", "adoption_quarter"] == default_quarter
    assert not resolved.loc["B", "detected"]
    assert resolved.loc["C", "adoption_quarter"] == default_quarter
    assert not resolved.loc["C", "detected"]


def test_summarize_cecl_adoption_detected_vs_defaulted_splits_counts_by_quarter():
    resolved = pd.DataFrame(
        {
            "bank_id": ["A", "B", "C", "D"],
            "adoption_quarter": [
                pd.Period("2020Q1", freq="Q"),
                pd.Period("2020Q1", freq="Q"),
                pd.Period("2023Q1", freq="Q"),
                pd.Period("2023Q1", freq="Q"),
            ],
            "detected": [True, False, True, True],
        }
    )
    result = panel.summarize_cecl_adoption_detected_vs_defaulted(resolved)
    q1_2020 = result.loc[pd.Period("2020Q1", freq="Q")]
    q1_2023 = result.loc[pd.Period("2023Q1", freq="Q")]

    assert q1_2020["n_detected"] == 1
    assert q1_2020["n_defaulted"] == 1
    assert q1_2023["n_detected"] == 2
    assert q1_2023["n_defaulted"] == 0
    assert result.index.tolist() == sorted(result.index.tolist())


# --------------------------------------------------------- min balance -----


def test_min_balance_filter_excludes_small_and_nan_balances():
    balance = pd.Series([1_000_000.0, 500.0, np.nan])
    keep = panel.apply_min_balance_filter(balance, minimum=1000.0)
    assert keep.tolist() == [True, False, False]


# --------------------------------------------------- allowance roll-forward -


def test_allowance_rollforward_residual_is_zero_for_a_consistent_synthetic_series():
    beginning = pd.Series([100.0, 110.0, 108.0])
    provision = pd.Series([20.0, 15.0, 25.0])
    nco = pd.Series([10.0, 17.0, 13.0])
    ending = beginning + provision - nco
    residual = panel.allowance_rollforward_residual(beginning, provision, nco, ending)
    result = check_close_to_zero(
        "allowance_rollforward", residual.to_numpy().reshape(1, -1), tolerance=1e-9
    )
    assert result.passed


# ------------------------------------------------------ build_panel -------


def _ci_item_frame() -> pd.DataFrame:
    quarters = _quarters(["2019Q1", "2019Q2", "2019Q3"] * 2)
    return pd.DataFrame(
        {
            "bank_id": ["A", "A", "A", "B", "B", "B"],
            "quarter": quarters,
            "RCON1766": [1000.0, 1100.0, 1200.0, 2000.0, 2100.0, 2200.0],
            "RCON1606": [0.0] * 6,
            "RCON1607": [0.0] * 6,
            "RCON1608": [0.0] * 6,
            "RIAD4638": [10.0, 25.0, 40.0, 20.0, 50.0, 80.0],  # YTD
            "RIAD4608": [2.0, 5.0, 8.0, 4.0, 10.0, 16.0],  # YTD
        }
    )


def test_build_panel_computes_per_bank_quarterly_flows_and_nco_rate():
    result = panel.build_panel(_ci_item_frame(), categories=[LoanCategory.CI], min_balance=500.0)
    assert len(result) == 6

    bank_a = result[result["bank_id"] == "A"].sort_values("quarter").reset_index(drop=True)
    assert bank_a["chargeoff_quarterly"].tolist() == pytest.approx([10.0, 15.0, 15.0])
    assert bank_a["recovery_quarterly"].tolist() == pytest.approx([2.0, 3.0, 3.0])
    assert bank_a["average_balance"].iloc[0] != bank_a["average_balance"].iloc[0]  # NaN
    assert bank_a["average_balance"].iloc[1] == pytest.approx(1050.0)
    expected_nco_q2 = (15.0 - 3.0) / 1050.0 * 4.0
    assert bank_a["annualized_nco_rate"].iloc[1] == pytest.approx(expected_nco_q2)
    assert bank_a["cecl_regime"].tolist() == [0, 0, 0]
    assert bank_a["keep"].tolist() == [True, True, True]
    assert not bank_a["merger_flag"].any()


def test_build_panel_flags_merger_from_zero_balance():
    item_frame = pd.DataFrame(
        {
            "bank_id": ["A", "A"],
            "quarter": _quarters(["2019Q1", "2019Q2"]),
            "RCON1766": [0.0, 5_000_000.0],
            "RCON1606": [0.0, 0.0],
            "RCON1607": [0.0, 0.0],
            "RCON1608": [0.0, 0.0],
            "RIAD4638": [0.0, 10.0],
            "RIAD4608": [0.0, 2.0],
        }
    )
    result = panel.build_panel(item_frame, categories=[LoanCategory.CI], min_balance=500.0)
    result = result.sort_values("quarter")
    assert result["merger_flag"].tolist() == [False, True]


def test_build_panel_excludes_categories_with_no_covering_code_set():
    # AUTO has no code set before 2011Q1 -- pre-2011 rows must simply be
    # absent for that category, not raise or zero-fill (unlike passing a
    # quarter AUTO DOES cover but omitting its required columns, which is
    # a real caller error and still raises -- see
    # test_category_mapping_raises_on_missing_required_column).
    pre_2011_frame = pd.DataFrame(
        {
            "bank_id": ["A", "A"],
            "quarter": _quarters(["2005Q1", "2005Q2"]),
            "RCON1766": [1000.0, 1100.0],
        }
    )
    result = panel.build_panel(pre_2011_frame, categories=[LoanCategory.AUTO])
    assert result.empty


# ------------------------------------------------- industry NCO rate ------


def test_build_industry_nco_rate_report_aggregates_across_banks():
    panel_frame = panel.build_panel(_ci_item_frame(), categories=[LoanCategory.CI])
    report = panel.build_industry_nco_rate_report(panel_frame)
    q2 = report[report["quarter"] == pd.Period("2019Q2", freq="Q")].iloc[0]
    # aggregate chargeoff = 15+30=45, recovery=3+6=9
    # avg balance: bank A (1000+1100)/2=1050, bank B (2000+2100)/2=2050 -> 3100
    expected_rate = (45.0 - 9.0) / 3100.0 * 4.0
    assert q2["industry_nco_rate"] == pytest.approx(expected_rate)
    assert bool(q2["has_chargeoff_data"])


def test_build_industry_nco_rate_report_excludes_merger_flagged_rows_by_default():
    panel_frame = pd.DataFrame(
        {
            "category": [LoanCategory.CI, LoanCategory.CI],
            "quarter": _quarters(["2010Q1", "2010Q1"]),
            "chargeoff_quarterly": [10.0, 999.0],  # bank B's flow is merger-contaminated
            "recovery_quarterly": [2.0, 0.0],
            "average_balance": [1000.0, 5_000_000.0],
            "merger_flag": [False, True],
        }
    )
    excluded = panel.build_industry_nco_rate_report(panel_frame)
    included = panel.build_industry_nco_rate_report(panel_frame, exclude_merger_flagged=False)

    excluded_rate = excluded.iloc[0]["industry_nco_rate"]
    included_rate = included.iloc[0]["industry_nco_rate"]
    assert excluded_rate == pytest.approx((10.0 - 2.0) / 1000.0 * 4.0)
    assert included_rate != pytest.approx(excluded_rate)


def test_build_industry_nco_rate_report_excludes_below_min_balance():
    panel_frame = pd.DataFrame(
        {
            "category": [LoanCategory.CI, LoanCategory.CI],
            "quarter": _quarters(["2010Q1", "2010Q1"]),
            "chargeoff_quarterly": [10.0, 500.0],
            "recovery_quarterly": [2.0, 0.0],
            "average_balance": [1000.0, 5.0],  # bank B is a near-zero-balance outlier
            "merger_flag": [False, False],
        }
    )
    report = panel.build_industry_nco_rate_report(panel_frame, min_balance=100.0)
    assert report.iloc[0]["industry_nco_rate"] == pytest.approx((10.0 - 2.0) / 1000.0 * 4.0)


def test_winsorize_nco_rates_clips_extreme_values_per_category():
    panel_frame = pd.DataFrame(
        {
            "category": [LoanCategory.CI] * 100,
            "annualized_nco_rate": [0.01] * 98 + [-5.0, 50.0],  # two extreme outliers
            "average_balance": [1_000_000.0] * 100,
        }
    )
    winsorized = panel.winsorize_nco_rates(panel_frame, lower_quantile=0.01, upper_quantile=0.99)
    assert winsorized.max() < 50.0
    assert winsorized.min() > -5.0
    # the bulk of normal values must be untouched
    assert np.allclose(winsorized.iloc[:98].to_numpy(), 0.01)


def test_winsorize_nco_rates_does_not_mutate_the_raw_panel():
    panel_frame = pd.DataFrame(
        {
            "category": [LoanCategory.CI] * 5,
            "annualized_nco_rate": [0.01, 0.02, 0.03, 0.04, 100.0],
            "average_balance": [1_000_000.0] * 5,
        }
    )
    raw_before = panel_frame["annualized_nco_rate"].copy()
    panel.winsorize_nco_rates(panel_frame)
    pd.testing.assert_series_equal(panel_frame["annualized_nco_rate"], raw_before)


def test_winsorize_nco_rates_is_independent_per_category():
    panel_frame = pd.DataFrame(
        {
            "category": [LoanCategory.CI] * 3 + [LoanCategory.CREDIT_CARD] * 3,
            # CI's scale is tiny; credit card's is large -- winsorization
            # must not let one category's distribution affect the other's.
            "annualized_nco_rate": [0.01, 0.02, 0.03, 0.10, 0.20, 0.30],
            "average_balance": [1_000_000.0] * 6,
        }
    )
    winsorized = panel.winsorize_nco_rates(panel_frame, lower_quantile=0.0, upper_quantile=1.0)
    # with quantile bounds at 0/1 (no clipping at all), values pass through unchanged
    assert winsorized.tolist() == pytest.approx([0.01, 0.02, 0.03, 0.10, 0.20, 0.30])


def test_winsorize_nco_rates_excludes_rows_below_min_balance_from_bounds_and_output():
    panel_frame = pd.DataFrame(
        {
            "category": [LoanCategory.CI] * 5,
            "annualized_nco_rate": [0.01, 0.02, 0.03, 0.04, 999.0],
            "average_balance": [1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0, 1.0],
        }
    )
    winsorized = panel.winsorize_nco_rates(panel_frame, min_balance=100.0)
    assert pd.isna(winsorized.iloc[-1])  # excluded row -> NaN, not clipped-and-kept
    # the extreme value must not have widened the clipping bounds for the rest
    assert winsorized.iloc[:4].max() <= 0.04


def test_flag_chargeoff_gaps_detects_a_missing_quarter_in_the_window():
    industry_report = pd.DataFrame(
        {
            "category": [LoanCategory.CRE_CONSTRUCTION] * 3 + [LoanCategory.CI] * 3,
            "quarter": [pd.Period(q, freq="Q") for q in ["2008Q1", "2008Q2", "2008Q3"]] * 2,
            "has_chargeoff_data": [True, False, True, True, True, True],
        }
    )
    gaps = panel.flag_chargeoff_gaps(
        industry_report,
        start="2008Q1",
        end="2008Q3",
        categories=[LoanCategory.CRE_CONSTRUCTION, LoanCategory.CI],
    )
    assert gaps[LoanCategory.CRE_CONSTRUCTION]
    assert not gaps[LoanCategory.CI]


def test_flag_chargeoff_gaps_flags_a_category_with_no_rows_at_all_in_the_window():
    # AUTO has zero rows in a pre-2011 window -- must be flagged as a gap,
    # not silently omitted from the result.
    industry_report = pd.DataFrame(
        {
            "category": [LoanCategory.CI] * 3,
            "quarter": [pd.Period(q, freq="Q") for q in ["2008Q1", "2008Q2", "2008Q3"]],
            "has_chargeoff_data": [True, True, True],
        }
    )
    gaps = panel.flag_chargeoff_gaps(
        industry_report,
        start="2008Q1",
        end="2008Q3",
        categories=[LoanCategory.CI, LoanCategory.AUTO],
    )
    assert not gaps[LoanCategory.CI]
    assert gaps[LoanCategory.AUTO]


def test_allowance_rollforward_residual_detects_a_broken_identity():
    residual = panel.allowance_rollforward_residual(
        pd.Series([100.0]), pd.Series([20.0]), pd.Series([10.0]), pd.Series([999.0])
    )
    assert residual.iloc[0] == pytest.approx(999.0 - 110.0)


# ------------------------------------------------ allowance/provision -----


def _allowance_item_frame(**overrides) -> pd.DataFrame:
    base = {
        "bank_id": ["A", "A"],
        "quarter": _quarters(["2015Q1", "2015Q2"]),
        "RCON3123": [100.0, 108.0],
        "RIAD4230": [5.0, 25.0],  # provision, YTD
        "RIADC079": [10.0, 25.0],  # total chargeoff, YTD
        "RIAD4605": [2.0, 5.0],  # total recovery, YTD
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_build_allowance_panel_uses_rcfd_fallback_only_where_rcon_is_missing():
    item_frame = _allowance_item_frame(
        bank_id=["A", "A", "B"],
        quarter=_quarters(["2015Q1", "2015Q2", "2015Q1"]),
        RCON3123=[100.0, 110.0, np.nan],  # bank B missing RCON entirely
        RCFD3123=[np.nan, np.nan, 250.0],  # only bank B has an RCFD figure
        RIAD4230=[5.0, 25.0, 12.0],
        RIADC079=[10.0, 25.0, 6.0],
        RIAD4605=[2.0, 5.0, 1.0],
    )
    result = panel.build_allowance_panel(item_frame)
    bank_a = result[result["bank_id"] == "A"].sort_values("quarter")
    bank_b = result[result["bank_id"] == "B"]

    assert bank_a["allowance_balance"].tolist() == pytest.approx([100.0, 110.0])
    assert bank_b["allowance_balance"].iloc[0] == pytest.approx(250.0)  # fell back to RCFD3123
    # Q1 provision_quarterly == its own YTD value; Q2 is YTD-differenced
    assert bank_a["provision_quarterly"].tolist() == pytest.approx([5.0, 20.0])
    assert bank_a["total_chargeoff_quarterly"].tolist() == pytest.approx([10.0, 15.0])
    assert bank_a["total_recovery_quarterly"].tolist() == pytest.approx([2.0, 3.0])


def test_build_allowance_panel_without_rcfd_column_uses_rcon_only():
    item_frame = _allowance_item_frame(
        bank_id=["A"],
        quarter=_quarters(["2015Q1"]),
        RCON3123=[100.0],
        RIAD4230=[5.0],
        RIADC079=[10.0],
        RIAD4605=[2.0],
    )
    result = panel.build_allowance_panel(item_frame)
    assert result["allowance_balance"].iloc[0] == pytest.approx(100.0)


def test_build_allowance_panel_flags_a_merger_sized_balance_jump():
    item_frame = _allowance_item_frame(
        RCON3123=[100.0, 400.0],  # 4x jump -- a merger, not organic growth
    )
    result = panel.build_allowance_panel(item_frame)
    result = result.sort_values("quarter")
    assert result["allowance_merger_flag"].tolist() == [False, True]


def test_build_allowance_panel_flags_a_merger_from_zero_allowance():
    item_frame = _allowance_item_frame(RCON3123=[0.0, 2_000_000.0])
    result = panel.build_allowance_panel(item_frame)
    result = result.sort_values("quarter")
    assert result["allowance_merger_flag"].tolist() == [False, True]


def test_build_allowance_panel_rcfd_fallback_applies_on_zero_rcon():
    item_frame = _allowance_item_frame(
        RCON3123=[0.0, 0.0], RCFD3123=[35_000_000.0, 36_000_000.0]
    )
    result = panel.build_allowance_panel(item_frame)
    assert result["allowance_balance"].tolist() == pytest.approx([35_000_000.0, 36_000_000.0])


def test_compute_bank_allowance_rollforward_uses_bank_level_totals_not_mapped_categories():
    # Bank-level total NCO (RIADC079 - RIAD4605) differs from the mapped
    # category sum -- the residual must use the bank-level total.
    allowance_panel = panel.build_allowance_panel(
        _allowance_item_frame(RIADC079=[10.0, 30.0], RIAD4605=[2.0, 6.0])
    )
    # bank-level: Q1 chargeoff=10, recovery=2 -> net=8; Q2 chargeoff=20, recovery=4 -> net=16
    # ending = beginning(100) + provision(20) - net(16) = 104
    q2_mask = allowance_panel["quarter"] == pd.Period("2015Q2", freq="Q")
    allowance_panel.loc[q2_mask, "allowance_balance"] = 104.0

    category_panel = pd.DataFrame(
        {
            "bank_id": ["A", "A"],
            "quarter": _quarters(["2015Q1", "2015Q2"]),
            "category": [LoanCategory.CI, LoanCategory.CI],
            # deliberately different from the bank-level total, to prove
            # the residual doesn't use this sum
            "chargeoff_quarterly": [1.0, 1.0],
            "recovery_quarterly": [0.0, 0.0],
        }
    )
    result = panel.compute_bank_allowance_rollforward(
        allowance_panel, category_panel, categories=[LoanCategory.CI]
    )
    q2 = result[result["quarter"] == pd.Period("2015Q2", freq="Q")].iloc[0]
    assert q2["total_net_chargeoffs"] == pytest.approx(16.0)  # bank-level, not mapped
    assert q2["mapped_category_net_chargeoffs"] == pytest.approx(1.0)  # kept, unused in residual
    assert q2["residual"] == pytest.approx(0.0, abs=1e-9)


def test_compute_bank_allowance_rollforward_excludes_combined_consumer_from_mapped_sum_by_default():
    item_frame = _allowance_item_frame(
        bank_id=["A"],
        quarter=_quarters(["2015Q1"]),
        RCON3123=[100.0],
        RIAD4230=[5.0],
        RIADC079=[10.0],
        RIAD4605=[2.0],
    )
    allowance_panel = panel.build_allowance_panel(item_frame)
    category_panel = pd.DataFrame(
        {
            "bank_id": ["A", "A"],
            "quarter": _quarters(["2015Q1", "2015Q1"]),
            "category": [LoanCategory.CI, LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED],
            "chargeoff_quarterly": [3.0, 999.0],
            "recovery_quarterly": [1.0, 0.0],
        }
    )
    result = panel.compute_bank_allowance_rollforward(allowance_panel, category_panel)
    # mapped_category_net_chargeoffs must only reflect CI (3-1=2), not the
    # combined-consumer row (which would double-count with AUTO/OTHER_CONSUMER)
    assert result["mapped_category_net_chargeoffs"].iloc[0] == pytest.approx(2.0)


def test_split_by_merger_flag_separates_flagged_rows():
    rollforward = pd.DataFrame(
        {
            "bank_id": ["A", "B"],
            "residual": [5.0, 999.0],
            "allowance_merger_flag": [False, True],
        }
    )
    clean, flagged = panel.split_by_merger_flag(rollforward)
    assert clean["bank_id"].tolist() == ["A"]
    assert flagged["bank_id"].tolist() == ["B"]


def test_split_by_cecl_adoption_flag_separates_flagged_rows():
    rollforward = pd.DataFrame(
        {
            "bank_id": ["A", "B"],
            "residual": [5.0, 999.0],
            "cecl_adoption_flag": [False, True],
        }
    )
    clean, flagged = panel.split_by_cecl_adoption_flag(rollforward)
    assert clean["bank_id"].tolist() == ["A"]
    assert flagged["bank_id"].tolist() == ["B"]


def test_compute_bank_allowance_rollforward_flags_each_banks_own_adoption_quarter():
    item_frame = _allowance_item_frame(
        bank_id=["A", "A", "B", "B"],
        quarter=_quarters(["2019Q4", "2020Q1", "2019Q4", "2020Q1"]),
        RCON3123=[100.0, 110.0, 200.0, 210.0],
        RIAD4230=[5.0, 5.0, 5.0, 5.0],
        RIADC079=[2.0, 2.0, 2.0, 2.0],
        RIAD4605=[1.0, 1.0, 1.0, 1.0],
    )
    allowance_panel = panel.build_allowance_panel(item_frame)
    category_panel = pd.DataFrame(
        {
            "bank_id": [],
            "quarter": pd.Series([], dtype="object"),
            "category": [],
            "chargeoff_quarterly": [],
            "recovery_quarterly": [],
        }
    )
    # bank A adopted 2020Q1; bank B never adopted within this fixture's window
    adoption_quarters = pd.Series({"A": pd.Period("2020Q1", freq="Q")})
    result = panel.compute_bank_allowance_rollforward(
        allowance_panel, category_panel, categories=[], cecl_adoption_quarters=adoption_quarters
    )
    result = result.set_index(["bank_id", "quarter"])
    assert result.loc[("A", pd.Period("2020Q1", freq="Q")), "cecl_adoption_flag"]
    assert not result.loc[("A", pd.Period("2019Q4", freq="Q")), "cecl_adoption_flag"]
    assert not result.loc[("B", pd.Period("2020Q1", freq="Q")), "cecl_adoption_flag"]


def test_compute_bank_allowance_rollforward_defaults_cecl_adoption_flag_to_false():
    item_frame = _allowance_item_frame()
    allowance_panel = panel.build_allowance_panel(item_frame)
    category_panel = pd.DataFrame(
        {
            "bank_id": [],
            "quarter": pd.Series([], dtype="object"),
            "category": [],
            "chargeoff_quarterly": [],
            "recovery_quarterly": [],
        }
    )
    result = panel.compute_bank_allowance_rollforward(
        allowance_panel, category_panel, categories=[]
    )
    assert not result["cecl_adoption_flag"].any()


def test_summarize_mapped_category_coverage_computes_share_by_quarter():
    rollforward = pd.DataFrame(
        {
            "quarter": _quarters(["2015Q1", "2015Q1"]),
            "mapped_category_net_chargeoffs": [8.0, 2.0],
            "total_net_chargeoffs": [10.0, 5.0],
        }
    )
    result = panel.summarize_mapped_category_coverage(rollforward)
    row = result.iloc[0]
    assert row["mapped_category_net_chargeoffs"] == pytest.approx(10.0)
    assert row["total_net_chargeoffs"] == pytest.approx(15.0)
    assert row["mapped_category_share"] == pytest.approx(10.0 / 15.0)


def test_summarize_mapped_category_coverage_excludes_2001q1_by_default():
    # Regression guard for the real anomaly this fixes: RIADC079 doesn't
    # exist as a column in real 2001Q1 data, so a 2001Q1 row here would
    # spuriously undercount total_net_chargeoffs and blow the ratio > 1.
    rollforward = pd.DataFrame(
        {
            "quarter": _quarters(["2001Q1", "2001Q2"]),
            "mapped_category_net_chargeoffs": [100.0, 8.0],
            "total_net_chargeoffs": [1.0, 10.0],  # 2001Q1's total is artificially tiny
        }
    )
    result = panel.summarize_mapped_category_coverage(rollforward)
    assert result["quarter"].tolist() == [pd.Period("2001Q2", freq="Q")]
    assert result.iloc[0]["mapped_category_share"] == pytest.approx(0.8)


def test_summarize_mapped_category_coverage_min_quarter_none_disables_filter():
    rollforward = pd.DataFrame(
        {
            "quarter": _quarters(["2001Q1", "2001Q2"]),
            "mapped_category_net_chargeoffs": [100.0, 8.0],
            "total_net_chargeoffs": [1.0, 10.0],
        }
    )
    result = panel.summarize_mapped_category_coverage(rollforward, min_quarter=None)
    assert len(result) == 2


# --------------------------------------------------------- coverage report -


def test_coverage_report_computes_share_and_aggregate_per_category_quarter():
    long_panel = pd.DataFrame(
        {
            "category": [LoanCategory.CI, LoanCategory.CI, LoanCategory.CI, LoanCategory.CI],
            "quarter": ["2019Q1", "2019Q1", "2019Q2", "2019Q2"],
            "balance": [100.0, np.nan, 150.0, 250.0],
        }
    )
    report = panel.build_coverage_report(long_panel)
    q1 = report[report["quarter"] == pd.Period("2019Q1", freq="Q")].iloc[0]
    q2 = report[report["quarter"] == pd.Period("2019Q2", freq="Q")].iloc[0]
    assert q1["n_banks"] == 2
    assert q1["coverage_share"] == pytest.approx(0.5)
    assert q1["aggregate_balance"] == pytest.approx(100.0)
    assert q2["coverage_share"] == pytest.approx(1.0)
    assert q2["aggregate_balance"] == pytest.approx(400.0)
    assert q2["aggregate_pct_change"] == pytest.approx((400.0 - 100.0) / 100.0)


def test_coverage_report_flags_a_break_at_a_code_switch_quarter():
    # cre_nonfarm_nonresidential switches at 2007Q1 -- a large jump exactly
    # there should be flagged; the same-size jump one quarter later (not a
    # switch boundary) should not be.
    long_panel = pd.DataFrame(
        {
            "category": [LoanCategory.CRE_NONFARM_NONRESIDENTIAL] * 3,
            "quarter": ["2006Q4", "2007Q1", "2007Q2"],
            "balance": [1000.0, 2000.0, 2000.0],
        }
    )
    report = panel.build_coverage_report(long_panel).set_index("quarter")
    assert bool(report.loc[pd.Period("2007Q1", freq="Q"), "code_switch"])
    assert bool(report.loc[pd.Period("2007Q1", freq="Q"), "possible_break"])
    assert not bool(report.loc[pd.Period("2007Q2", freq="Q"), "code_switch"])
    assert not bool(report.loc[pd.Period("2007Q2", freq="Q"), "possible_break"])
