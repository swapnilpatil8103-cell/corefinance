"""Structural sanity checks on the loan-category MDRM mapping table --
not a test of the codes' real-world correctness (that needs a real
filed Call Report), just that the table is internally consistent."""

import pandas as pd

from corefin.credit.schema import CATEGORY_MDRM_CODES, LoanCategory, MdrmCodeSet


def test_every_loan_category_is_mapped():
    assert set(CATEGORY_MDRM_CODES) == set(LoanCategory)


def test_every_code_set_has_at_least_one_balance_item():
    for category, code_sets in CATEGORY_MDRM_CODES.items():
        for code_set in code_sets:
            label = f"{category}/{code_set.valid_from}"
            assert code_set.balance_items, f"{label} has no balance item(s)"


def test_code_sets_missing_chargeoff_or_recovery_items_have_an_explanatory_note():
    # Most code sets have charge-off/recovery items; a few (CRE construction
    # and nonfarm-nonresidential from 2007Q1 onward) genuinely don't -- the
    # real bulk Call Report data has no breakout at that granularity. Any
    # such gap must be documented, not silent.
    for category, code_sets in CATEGORY_MDRM_CODES.items():
        for code_set in code_sets:
            if not code_set.chargeoff_items or not code_set.recovery_items:
                assert code_set.note, (
                    f"{category}/{code_set.valid_from} has no charge-off/recovery "
                    "item(s) and no note explaining why"
                )


def test_needs_confirmation_code_sets_all_have_an_explanatory_note():
    for category, code_sets in CATEGORY_MDRM_CODES.items():
        for code_set in code_sets:
            if code_set.confidence == "needs_confirmation":
                label = f"{category}/{code_set.valid_from}"
                assert code_set.note, f"{label} is needs_confirmation but has no note"


def test_code_sets_within_a_category_are_contiguous_and_non_overlapping():
    for category, code_sets in CATEGORY_MDRM_CODES.items():
        ordered = sorted(code_sets, key=lambda cs: cs.valid_from)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            assert earlier.valid_to is not None, (
                f"{category}: {earlier.valid_from} has no valid_to but is followed by "
                f"{later.valid_from}"
            )
            earlier_end = pd.Period(earlier.valid_to, freq="Q")
            later_start = pd.Period(later.valid_from, freq="Q")
            assert later_start == earlier_end + 1, (
                f"{category}: gap or overlap between {earlier.valid_to} and {later.valid_from}"
            )
        assert ordered[-1].valid_to is None, (
            f"{category}'s last code set should still be active (valid_to=None)"
        )


def test_covers_is_inclusive_of_both_boundaries_and_open_ended_when_valid_to_is_none():
    bounded = MdrmCodeSet(valid_from="2007Q1", valid_to="2007Q4", balance_items=("X",))
    assert not bounded.covers(pd.Period("2006Q4", freq="Q"))
    assert bounded.covers(pd.Period("2007Q1", freq="Q"))
    assert bounded.covers(pd.Period("2007Q4", freq="Q"))
    assert not bounded.covers(pd.Period("2008Q1", freq="Q"))

    open_ended = MdrmCodeSet(valid_from="2011Q1", valid_to=None, balance_items=("X",))
    assert open_ended.covers(pd.Period("2011Q1", freq="Q"))
    assert open_ended.covers(pd.Period("2099Q4", freq="Q"))
    assert not open_ended.covers(pd.Period("2010Q4", freq="Q"))


def test_auto_category_has_no_code_set_before_2011():
    auto_code_sets = CATEGORY_MDRM_CODES[LoanCategory.AUTO]
    assert all(cs.valid_from == "2011Q1" for cs in auto_code_sets)
    assert not any(cs.covers(pd.Period("2010Q4", freq="Q")) for cs in auto_code_sets)


def test_other_consumer_category_has_no_code_set_before_2011():
    other_consumer_code_sets = CATEGORY_MDRM_CODES[LoanCategory.OTHER_CONSUMER]
    assert all(cs.valid_from == "2011Q1" for cs in other_consumer_code_sets)
    assert not any(cs.covers(pd.Period("2010Q4", freq="Q")) for cs in other_consumer_code_sets)


def test_combined_consumer_series_covers_2001_through_present_continuously():
    combined = LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED
    code_sets = CATEGORY_MDRM_CODES[combined]
    assert code_sets[0].valid_from == "2001Q1"
    assert code_sets[-1].valid_to is None
    # every quarter from 2001Q1 through a recent quarter must be covered by
    # exactly one code set (no gap for this category, unlike AUTO/OTHER_CONSUMER)
    quarters = pd.period_range("2001Q1", "2026Q2", freq="Q")
    for quarter in quarters:
        matches = [cs for cs in code_sets if cs.covers(quarter)]
        assert len(matches) == 1, f"{combined} at {quarter}: {len(matches)} code sets match"


def test_combined_consumer_series_post_2011_includes_other_revolving_credit():
    # RCON2011's own MDRM COMPARABILITY note defines it as SUM(RCONK137,
    # RCONK207) -- excluding RCONB539 ("other revolving credit plans") --
    # but RIADK205's own MDRM Description covers RCONB539's scope too, so a
    # combined balance built from RCON2011's derivation formula alone would
    # mismatch its own charge-off item's scope (verified: real 2008Q4/2010Q4
    # data showed zero-balance/nonzero-chargeoff rows this fix eliminates).
    # The combined series must match the charge-off item's real scope, which
    # makes it an EXACT union of AUTO + OTHER_CONSUMER, not RCON2011's own
    # (narrower) derivation formula.
    (auto_code_set,) = CATEGORY_MDRM_CODES[LoanCategory.AUTO]
    (other_consumer_code_set,) = CATEGORY_MDRM_CODES[LoanCategory.OTHER_CONSUMER]
    combined_code_sets = CATEGORY_MDRM_CODES[LoanCategory.AUTO_AND_OTHER_CONSUMER_COMBINED]
    post_2011 = next(cs for cs in combined_code_sets if cs.valid_from == "2011Q1")

    assert set(post_2011.balance_items) == {"RCONK137", "RCONK207", "RCONB539"}
    assert set(post_2011.balance_items) == set(auto_code_set.balance_items) | set(
        other_consumer_code_set.balance_items
    )
