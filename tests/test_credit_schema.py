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
