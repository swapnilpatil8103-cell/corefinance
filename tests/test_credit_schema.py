"""Structural sanity checks on the loan-category MDRM mapping table --
not a test of the codes' real-world correctness (that needs a real
filed Call Report), just that the table is internally consistent."""

from corefin.credit.schema import CATEGORY_MDRM_CODES, LoanCategory


def test_every_loan_category_is_mapped():
    assert set(CATEGORY_MDRM_CODES) == set(LoanCategory)


def test_every_category_has_at_least_one_balance_item():
    for category, codes in CATEGORY_MDRM_CODES.items():
        assert codes.balance_items, f"{category} has no balance item(s)"


def test_every_category_has_chargeoff_and_recovery_items():
    for category, codes in CATEGORY_MDRM_CODES.items():
        assert codes.chargeoff_items, f"{category} has no charge-off item(s)"
        assert codes.recovery_items, f"{category} has no recovery item(s)"


def test_needs_confirmation_categories_all_have_an_explanatory_note():
    for category, codes in CATEGORY_MDRM_CODES.items():
        if codes.confidence == "needs_confirmation":
            assert codes.note, f"{category} is flagged needs_confirmation but has no note"


def test_no_item_code_reused_across_different_categories_balance_items():
    seen: dict[str, LoanCategory] = {}
    for category, codes in CATEGORY_MDRM_CODES.items():
        for item in codes.balance_items:
            assert item not in seen, f"{item} used by both {seen.get(item)} and {category}"
            seen[item] = category
