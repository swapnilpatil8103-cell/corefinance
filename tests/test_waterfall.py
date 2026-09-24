import numpy as np

from corefin.assumptions.schema import RateType, TrancheConfig, TrancheType
from corefin.debt.waterfall import run_period_waterfall


def _tranches():
    revolver = TrancheConfig(
        name="Revolver",
        tranche_type=TrancheType.REVOLVER,
        size_mm=50.0,
        rate_type=RateType.FLOATING,
        spread=0.04,
        commitment_fee_pct=0.005,
    )
    tlb = TrancheConfig(
        name="TLB",
        tranche_type=TrancheType.TERM_LOAN_B,
        size_mm=300.0,
        rate_type=RateType.FLOATING,
        spread=0.05,
        mandatory_amort_pct_of_original=0.01,
        cash_sweep_eligible=True,
        sweep_priority=1,
    )
    notes = TrancheConfig(
        name="Notes",
        tranche_type=TrancheType.SENIOR_NOTES,
        size_mm=100.0,
        rate_type=RateType.FIXED,
        fixed_rate=0.08,
        cash_sweep_eligible=True,
        sweep_priority=2,
    )
    return [revolver, tlb, notes]


def _zero_pik(tranches, n_scenarios):
    return {t.name: np.zeros(n_scenarios) for t in tranches if not t.is_revolver}


def test_mandatory_amortization_capped_at_outstanding_balance():
    tranches = _tranches()
    beginning = {"Revolver": np.array([0.0]), "TLB": np.array([2.0]), "Notes": np.array([100.0])}
    scheduled = {"TLB": np.array([3.0]), "Notes": np.array([0.0])}
    outcome = run_period_waterfall(
        tranches,
        beginning,
        scheduled,
        np.array([50.0]),
        minimum_cash_mm=10.0,
        sweep_pct=1.0,
        pik_accrual_t=_zero_pik(tranches, 1),
    )
    assert outcome.mandatory_amort["TLB"][0] == 2.0
    assert outcome.ending_balances["TLB"][0] == 0.0


def test_revolver_draws_on_shortfall():
    tranches = _tranches()
    beginning = {"Revolver": np.array([0.0]), "TLB": np.array([300.0]), "Notes": np.array([100.0])}
    scheduled = {"TLB": np.array([3.0]), "Notes": np.array([0.0])}
    outcome = run_period_waterfall(
        tranches,
        beginning,
        scheduled,
        np.array([5.0]),
        minimum_cash_mm=10.0,
        sweep_pct=1.0,
        pik_accrual_t=_zero_pik(tranches, 1),
    )
    assert outcome.revolver_draw[0] == 8.0
    assert outcome.ending_balances["Revolver"][0] == 8.0
    assert not outcome.shortfall_flag[0]
    assert outcome.ending_cash[0] == 10.0


def test_revolver_shortfall_flagged_when_commitment_exhausted():
    tranches = _tranches()
    beginning = {"Revolver": np.array([48.0]), "TLB": np.array([300.0]), "Notes": np.array([100.0])}
    scheduled = {"TLB": np.array([3.0]), "Notes": np.array([0.0])}
    outcome = run_period_waterfall(
        tranches,
        beginning,
        scheduled,
        np.array([5.0]),
        minimum_cash_mm=10.0,
        sweep_pct=1.0,
        pik_accrual_t=_zero_pik(tranches, 1),
    )
    assert outcome.revolver_draw[0] == 2.0
    assert outcome.ending_balances["Revolver"][0] == 50.0
    assert outcome.shortfall_flag[0]


def test_revolver_paydown_then_sweep_by_priority():
    tranches = _tranches()
    beginning = {"Revolver": np.array([20.0]), "TLB": np.array([300.0]), "Notes": np.array([100.0])}
    scheduled = {"TLB": np.array([3.0]), "Notes": np.array([0.0])}
    # cash_available_pre_financing = 60 -> after mandatory (3) = 57 -> paydown revolver (20) -> 37
    # -> above minimum cash (10) by 27, swept 100% into TLB (priority 1) first.
    outcome = run_period_waterfall(
        tranches,
        beginning,
        scheduled,
        np.array([60.0]),
        minimum_cash_mm=10.0,
        sweep_pct=1.0,
        pik_accrual_t=_zero_pik(tranches, 1),
    )
    assert outcome.revolver_paydown[0] == 20.0
    assert outcome.ending_balances["Revolver"][0] == 0.0
    assert outcome.sweep_amort["TLB"][0] == 27.0
    assert outcome.sweep_amort["Notes"][0] == 0.0
    assert outcome.ending_cash[0] == 10.0


def test_sweep_spills_to_next_priority_once_higher_is_repaid():
    tranches = _tranches()
    beginning = {"Revolver": np.array([0.0]), "TLB": np.array([5.0]), "Notes": np.array([100.0])}
    scheduled = {"TLB": np.array([0.0]), "Notes": np.array([0.0])}
    # cash_available=60, no mandatory/revolver activity, excess over min cash (10)=50, swept 100%.
    # TLB (priority 1) fully repaid with 5, remaining 45 spills to Notes (priority 2).
    outcome = run_period_waterfall(
        tranches,
        beginning,
        scheduled,
        np.array([60.0]),
        minimum_cash_mm=10.0,
        sweep_pct=1.0,
        pik_accrual_t=_zero_pik(tranches, 1),
    )
    assert outcome.sweep_amort["TLB"][0] == 5.0
    assert outcome.ending_balances["TLB"][0] == 0.0
    assert outcome.sweep_amort["Notes"][0] == 45.0
    assert outcome.ending_balances["Notes"][0] == 55.0


def test_sweep_pct_below_100_leaves_cash_above_minimum():
    tranches = _tranches()
    beginning = {"Revolver": np.array([0.0]), "TLB": np.array([300.0]), "Notes": np.array([100.0])}
    scheduled = {"TLB": np.array([0.0]), "Notes": np.array([0.0])}
    outcome = run_period_waterfall(
        tranches,
        beginning,
        scheduled,
        np.array([60.0]),
        minimum_cash_mm=10.0,
        sweep_pct=0.5,
        pik_accrual_t=_zero_pik(tranches, 1),
    )
    # excess = 50, sweep_pool = 25 swept into TLB, remaining 25 stays as cash on top of minimum.
    assert outcome.sweep_amort["TLB"][0] == 25.0
    assert outcome.ending_cash[0] == 35.0


def test_pik_accrual_increases_ending_balance():
    tranches = _tranches()
    beginning = {"Revolver": np.array([0.0]), "TLB": np.array([300.0]), "Notes": np.array([100.0])}
    scheduled = {"TLB": np.array([0.0]), "Notes": np.array([0.0])}
    pik = {"TLB": np.array([0.0]), "Notes": np.array([8.0])}
    outcome = run_period_waterfall(
        tranches,
        beginning,
        scheduled,
        np.array([15.0]),
        minimum_cash_mm=10.0,
        sweep_pct=1.0,
        pik_accrual_t=pik,
    )
    assert outcome.ending_balances["Notes"][0] == 108.0


def test_vectorized_across_scenarios():
    tranches = _tranches()
    n = 5
    beginning = {
        "Revolver": np.zeros(n),
        "TLB": np.full(n, 300.0),
        "Notes": np.full(n, 100.0),
    }
    scheduled = {"TLB": np.full(n, 3.0), "Notes": np.zeros(n)}
    cash_available = np.array([5.0, 60.0, 10.0, 0.0, 100.0])
    outcome = run_period_waterfall(
        tranches,
        beginning,
        scheduled,
        cash_available,
        minimum_cash_mm=10.0,
        sweep_pct=1.0,
        pik_accrual_t=_zero_pik(tranches, n),
    )
    assert outcome.ending_cash.shape == (n,)
    assert np.all(outcome.ending_balances["Revolver"] >= 0.0)
    assert np.all(outcome.ending_balances["Revolver"] <= 50.0)
