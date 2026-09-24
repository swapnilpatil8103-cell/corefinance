"""Credit metrics derived from the debt schedule and balance sheet.

All leverage figures are net of cash. FCCR here is defined as
(EBITDA - capex) / (cash interest + scheduled mandatory amortization); this
is one common convention among several used in practice, and callers who
need a different fixed-charge definition should compute it directly from
the underlying arrays rather than relying on this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corefin.assumptions.schema import TrancheConfig, TrancheType
from corefin.debt.circularity import DebtScheduleResult

SENIOR_TRANCHE_TYPES = frozenset(
    {
        TrancheType.REVOLVER,
        TrancheType.TERM_LOAN_A,
        TrancheType.TERM_LOAN_B,
        TrancheType.SENIOR_NOTES,
    }
)


def is_senior(tranche: TrancheConfig) -> bool:
    return tranche.tranche_type in SENIOR_TRANCHE_TYPES


@dataclass(frozen=True)
class CreditMetrics:
    total_net_leverage: np.ndarray
    senior_net_leverage: np.ndarray
    interest_coverage: np.ndarray
    fccr: np.ndarray
    cumulative_debt_paydown: np.ndarray


def senior_debt_balance(
    debt_schedule: DebtScheduleResult, tranches: list[TrancheConfig]
) -> np.ndarray:
    senior_names = [t.name for t in tranches if is_senior(t)]
    return sum(debt_schedule.ending_balance[name] for name in senior_names)


def compute_credit_metrics(
    debt_schedule: DebtScheduleResult,
    tranches: list[TrancheConfig],
    ebitda: np.ndarray,
    capex: np.ndarray,
    cash: np.ndarray,
    total_debt: np.ndarray,
) -> CreditMetrics:
    senior_debt = senior_debt_balance(debt_schedule, tranches)
    cash_interest_total = sum(debt_schedule.cash_interest.values())
    mandatory_amort_total = sum(debt_schedule.mandatory_amort.values())
    initial_total_debt = sum((0.0 if t.is_revolver else t.size_mm) for t in tranches)

    with np.errstate(divide="ignore", invalid="ignore"):
        return CreditMetrics(
            total_net_leverage=(total_debt - cash) / ebitda,
            senior_net_leverage=(senior_debt - cash) / ebitda,
            interest_coverage=ebitda / cash_interest_total,
            fccr=(ebitda - capex) / (cash_interest_total + mandatory_amort_total),
            cumulative_debt_paydown=initial_total_debt - total_debt,
        )
