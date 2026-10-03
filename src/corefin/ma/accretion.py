"""Stage 4 deal outputs: EPS accretion/dilution, TBV per share dilution
and crossover earnback, and acquirer IRR -- all built on top of
`projection.ProFormaProjection` and each bank's own standalone Stage 2
`BankModelResult`.

SHARE COUNTS are config inputs (`DealConfig.consideration.
acquirer_share_price`/`acquirer_shares_outstanding_mm`), never scraped
market data, per the approved plan. New shares issued = the stock
consideration's DOLLAR value divided by the acquirer's share price;
pro forma shares outstanding = the acquirer's own standalone count plus
those new shares (target's own shares are retired/exchanged, not
added -- standard stock-for-stock merger accounting)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from corefin.bank.model import BankModelResult
from corefin.ma.projection import ProFormaProjection
from corefin.ma.purchase_accounting import SourcesAndUses
from corefin.ma.schema import DealConfig
from corefin.timeline import Timeline

_PERIOD_LENGTH_YEARS = 0.25


def compute_new_shares_issued_mm(sources_and_uses: SourcesAndUses, config: DealConfig) -> float:
    return sources_and_uses.stock_consideration_mm / config.consideration.acquirer_share_price


def compute_pro_forma_shares_outstanding_mm(
    sources_and_uses: SourcesAndUses, config: DealConfig
) -> float:
    return config.consideration.acquirer_shares_outstanding_mm + compute_new_shares_issued_mm(
        sources_and_uses, config
    )


@dataclass(frozen=True)
class EpsAccretionResult:
    timeline: Timeline
    new_shares_issued_mm: float
    pro_forma_shares_outstanding_mm: float
    exchange_ratio: float | None  # new shares issued per target share, if target share count given

    standalone_eps: np.ndarray  # $ per share, periods 0..n-1 (period 0 = pre-close actual)
    pro_forma_eps: np.ndarray  # $ per share, period 0 = 0.0 (no income flow at the close instant)
    accretion_dilution_pct: np.ndarray  # pro_forma/standalone - 1; NaN at period 0 (not meaningful)

    def annual(self) -> dict[str, tuple[float, float, float]]:
        """{"Year 1": (standalone_eps, pro_forma_eps, accretion_dilution_pct), "Year 2": ...}
        -- sums quarterly EPS within each 4-quarter year (periods 1-4,
        5-8, ...; period 0 excluded, see module docstring) since EPS is
        additive across quarters at a CONSTANT share count."""
        n_years = (len(self.standalone_eps) - 1) // 4
        result = {}
        for year in range(n_years):
            start = 1 + year * 4
            end = start + 4
            standalone_year = float(self.standalone_eps[start:end].sum())
            pro_forma_year = float(self.pro_forma_eps[start:end].sum())
            accretion = pro_forma_year / standalone_year - 1.0 if standalone_year else float("nan")
            result[f"Year {year + 1}"] = (standalone_year, pro_forma_year, accretion)
        return result


def compute_eps_accretion_dilution(
    acquirer_result: BankModelResult,
    pro_forma: ProFormaProjection,
    sources_and_uses: SourcesAndUses,
    config: DealConfig,
) -> EpsAccretionResult:
    new_shares_issued_mm = compute_new_shares_issued_mm(sources_and_uses, config)
    pro_forma_shares_mm = compute_pro_forma_shares_outstanding_mm(sources_and_uses, config)
    exchange_ratio = (
        new_shares_issued_mm / config.consideration.target_shares_outstanding_mm
        if config.consideration.target_shares_outstanding_mm
        else None
    )

    standalone_eps = (
        acquirer_result.income_statement.net_income_available_to_common_mm
        / config.consideration.acquirer_shares_outstanding_mm
    )
    pro_forma_eps = pro_forma.net_income_available_to_common_mm / pro_forma_shares_mm

    with np.errstate(divide="ignore", invalid="ignore"):
        accretion_dilution_pct = pro_forma_eps / standalone_eps - 1.0
    accretion_dilution_pct[0] = float("nan")  # period 0 isn't a meaningful pro forma comparison

    return EpsAccretionResult(
        timeline=pro_forma.timeline,
        new_shares_issued_mm=new_shares_issued_mm,
        pro_forma_shares_outstanding_mm=pro_forma_shares_mm,
        exchange_ratio=exchange_ratio,
        standalone_eps=standalone_eps,
        pro_forma_eps=pro_forma_eps,
        accretion_dilution_pct=accretion_dilution_pct,
    )


@dataclass(frozen=True)
class TbvEarnbackResult:
    timeline: Timeline
    standalone_tbv_per_share: np.ndarray
    pro_forma_tbv_per_share: np.ndarray
    tbv_dilution_at_close_pct: float
    earnback_period_index: int | None  # None if TBV/share never crosses over within the horizon
    earnback_years: float | None


def compute_tbv_dilution_and_earnback(
    acquirer_result: BankModelResult,
    pro_forma: ProFormaProjection,
    sources_and_uses: SourcesAndUses,
    config: DealConfig,
) -> TbvEarnbackResult:
    pro_forma_shares_mm = compute_pro_forma_shares_outstanding_mm(sources_and_uses, config)

    standalone_tbv_mm = (
        acquirer_result.balance_sheet.equity_mm
        - acquirer_result.opening.preferred_stock_mm
        - acquirer_result.balance_sheet.goodwill_mm
        - acquirer_result.balance_sheet.other_intangibles_mm
    )
    standalone_tbv_per_share = (
        standalone_tbv_mm / config.consideration.acquirer_shares_outstanding_mm
    )
    pro_forma_tbv_per_share = pro_forma.tangible_common_equity_mm / pro_forma_shares_mm

    tbv_dilution_at_close_pct = pro_forma_tbv_per_share[0] / standalone_tbv_per_share[0] - 1.0

    earnback_period_index = None
    for t in range(1, len(pro_forma_tbv_per_share)):
        if pro_forma_tbv_per_share[t] >= standalone_tbv_per_share[t]:
            earnback_period_index = t
            break
    earnback_years = (
        earnback_period_index * _PERIOD_LENGTH_YEARS if earnback_period_index is not None else None
    )

    return TbvEarnbackResult(
        timeline=pro_forma.timeline,
        standalone_tbv_per_share=standalone_tbv_per_share,
        pro_forma_tbv_per_share=pro_forma_tbv_per_share,
        tbv_dilution_at_close_pct=float(tbv_dilution_at_close_pct),
        earnback_period_index=earnback_period_index,
        earnback_years=earnback_years,
    )


def compute_acquirer_irr(
    acquirer_result: BankModelResult,
    pro_forma: ProFormaProjection,
    sources_and_uses: SourcesAndUses,
    config: DealConfig,
) -> float:
    """Illustrative deal IRR over the available projection horizon
    (`len(pro_forma...) - 1` quarters -- short relative to a typical 3-5
    year bank M&A IRR horizon; flagged as illustrative, not a substitute
    for a longer-horizon analysis). CF[0] = -total consideration (cash +
    stock, both treated as a real cost -- stock dilutes existing
    shareholders just as cash would be spent); CF[1..n-1] = the deal's
    INCREMENTAL net income available to common each quarter (pro forma
    minus what the acquirer would have earned standalone); a terminal
    value (the deal's incremental tangible common equity at the end of
    the horizon) is added to the LAST cash flow."""
    standalone_nicc_mm = acquirer_result.income_statement.net_income_available_to_common_mm
    incremental_income_mm = pro_forma.net_income_available_to_common_mm[1:] - standalone_nicc_mm[1:]
    cash_flows = np.concatenate([[-sources_and_uses.consideration_mm], incremental_income_mm])

    standalone_tbv_mm = (
        acquirer_result.balance_sheet.equity_mm
        - acquirer_result.opening.preferred_stock_mm
        - acquirer_result.balance_sheet.goodwill_mm
        - acquirer_result.balance_sheet.other_intangibles_mm
    )
    terminal_incremental_tbv_mm = pro_forma.tangible_common_equity_mm[-1] - standalone_tbv_mm[-1]
    cash_flows[-1] += terminal_incremental_tbv_mm

    def npv(quarterly_rate: float) -> float:
        periods = np.arange(len(cash_flows))
        return float(np.sum(cash_flows / (1.0 + quarterly_rate) ** periods))

    quarterly_irr = brentq(npv, -0.99, 10.0)
    return (1.0 + quarterly_irr) ** 4 - 1.0
