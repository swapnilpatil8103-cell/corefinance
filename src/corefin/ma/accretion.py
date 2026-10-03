"""Stage 4 deal outputs: EPS accretion/dilution (in three views -- GAAP,
excluding one-time charges, and excluding one-time charges AND purchase-
accounting marks, see `projection.py`'s own module docstring), TBV per
share dilution and crossover earnback, and acquirer IRR -- all built on
top of `projection.ProFormaProjection` and each bank's own standalone
Stage 2 `BankModelResult`.

SHARE COUNTS are config inputs (`DealConfig.consideration.
acquirer_share_price`/`acquirer_shares_outstanding_mm`), never scraped
market data, per the approved plan. New shares issued = the stock
consideration's DOLLAR value divided by the acquirer's share price;
pro forma shares outstanding = the acquirer's own standalone count plus
those new shares (target's own shares are retired/exchanged, not
added -- standard stock-for-stock merger accounting).

IRR CASH FLOWS -- fixed to not double-count retained earnings. CF[0] =
-total consideration (cash + stock). CF[1..n-1] = the deal's INCREMENTAL
DISTRIBUTABLE cash each quarter (pro forma minus what the acquirer would
have generated standalone) -- NOT full net income available to common,
which would also be retained (and so would reappear in the terminal
tangible-equity value below, double-counting it). `IrrConfig.
distributable_cash_method` picks either actual common DIVIDENDS or cash
above a TARGET CET1 RATIO (see `_distributable_cash_above_target_mm`).
The terminal value (added to the LAST cash flow) is `IrrConfig.
exit_multiple` times either the deal's incremental forward (final-year)
net income available to common (a P/E-style exit) or its incremental
tangible common equity at the end of the horizon (a P/TBV-style exit) --
`IrrConfig.exit_multiple_basis`. This represents the value of whatever
WASN'T already paid out as distributable cash during the horizon, not a
re-capture of cash already counted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from corefin.bank.model import BankModelResult
from corefin.ma.projection import ProFormaProjection
from corefin.ma.purchase_accounting import SourcesAndUses
from corefin.ma.schema import DealConfig, DistributableCashMethod, ExitMultipleBasis
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
class AnnualEpsAccretion:
    standalone_eps: float
    pro_forma_eps_gaap: float
    accretion_dilution_pct_gaap: float
    pro_forma_eps_excl_one_time: float
    accretion_dilution_pct_excl_one_time: float
    pro_forma_eps_excl_one_time_and_marks: float
    accretion_dilution_pct_excl_one_time_and_marks: float


@dataclass(frozen=True)
class EpsAccretionResult:
    timeline: Timeline
    new_shares_issued_mm: float
    pro_forma_shares_outstanding_mm: float
    exchange_ratio: float | None  # new shares issued per target share, if target share count given

    standalone_eps: np.ndarray  # $ per share, periods 0..n-1 (period 0 = pre-close actual)

    # view (a): GAAP
    pro_forma_eps_gaap: np.ndarray  # period 0 = 0.0 (no income flow at the close instant)
    accretion_dilution_pct_gaap: np.ndarray  # pro_forma/standalone - 1; NaN at period 0

    # view (b): GAAP excluding one-time Day-2/restructuring charges
    pro_forma_eps_excl_one_time: np.ndarray
    accretion_dilution_pct_excl_one_time: np.ndarray

    # view (c): (b) ALSO excluding loan mark accretion and CDI amortization
    pro_forma_eps_excl_one_time_and_marks: np.ndarray
    accretion_dilution_pct_excl_one_time_and_marks: np.ndarray

    def __post_init__(self) -> None:
        n = self.timeline.n_periods
        for name in (
            "standalone_eps",
            "pro_forma_eps_gaap",
            "accretion_dilution_pct_gaap",
            "pro_forma_eps_excl_one_time",
            "accretion_dilution_pct_excl_one_time",
            "pro_forma_eps_excl_one_time_and_marks",
            "accretion_dilution_pct_excl_one_time_and_marks",
        ):
            array = getattr(self, name)
            if array.shape != (n,):
                raise ValueError(f"{name} has shape {array.shape}, expected ({n},)")

    def annual(self) -> dict[str, AnnualEpsAccretion]:
        """{"Year 1": AnnualEpsAccretion(...), "Year 2": ...} -- sums
        quarterly EPS within each 4-quarter year (periods 1-4, 5-8, ...;
        period 0 excluded, see module docstring) since EPS is additive
        across quarters at a CONSTANT share count."""
        def _pro_forma_and_accretion(
            pro_forma_eps: np.ndarray, start: int, end: int, standalone_year: float
        ) -> tuple[float, float]:
            pro_forma_year = float(pro_forma_eps[start:end].sum())
            accretion = pro_forma_year / standalone_year - 1.0 if standalone_year else float("nan")
            return pro_forma_year, accretion

        n_years = (len(self.standalone_eps) - 1) // 4
        result = {}
        for year in range(n_years):
            start = 1 + year * 4
            end = start + 4
            standalone_year = float(self.standalone_eps[start:end].sum())

            gaap_eps, gaap_accretion = _pro_forma_and_accretion(
                self.pro_forma_eps_gaap, start, end, standalone_year
            )
            excl_one_time_eps, excl_one_time_accretion = _pro_forma_and_accretion(
                self.pro_forma_eps_excl_one_time, start, end, standalone_year
            )
            excl_marks_eps, excl_marks_accretion = _pro_forma_and_accretion(
                self.pro_forma_eps_excl_one_time_and_marks, start, end, standalone_year
            )
            result[f"Year {year + 1}"] = AnnualEpsAccretion(
                standalone_eps=standalone_year,
                pro_forma_eps_gaap=gaap_eps,
                accretion_dilution_pct_gaap=gaap_accretion,
                pro_forma_eps_excl_one_time=excl_one_time_eps,
                accretion_dilution_pct_excl_one_time=excl_one_time_accretion,
                pro_forma_eps_excl_one_time_and_marks=excl_marks_eps,
                accretion_dilution_pct_excl_one_time_and_marks=excl_marks_accretion,
            )
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

    def _pro_forma_eps_and_accretion(
        net_income_available_to_common_mm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        eps = net_income_available_to_common_mm / pro_forma_shares_mm
        with np.errstate(divide="ignore", invalid="ignore"):
            accretion_dilution_pct = eps / standalone_eps - 1.0
        accretion_dilution_pct[0] = float("nan")  # period 0 isn't a meaningful pro forma comparison
        return eps, accretion_dilution_pct

    pro_forma_eps_gaap, accretion_dilution_pct_gaap = _pro_forma_eps_and_accretion(
        pro_forma.net_income_available_to_common_mm
    )
    pro_forma_eps_excl_one_time, accretion_dilution_pct_excl_one_time = (
        _pro_forma_eps_and_accretion(pro_forma.net_income_available_to_common_excl_one_time_mm)
    )
    pro_forma_eps_excl_one_time_and_marks, accretion_dilution_pct_excl_one_time_and_marks = (
        _pro_forma_eps_and_accretion(
            pro_forma.net_income_available_to_common_excl_one_time_and_marks_mm
        )
    )

    return EpsAccretionResult(
        timeline=pro_forma.timeline,
        new_shares_issued_mm=new_shares_issued_mm,
        pro_forma_shares_outstanding_mm=pro_forma_shares_mm,
        exchange_ratio=exchange_ratio,
        standalone_eps=standalone_eps,
        pro_forma_eps_gaap=pro_forma_eps_gaap,
        accretion_dilution_pct_gaap=accretion_dilution_pct_gaap,
        pro_forma_eps_excl_one_time=pro_forma_eps_excl_one_time,
        accretion_dilution_pct_excl_one_time=accretion_dilution_pct_excl_one_time,
        pro_forma_eps_excl_one_time_and_marks=pro_forma_eps_excl_one_time_and_marks,
        accretion_dilution_pct_excl_one_time_and_marks=accretion_dilution_pct_excl_one_time_and_marks,
    )


@dataclass(frozen=True)
class TbvEarnbackResult:
    timeline: Timeline
    standalone_tbv_per_share: np.ndarray
    pro_forma_tbv_per_share: np.ndarray
    tbv_dilution_at_close_pct: float
    earnback_period_index: int | None  # None if TBV/share never crosses over within the horizon
    earnback_years: float | None

    @property
    def earnback_label(self) -> str:
        """Human-readable earnback for reporting -- "beyond horizon"
        rather than a bare `None`, which reads as "no dilution"/missing
        data rather than "didn't earn back within however long we
        looked"."""
        if self.earnback_years is None:
            return "beyond horizon"
        return f"{self.earnback_years:.2f} years (quarter {self.earnback_period_index})"


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


def _distributable_cash_above_target_mm(
    net_income_available_to_common_mm: np.ndarray,
    cet1_at_close_mm: float,
    rwa_at_close_mm: float,
    target_cet1_ratio: float,
) -> np.ndarray:
    """One entity's (pro forma OR standalone acquirer) distributable
    cash each post-close quarter under a "retain only what's needed to
    hold CET1 at `target_cet1_ratio`" policy: `payout[t] = max(cet1[t-1]
    + net_income[t] - target_cet1_mm, 0)`. RWA is held FLAT at
    `rwa_at_close_mm` for the whole horizon (this project doesn't
    project RWA growth beyond close -- see `corefin.ma.capital`'s own
    module docstring), so once CET1 reaches the target level it never
    needs to retain anything further: if `cet1_at_close_mm` already
    exceeds the target, the WHOLE excess is released in the first
    quarter along with that quarter's income, and every quarter after
    that pays out 100% of net income, since the target capital
    requirement never grows. If `cet1_at_close_mm` starts BELOW target,
    quarters retain 100% of income (payout floored at 0) until CET1
    reaches the target, then switch to 100% payout."""
    target_cet1_mm = target_cet1_ratio * rwa_at_close_mm
    cet1_mm = cet1_at_close_mm
    distributable_mm = np.empty(len(net_income_available_to_common_mm))
    for t, net_income_mm in enumerate(net_income_available_to_common_mm):
        payout_mm = max(cet1_mm + net_income_mm - target_cet1_mm, 0.0)
        distributable_mm[t] = payout_mm
        cet1_mm = cet1_mm + net_income_mm - payout_mm
    return distributable_mm


def compute_irr_from_cash_flows(cash_flows: np.ndarray) -> float:
    """Solves for the ANNUALIZED internal rate of return of a quarterly
    `cash_flows` array (CF[0] typically the initial outlay, negative;
    the rest the periodic returns, with any terminal value already
    folded into the last entry) via `scipy.optimize.brentq` on the
    quarterly rate, then annualizes: `(1 + quarterly_irr) ** 4 - 1`."""

    def npv(quarterly_rate: float) -> float:
        periods = np.arange(len(cash_flows))
        return float(np.sum(cash_flows / (1.0 + quarterly_rate) ** periods))

    quarterly_irr = brentq(npv, -0.99, 10.0)
    return (1.0 + quarterly_irr) ** 4 - 1.0


def compute_acquirer_irr(
    acquirer_result: BankModelResult,
    pro_forma: ProFormaProjection,
    sources_and_uses: SourcesAndUses,
    pro_forma_cet1_at_close_mm: float,
    pro_forma_rwa_at_close_mm: float,
    config: DealConfig,
) -> float:
    """Illustrative deal IRR over the available projection horizon (see
    `config.deal_horizon_quarters`) -- CF[0] = -total consideration (cash
    + stock, both treated as a real cost: stock dilutes existing
    shareholders just as cash would be spent); CF[1..n-1] and the
    terminal value follow `config.irr` (`IrrConfig`), per this module's
    own docstring -- NOT full incremental net income plus incremental
    tangible equity, which would double-count retained earnings (the
    original Stage 4 bug)."""
    irr_config = config.irr
    standalone_nicc_mm = acquirer_result.income_statement.net_income_available_to_common_mm

    if irr_config.distributable_cash_method == DistributableCashMethod.DIVIDENDS:
        incremental_distributable_mm = (
            pro_forma.dividends_mm[1:] - acquirer_result.income_statement.dividends_mm[1:]
        )
    else:
        assert irr_config.target_cet1_ratio is not None  # enforced by IrrConfig's own validator
        pro_forma_distributable_mm = _distributable_cash_above_target_mm(
            pro_forma.net_income_available_to_common_mm[1:],
            pro_forma_cet1_at_close_mm,
            pro_forma_rwa_at_close_mm,
            irr_config.target_cet1_ratio,
        )
        standalone_distributable_mm = _distributable_cash_above_target_mm(
            standalone_nicc_mm[1:],
            float(acquirer_result.capital.cet1_capital_mm[0]),
            float(acquirer_result.capital.rwa_mm[0]),
            irr_config.target_cet1_ratio,
        )
        incremental_distributable_mm = pro_forma_distributable_mm - standalone_distributable_mm

    cash_flows = np.concatenate(
        [[-sources_and_uses.consideration_mm], incremental_distributable_mm]
    )

    standalone_tbv_mm = (
        acquirer_result.balance_sheet.equity_mm
        - acquirer_result.opening.preferred_stock_mm
        - acquirer_result.balance_sheet.goodwill_mm
        - acquirer_result.balance_sheet.other_intangibles_mm
    )
    incremental_tbv_mm = pro_forma.tangible_common_equity_mm - standalone_tbv_mm

    if irr_config.exit_multiple_basis == ExitMultipleBasis.FORWARD_PE:
        incremental_nicc_mm = pro_forma.net_income_available_to_common_mm - standalone_nicc_mm
        n_final_year = min(4, len(incremental_distributable_mm))
        exit_basis_mm = float(incremental_nicc_mm[-n_final_year:].sum())
    else:
        exit_basis_mm = float(incremental_tbv_mm[-1])
    cash_flows[-1] += irr_config.exit_multiple * exit_basis_mm

    return compute_irr_from_cash_flows(cash_flows)
