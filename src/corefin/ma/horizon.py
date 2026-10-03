"""Extends a `CreditLossProjection`'s horizon for Stage 4 deal economics
(EPS accretion, TBV earnback, IRR), which run over their own default
5-year (20-quarter) post-close horizon -- separate from whatever shorter
or longer horizon a credit-engine scenario natively covers (Project #7's
official DFAST/CCAR scenarios run a fixed 9-quarter regulatory horizon;
an ad hoc baseline scenario pulled for this project's own real-data
validation ran 13 projected quarters).

DOCUMENTED EXTENSION RULE: beyond the scenario's own last projected
quarter, each category's annualized NCO rate is held FLAT at that last
quarter's own rate (the "long-run loss rate" -- there is no scenario
path beyond the credit engine's own horizon, so holding the terminal
rate flat is the simplest defensible continuation, not a reversion to
some other historical average), and `allowance_required` is held FLAT
at its own last-scenario-quarter level too (by the end of a multi-year
scenario, the lifetime-expected-loss WAL window has already absorbed
the scenario's own late-horizon quarters -- see
`credit.projection.build_cecl_projection`'s windowing -- so growing the
allowance further would require a NEW rate assumption, and this module
makes none). With both held flat, the allowance roll-forward identity
(provision = allowance's own quarter-over-quarter change, plus that
quarter's net charge-off) collapses to provision = net charge-off for
every extended quarter. Loan `balance_mm` is already held flat across
the WHOLE horizon by the credit engine itself (Project #7's own static-
balance simplification), so no extension decision is needed there --
padding with the same flat value is exact, not an approximation."""

from __future__ import annotations

import numpy as np

from corefin.bank.model import BankModelResult, run_bank_model
from corefin.bank.schema import BankConfig, BankOpeningBalance
from corefin.credit.interface import CreditLossProjection
from corefin.timeline import Timeline


def extend_credit_loss_projection(
    projection: CreditLossProjection, n_periods: int
) -> CreditLossProjection:
    """Returns `projection` unchanged if it already has exactly
    `n_periods`. Raises `ValueError` if `n_periods` is SHORTER than
    `projection`'s own horizon -- truncating isn't this function's
    concern; slice the arrays yourself if that's what you need."""
    original_n = projection.timeline.n_periods
    if n_periods == original_n:
        return projection
    if n_periods < original_n:
        raise ValueError(
            f"n_periods ({n_periods}) is shorter than projection's own horizon "
            f"({original_n}) -- extend_credit_loss_projection only extends, it never truncates"
        )

    n_historical = int(np.sum(~projection.timeline.is_projection))
    extended_timeline = Timeline.quarterly(
        n_periods=n_periods,
        n_historical=n_historical,
        start_year=projection.timeline.start_year,
        start_quarter=projection.timeline.start_quarter,
    )
    n_extra = n_periods - original_n

    def _pad_flat(array: np.ndarray) -> np.ndarray:
        return np.concatenate([array, np.tile(array[:, -1:], (1, n_extra))], axis=1)

    balance_mm = _pad_flat(projection.balance_mm)
    nco_rate = _pad_flat(projection.nco_rate)
    npl_ratio = _pad_flat(projection.npl_ratio)
    allowance_mm = _pad_flat(projection.allowance_mm)
    npl_mm = balance_mm * npl_ratio

    extra_net_charge_off_mm = (nco_rate[:, -n_extra:] / 4.0) * balance_mm[:, -n_extra:]
    # allowance is held flat beyond the scenario (see module docstring), so the allowance
    # roll-forward's own delta term is zero for every extended quarter -- provision collapses
    # to that quarter's net charge-off alone.
    extra_provision_expense_mm = extra_net_charge_off_mm

    net_charge_off_mm = np.concatenate(
        [projection.net_charge_off_mm, extra_net_charge_off_mm], axis=1
    )
    provision_expense_mm = np.concatenate(
        [projection.provision_expense_mm, extra_provision_expense_mm], axis=1
    )

    return CreditLossProjection(
        timeline=extended_timeline,
        categories=projection.categories,
        scenario_name=projection.scenario_name,
        bank_identifier=projection.bank_identifier,
        balance_mm=balance_mm,
        net_charge_off_mm=net_charge_off_mm,
        provision_expense_mm=provision_expense_mm,
        allowance_mm=allowance_mm,
        npl_mm=npl_mm,
        nco_rate=nco_rate,
        npl_ratio=npl_ratio,
        monte_carlo_mean=projection.monte_carlo_mean,
        monte_carlo_percentiles=projection.monte_carlo_percentiles,
    )


def build_deal_horizon_bank_result(
    opening: BankOpeningBalance,
    credit_projection: CreditLossProjection,
    bank_config: BankConfig,
    deal_horizon_quarters: int,
) -> BankModelResult:
    """Convenience wrapper: extends `credit_projection` to
    `deal_horizon_quarters` PROJECTED quarters (`deal_horizon_quarters +
    1` periods, including the jump-off quarter) if it's shorter, then
    reruns `corefin.bank.model.run_bank_model` over that horizon. Pass
    the result into `corefin.ma.model.run_deal_model` in place of a
    bank's native-horizon `BankModelResult` wherever Stage 4 deal
    economics are needed -- the at-close Stage 3 figures only ever read
    period 0, so they're unaffected either way."""
    n_periods = deal_horizon_quarters + 1
    extended = extend_credit_loss_projection(credit_projection, n_periods)
    return run_bank_model(opening, extended, bank_config, extended.timeline)
