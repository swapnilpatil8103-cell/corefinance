"""Anchors a `CreditLossProjection`'s allowance (and the provision this
implies) to a bank's own REPORTED jump-off allowance, instead of taking
the credit engine's own modeled jump-off level directly.

THE BUG THIS FIXES: the credit engine's own jump-off allowance (`credit.
projection.build_cecl_projection`'s "jump_off_rate" row) is computed from
BACKWARD-LOOKING realized NCO history, while every PROJECTED quarter's
allowance is computed from the FORWARD-LOOKING scenario's own NCO path
(see that module's docstring). For a bank whose scenario path diverges
from its own realized history, those two bases don't level-match, so
feeding the model's allowance straight into `corefin.bank.balance_sheet`
produces a large, artificial, ONE-TIME provision at the very first
projected quarter -- not a real credit event, just a methodology seam
between two different allowance ESTIMATES of the SAME jump-off quarter
-- confirmed against the real acquirer: a $285.8mm one-time provision
spike in period 1 versus an ~$18-19mm/quarter steady state everywhere
else.

THE FIX: anchor period 0 to the bank's own REPORTED allowance (a real,
reported number, not a model estimate), then scale every OTHER period
by the MODEL's own relative path: `allowance_t = reported_allowance x
(model_allowance_t / model_allowance_0)`. This keeps the model's own
loss-rate DYNAMICS (the shape of the scenario path) while removing the
bias in its ABSOLUTE LEVEL, so there is no seam at the jump-off-to-
projection transition: period 0 is exactly the reported figure by
construction, and period 1 is a smooth, proportional continuation of
it -- a bank whose model path is flat (steady losses) anchors to a flat
path too, with NO provision spike, regardless of how far the model's
own raw jump-off estimate happened to sit from the reported figure.

TOTAL, NOT PER-CATEGORY: real reported allowance data is bank-TOTAL
only (Call Report RCON3123, RCFD3123 fallback -- confirmed in `credit.
projection.CalibrationCheck`'s own docstring: there is no real per-
category allowance to anchor against). This anchors the TOTAL and
preserves the model's own category MIX exactly (each category's own
share of the anchored total, each period, equals its share of the
model's own unanchored total that period, since every category is
scaled by the SAME constant factor) -- only the aggregate LEVEL is
corrected, not the model's own relative shape across categories."""

from __future__ import annotations

import numpy as np

from corefin.credit.interface import CreditLossProjection


def anchor_allowance_to_reported(
    credit_projection: CreditLossProjection, reported_allowance_mm: float
) -> CreditLossProjection:
    jumpoff_model_total_mm = float(credit_projection.allowance_total_mm[0])
    if jumpoff_model_total_mm <= 0:
        raise ValueError(
            f"model jump-off allowance was {jumpoff_model_total_mm}, cannot anchor against it"
        )
    scale = reported_allowance_mm / jumpoff_model_total_mm
    anchored_allowance_mm = credit_projection.allowance_mm * scale

    net_charge_off_mm = credit_projection.net_charge_off_mm
    anchored_provision_mm = np.full_like(credit_projection.provision_expense_mm, np.nan)
    anchored_provision_mm[:, 1:] = (
        anchored_allowance_mm[:, 1:] - anchored_allowance_mm[:, :-1]
    ) + net_charge_off_mm[:, 1:]

    return CreditLossProjection(
        timeline=credit_projection.timeline,
        categories=credit_projection.categories,
        scenario_name=credit_projection.scenario_name,
        bank_identifier=credit_projection.bank_identifier,
        balance_mm=credit_projection.balance_mm,
        net_charge_off_mm=net_charge_off_mm,
        provision_expense_mm=anchored_provision_mm,
        allowance_mm=anchored_allowance_mm,
        npl_mm=credit_projection.npl_mm,
        nco_rate=credit_projection.nco_rate,
        npl_ratio=credit_projection.npl_ratio,
        monte_carlo_mean=credit_projection.monte_carlo_mean,
        monte_carlo_percentiles=credit_projection.monte_carlo_percentiles,
    )
