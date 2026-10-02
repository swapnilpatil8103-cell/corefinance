"""Survivorship-bias sensitivity for `anchored_to_aggregate`
(`corefin credit anchoring-sensitivity`): its per-bank "relative level"
anchoring factor (models.compute_bank_relative_levels) is computed from
TODAY's SURVIVING bank population's own FULL-SAMPLE (2001Q1-2025Q4)
history -- a bank that failed or was acquired during 2008-2010 never
contributes to it, even though it existed and had its own (plausibly
worse) loss experience then, because `forecast_anchored_to_aggregate`
only ever applies the factor to banks present in the STATIC projection
population (today's survivors). This is a real, documented
simplification, not a bug (see projection.py's module docstring on the
static balance sheet convention) -- this module makes it checkable.

Three variants, compared under the SAME scenario/fitted-aggregate-
forecast mechanics `projection.project_category_nco_rate` already uses
-- no change to that function or to which family is actually SELECTED
for a real projection; this is a reportable sensitivity check only:

(a) `full_sample_survivors` -- the PRODUCTION default: today's survivors'
    own full-sample relative levels (exactly `project_category_nco_rate`
    with family="anchored_to_aggregate").
(b) `industry_average` -- every bank's relative level forced to 1.0 (no
    bank-specific differentiation at all). Mathematically IDENTICAL to
    the raw aggregate_long forecast: a balance-weighted average of a
    constant equals that constant, so this is computed by calling
    `project_category_nco_rate` with family="aggregate_long" directly,
    not by actually setting every level to 1.0 and redoing the
    arithmetic -- same result, no new arithmetic needed.
(c) `crisis_era_survivors` -- relative levels computed ONLY from
    [CRISIS_START, CRISIS_END] (default: the 2007Q4-2010Q4 financial
    crisis), for whichever of today's survivors were ALSO present then.
    A survivor absent from the crisis window (e.g. chartered after it)
    gets no (c) factor and is dropped from THIS variant's aggregation --
    the same "can't project a bank with no basis" limitation
    `forecast_anchored_to_aggregate` already documents for any unseen
    bank.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from corefin.credit import macro, models, projection

CRISIS_START = projection.CRISIS_REPLAY_START
CRISIS_END = projection.CRISIS_REPLAY_END

ANCHORING_VARIANTS: tuple[str, ...] = (
    "full_sample_survivors",
    "industry_average",
    "crisis_era_survivors",
)


def project_with_crisis_era_anchoring(
    category: str,
    category_train_dataset: pd.DataFrame,
    macro_history: pd.DataFrame,
    scenario: pd.DataFrame,
    long_history_frame: pd.DataFrame,
    crisis_start: pd.Period = CRISIS_START,
    crisis_end: pd.Period = CRISIS_END,
) -> pd.Series:
    """Variant (c): identical to `projection.project_category_nco_rate`'s
    own "anchored_to_aggregate" branch (same aggregate_long fit, same
    scenario macro path, same static "last actual quarter" bank
    population), EXCEPT the relative-level anchoring factor is computed
    from `category_train_dataset`/`long_history_frame` restricted to
    [`crisis_start`, `crisis_end`] instead of the full sample."""
    jump_off_quarter = macro_history.index.max()
    long_history_frame = long_history_frame[long_history_frame["quarter"] <= jump_off_quarter]

    full_macro_path = macro.build_full_macro_path(macro_history, scenario)
    projection_quarters = list(scenario.index)
    macro_lag_frame = projection.build_projection_macro_lag_frame(
        full_macro_path, projection_quarters
    )

    long_extended = pd.concat(
        [long_history_frame, macro_lag_frame], ignore_index=True, sort=False
    )
    long_result = models.fit_aggregate_model(
        long_history_frame, category, include_pandemic_dummy=True
    )
    long_forecast = models.forecast_aggregate_dynamic(
        long_result,
        long_extended,
        category,
        projection_quarters[0],
        projection_quarters[-1],
        include_pandemic_dummy=True,
    )

    crisis_bank = category_train_dataset[
        (category_train_dataset["quarter"] >= crisis_start)
        & (category_train_dataset["quarter"] <= crisis_end)
    ]
    crisis_long = long_history_frame[
        (long_history_frame["quarter"] >= crisis_start)
        & (long_history_frame["quarter"] <= crisis_end)
    ]
    relative_levels = models.compute_bank_relative_levels(
        crisis_bank, "winsorized_nco_rate", crisis_long
    )

    last_quarter = category_train_dataset["quarter"].max()
    last_actual_bank_quarter = category_train_dataset[
        category_train_dataset["quarter"] == last_quarter
    ]
    synthetic_future_bank = projection.build_synthetic_future_bank_frame(
        last_actual_bank_quarter, macro_lag_frame
    )
    predicted_bank = models.forecast_anchored_to_aggregate(
        relative_levels, long_forecast, synthetic_future_bank
    )
    return models.aggregate_bank_predictions_to_industry_rate(synthetic_future_bank, predicted_bank)


def compute_anchoring_variants(
    category: str,
    category_train_dataset: pd.DataFrame,
    macro_history: pd.DataFrame,
    scenario: pd.DataFrame,
    long_history_frame: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Returns {variant -> forecast Series} for all of `ANCHORING_
    VARIANTS`, under `scenario`. `category` must be one whose selected
    family is (or could be) anchored_to_aggregate -- i.e. `long_history_
    frame` must be available."""
    return {
        "full_sample_survivors": projection.project_category_nco_rate(
            category,
            "anchored_to_aggregate",
            category_train_dataset,
            macro_history,
            scenario,
            long_history_frame,
        ),
        "industry_average": projection.project_category_nco_rate(
            category,
            "aggregate_long",
            category_train_dataset,
            macro_history,
            scenario,
            long_history_frame,
        ),
        "crisis_era_survivors": project_with_crisis_era_anchoring(
            category, category_train_dataset, macro_history, scenario, long_history_frame
        ),
    }


@dataclass(frozen=True)
class AnchoringVariantResult:
    category: str
    variant: str
    severely_adverse_9q_loss_rate_percent: float
    fed_dfast_2026_9q_loss_rate_percent: float | None
    gap_pct_points: float | None
