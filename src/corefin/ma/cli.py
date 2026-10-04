"""CLI stub for Project #2 (the Bank M&A CET1 & Accretion Simulator):
`corefin ma run --config configs/example_bank_deal.yaml` runs one deal
end to end against real, already-fetched FFIEC Call Report / FR Y-9C /
credit-engine data and prints a deal summary (EPS accretion in all three
views, TBV dilution/earnback, IRR across exit multiples, pro forma
capital at close, each bank's realized vs baseline projected NCO rate,
and a Stage 5 stress test -- standalone acquirer and the pro forma
combined bank's minimum CET1 ratio under the Fed's severely adverse
scenario, vs. the Basel III 4.5% minimum, see `ma/stress.py`). A fuller
CLI (Excel export, charts, general multi-deal config) is Stage 7,
queued -- this is deliberately a stub covering Stage 4/5's own
deliverables.

Needs the SAME locally-fetched raw/processed data files the rest of
this project's real-data workflows do (never committed -- `data/` is
gitignored): the acquirer/target's Call Report and FR Y-9C bulk files,
plus the credit engine's modeling dataset/macro history/industry
history/model selection/baseline AND severely adverse scenario outputs
(`corefin credit fetch`/`build`/`fetch-macro-fred`/`fetch-industry-
history`/`fit-models`/`project`/`fetch-scenarios` -- see credit/cli.py).
Fails with a clear, named "no such file" error (not an opaque pandas/
numpy traceback) if any are missing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
import yaml
from pydantic import BaseModel, ConfigDict

from corefin.bank.model import BankModelResult
from corefin.bank.schema import BankConfig
from corefin.bank.sources.capital_parse import parse_bank_capital_zip
from corefin.bank.sources.fr_y9c import parse_y9c_bulk_zip
from corefin.bank.sources.real_data import DataQualityFlags, build_opening_balance_from_real_data
from corefin.credit import backtest, interface
from corefin.credit.cli import _load_long_history_frames
from corefin.credit.projection import FED_COMPARISON_QUARTERS
from corefin.ma.accretion import compute_acquirer_irr_sensitivity
from corefin.ma.horizon import build_deal_horizon_bank_result
from corefin.ma.model import (
    check_cet1_bridge_matches_balance_sheet,
    check_goodwill_equals_consideration_less_fair_value,
    check_pcd_has_no_net_effect_at_close,
    check_pro_forma_balance_sheet_balances,
    run_deal_model,
)
from corefin.ma.schema import DealConfig, ExitMultipleBasis
from corefin.ma.stress import run_stress_test

app = typer.Typer(add_completion=False)

QUARTER_Q4 = pd.Period("2025Q4", freq="Q")
QUARTER_Q3 = pd.Period("2025Q3", freq="Q")
EXIT_MULTIPLE_SENSITIVITY = (1.0, 1.25, 1.5, 1.75, 2.0)

# This example's acquirer is the one real-data quirk this stub hardcodes rather than
# generalizes (a fuller Stage 7 CLI would make this a per-bank config option): its assets
# grew >10% QoQ (2025Q3->Q4, apparently its own acquisition), so the bank.sources.real_data
# YTD/4 fallback materially understates its Q4-only run-rate -- see that module's own
# docstring. The FFIEC NIC bulk-download subsystem (FR Y-9C Q3 data) has been down for the
# full session this project was built in, so this substitutes the bank-level (Call Report)
# Q3/Q4 delta, confirmed to closely match independently-sourced real figures for this bank.
ACQUIRER_QOQ_OVERRIDE_BANK_ID = "34537"
ACQUIRER_PREFERRED_DIVIDEND_ANNUAL_RATE = 0.06  # contractual-rate assumption, not derived


class BankIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    bank_id: str  # Call Report RSSD ID (bank subsidiary)
    hc_rssd_id: str  # FR Y-9C RSSD ID (holding company)


class ExampleDealConfig(BaseModel):
    """`configs/example_bank_deal.yaml`'s own schema: RSSD identification
    for the two real banks (never named directly in any committed output
    -- see that file's own header), plus every `ma.schema.DealConfig`
    assumption, all in one file."""

    model_config = ConfigDict(extra="forbid")
    acquirer: BankIdentity
    target: BankIdentity
    deal: DealConfig


def load_example_deal_config(path: Path) -> ExampleDealConfig:
    return ExampleDealConfig.model_validate(yaml.safe_load(path.read_text()))


def _require_file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise typer.BadParameter(f"{path} does not exist -- {hint}")
    return path


def _balance_weighted_total_nco_rate(
    credit_proj: interface.CreditLossProjection,
    training: pd.DataFrame,
    bank_id: str,
    as_of_quarter: pd.Period,
) -> tuple[float, float]:
    """(realized_total, baseline_projected_total) annualized NCO rate,
    balance-weighted across the bank's own categories: realized = each
    category's own last-actual (`as_of_quarter`) winsorized_nco_rate;
    projected = each category's own first-projected-quarter rate
    (`credit_proj.nco_rate`, period 1) -- both weighted by that
    category's own jump-off balance."""
    realized_weighted = realized_weight = projected_weighted = projected_weight = 0.0
    for i, category in enumerate(credit_proj.categories):
        rows = training[
            (training["bank_id"] == bank_id)
            & (training["category"] == category)
            & (training["quarter"] == as_of_quarter)
        ]
        if rows.empty:
            continue
        balance = float(credit_proj.balance_mm[i, 0])
        realized_weighted += float(rows["winsorized_nco_rate"].iloc[0]) * balance
        realized_weight += balance
        projected_weighted += float(credit_proj.nco_rate[i, 1]) * balance
        projected_weight += balance
    return realized_weighted / realized_weight, projected_weighted / projected_weight


def _build_credit_projection(
    identity: BankIdentity,
    scenario_name: str,
    scenario: pd.DataFrame,
    training: pd.DataFrame,
    macro_history: pd.DataFrame,
    long_history_frames: dict,
    categories: list[str],
    selected_family_by_category: dict[str, str],
) -> interface.CreditLossProjection:
    bank_rows_all = training[training["bank_id"] == identity.bank_id]
    current_categories = set(
        bank_rows_all.groupby("category")["quarter"].max().loc[lambda s: s == QUARTER_Q4].index
    )
    bank_categories = [c for c in categories if c in current_categories]
    return interface.build_credit_loss_projection(
        bank_categories,
        selected_family_by_category,
        training,
        macro_history,
        scenario,
        scenario_name,
        long_history_frames,
        bank_id=identity.bank_id,
    )


def _build_bank_result(
    identity: BankIdentity,
    training: pd.DataFrame,
    macro_history: pd.DataFrame,
    scenario: pd.DataFrame,
    long_history_frames: dict,
    categories: list[str],
    selected_family_by_category: dict[str, str],
    call_df_q4: pd.DataFrame,
    call_df_q3: pd.DataFrame,
    y9c_df: pd.DataFrame,
    deal_horizon_quarters: int,
) -> tuple[BankModelResult, BankConfig, DataQualityFlags, float, float]:
    """Returns (result, bank_config, flags, realized_total_nco_rate,
    baseline_projected_total_nco_rate)."""
    credit_projection = _build_credit_projection(
        identity,
        "baseline",
        scenario,
        training,
        macro_history,
        long_history_frames,
        categories,
        selected_family_by_category,
    )
    realized_total, projected_total = _balance_weighted_total_nco_rate(
        credit_projection, training, identity.bank_id, QUARTER_Q4
    )

    call_row = call_df_q4[call_df_q4["bank_id"] == identity.bank_id].iloc[0]
    y9c_row = y9c_df[y9c_df["rssd_id"] == identity.hc_rssd_id].iloc[0]
    net_loans_mm = credit_projection.balance_total_mm[0] - credit_projection.allowance_total_mm[0]

    build_kwargs = dict(
        name=identity.label,
        bank_id=identity.bank_id,
        hc_rssd_id=identity.hc_rssd_id,
        call_report_row=call_row,
        y9c_row=y9c_row,
        bank_level_net_loans_mm=net_loans_mm,
    )
    if identity.bank_id == ACQUIRER_QOQ_OVERRIDE_BANK_ID:
        acq_q4 = call_df_q4[call_df_q4["bank_id"] == identity.bank_id].iloc[0]
        acq_q3 = call_df_q3[call_df_q3["bank_id"] == identity.bank_id].iloc[0]
        build_kwargs["net_interest_income_quarterly_override_mm"] = (
            (acq_q4["RIAD4107"] - acq_q4["RIAD4073"]) - (acq_q3["RIAD4107"] - acq_q3["RIAD4073"])
        ) / 1000.0
        build_kwargs["noninterest_income_quarterly_override_mm"] = (
            acq_q4["RIAD4079"] - acq_q3["RIAD4079"]
        ) / 1000.0
        build_kwargs["noninterest_expense_quarterly_override_mm"] = (
            acq_q4["RIAD4093"] - acq_q3["RIAD4093"]
        ) / 1000.0
        build_kwargs["preferred_dividend_annual_rate"] = ACQUIRER_PREFERRED_DIVIDEND_ANNUAL_RATE

    opening, flags = build_opening_balance_from_real_data(**build_kwargs)
    bank_config = BankConfig()
    result = build_deal_horizon_bank_result(
        opening, credit_projection, bank_config, deal_horizon_quarters=deal_horizon_quarters
    )
    return result, bank_config, flags, realized_total, projected_total


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, help="Path to an ExampleDealConfig YAML."
    ),
    call_report_q4_zip: Path = typer.Option(
        Path("data/raw/ffiec/12-31-2025.zip"), "--call-report-q4"
    ),
    call_report_q3_zip: Path = typer.Option(
        Path("data/raw/ffiec/09-30-2025.zip"), "--call-report-q3"
    ),
    y9c_q4_zip: Path = typer.Option(Path("data/raw/fr_y9c/BHCF20251231.ZIP"), "--y9c-q4"),
    modeling_dataset_path: Path = typer.Option(
        Path("data/processed/modeling_dataset.parquet"), "--modeling-dataset"
    ),
    macro_history_path: Path = typer.Option(
        Path("data/processed/macro_history.parquet"), "--macro-history"
    ),
    industry_history_path: Path = typer.Option(
        Path("data/processed/industry_history_fred.parquet"), "--industry-history"
    ),
    model_selection_path: Path = typer.Option(
        Path("data/processed/projections/model_selection.csv"), "--model-selection"
    ),
    baseline_scenario_path: Path = typer.Option(
        Path("data/processed/scenarios/baseline.parquet"), "--baseline-scenario"
    ),
    severely_adverse_scenario_path: Path = typer.Option(
        Path("data/processed/scenarios/severely_adverse.parquet"), "--severely-adverse-scenario"
    ),
) -> None:
    """Runs `config` (an `ExampleDealConfig` YAML -- see `configs/
    example_bank_deal.yaml`) end to end: builds each bank's credit-loss
    projection (Project #7) and Stage 2 standalone model (with allowance
    anchoring and the bank-relative rate scaling, both automatic once
    `BankOpeningBalance.reported_allowance_mm` is set), extends both to
    the deal's own horizon, runs Stage 3/4 purchase accounting and deal
    economics, checks the Stage 3 integrity checks, runs the Stage 5
    stress test against the Fed's severely adverse scenario over its own
    9-quarter DFAST window (separate from the baseline-scenario deal
    horizon above), and prints a deal summary."""
    for path, hint in (
        (
            call_report_q4_zip,
            "run `corefin credit fetch` for this quarter, or pass --call-report-q4",
        ),
        (
            call_report_q3_zip,
            "run `corefin credit fetch` for this quarter, or pass --call-report-q3",
        ),
        (
            y9c_q4_zip,
            "download it from the FFIEC NIC site (see bank/sources/fr_y9c.py), or pass --y9c-q4",
        ),
        (modeling_dataset_path, "run `corefin credit build-modeling-dataset-cmd`"),
        (macro_history_path, "run `corefin credit fetch-macro-fred`"),
        (model_selection_path, "run `corefin credit project` first"),
        (baseline_scenario_path, "run `corefin credit fetch-scenarios`"),
        (severely_adverse_scenario_path, "run `corefin credit fetch-scenarios`"),
    ):
        _require_file(path, hint)

    example = load_example_deal_config(config)

    with open(call_report_q4_zip, "rb") as f:
        call_df_q4, _ = parse_bank_capital_zip(f.read(), QUARTER_Q4)
    with open(call_report_q3_zip, "rb") as f:
        call_df_q3, _ = parse_bank_capital_zip(f.read(), QUARTER_Q3)
    with open(y9c_q4_zip, "rb") as f:
        y9c_df = parse_y9c_bulk_zip(f.read(), QUARTER_Q4)

    training = pd.read_parquet(modeling_dataset_path)
    training["quarter"] = pd.PeriodIndex(training["quarter"].astype(str), freq="Q")
    macro_history = pd.read_parquet(macro_history_path)
    macro_history["quarter"] = pd.PeriodIndex(macro_history["quarter"].astype(str), freq="Q")
    macro_history = macro_history.set_index("quarter")
    long_history_frames = (
        _load_long_history_frames(industry_history_path, macro_history)
        if industry_history_path.exists()
        else {}
    )
    model_selection = pd.read_csv(model_selection_path)
    selected_family_by_category = (
        model_selection[model_selection["selected"]].set_index("category")["model_family"].to_dict()
    )
    categories = [
        str(c) for c in backtest.MODELING_CATEGORIES if str(c) in selected_family_by_category
    ]

    scenario = pd.read_parquet(baseline_scenario_path)
    scenario["quarter"] = pd.PeriodIndex(scenario["quarter"].astype(str), freq="Q")
    scenario = scenario.set_index("quarter")

    severely_adverse_scenario = pd.read_parquet(severely_adverse_scenario_path)
    severely_adverse_scenario["quarter"] = pd.PeriodIndex(
        severely_adverse_scenario["quarter"].astype(str), freq="Q"
    )
    severely_adverse_scenario = severely_adverse_scenario.set_index("quarter")

    results: dict[str, BankModelResult] = {}
    bank_configs: dict[str, BankConfig] = {}
    for key, identity in (("acquirer", example.acquirer), ("target", example.target)):
        result, bank_config, flags, realized_total, projected_total = _build_bank_result(
            identity,
            training,
            macro_history,
            scenario,
            long_history_frames,
            categories,
            selected_family_by_category,
            call_df_q4,
            call_df_q3,
            y9c_df,
            example.deal.deal_horizon_quarters,
        )
        results[key] = result
        bank_configs[key] = bank_config
        typer.echo(
            f"{identity.label}: realized NCO rate = {realized_total:.4%}, "
            f"baseline projected (Q1) NCO rate = {projected_total:.4%}, "
            f"QoQ asset change flagged = {flags.qoq_asset_change_flagged}"
        )

    deal_result = run_deal_model(
        acquirer_result=results["acquirer"],
        target_result=results["target"],
        acquirer_bank_config=bank_configs["acquirer"],
        target_bank_config=bank_configs["target"],
        config=example.deal,
    )

    typer.echo("\n=== Stage 3 Checks ===")
    typer.echo(check_pro_forma_balance_sheet_balances(deal_result).describe())
    typer.echo(check_goodwill_equals_consideration_less_fair_value(deal_result).describe())
    typer.echo(check_pcd_has_no_net_effect_at_close(deal_result).describe())
    typer.echo(
        check_cet1_bridge_matches_balance_sheet(
            results["acquirer"].opening,
            deal_result,
            results["acquirer"].capital.unexplained_cet1_residual_mm,
        ).describe()
    )

    r = deal_result.pro_forma_capital_ratios
    typer.echo("\n=== Pro Forma Capital at Close ===")
    typer.echo(f"CET1 ratio:            {r.pro_forma_cet1_ratio:.4%}")
    typer.echo(f"Tier 1 leverage ratio: {r.pro_forma_tier1_leverage_ratio:.4%}")
    if r.pro_forma_total_capital_ratio is not None:
        typer.echo(f"Total capital ratio:   {r.pro_forma_total_capital_ratio:.4%}")

    typer.echo("\n=== EPS Accretion/Dilution (3 views) ===")
    eps = deal_result.eps_accretion
    exchange = f"  Exchange ratio: {eps.exchange_ratio:.4f}" if eps.exchange_ratio else ""
    typer.echo(
        f"New shares issued: {eps.new_shares_issued_mm:.2f}mm  "
        f"Pro forma shares: {eps.pro_forma_shares_outstanding_mm:.2f}mm{exchange}"
    )
    for year, a in eps.annual().items():
        typer.echo(f"  {year}: standalone EPS=${a.standalone_eps:.2f}")
        for label, pro_forma, dollar, pct in (
            (
                "(a) GAAP:",
                a.pro_forma_eps_gaap,
                a.dollar_accretion_gaap,
                a.accretion_dilution_pct_gaap,
            ),
            (
                "(b) excl. one-time charges:",
                a.pro_forma_eps_excl_one_time,
                a.dollar_accretion_excl_one_time,
                a.accretion_dilution_pct_excl_one_time,
            ),
            (
                "(c) excl. one-time + marks/CDI:",
                a.pro_forma_eps_excl_one_time_and_marks,
                a.dollar_accretion_excl_one_time_and_marks,
                a.accretion_dilution_pct_excl_one_time_and_marks,
            ),
        ):
            pct_str = f"{pct:+.2%}" if a.accretion_pct_meaningful else "n.m."
            typer.echo(
                f"    {label:<32} pro forma=${pro_forma:.2f}  "
                f"$accretion={dollar:+.2f}  %accretion={pct_str}"
            )

    tbv = deal_result.tbv_earnback
    typer.echo("\n=== TBV Dilution & Earnback ===")
    typer.echo(f"TBV/share dilution at close: {tbv.tbv_dilution_at_close_pct:+.2%}")
    typer.echo(f"Earnback: {tbv.earnback_label}")

    typer.echo("\n=== Acquirer IRR across exit multiples ===")
    irr_by_multiple = compute_acquirer_irr_sensitivity(
        results["acquirer"],
        deal_result.pro_forma_projection,
        deal_result.sources_and_uses,
        deal_result.pro_forma_cet1_bridge.pro_forma_cet1_mm,
        deal_result.pro_forma_capital_ratios.pro_forma_rwa_mm,
        example.deal,
        exit_multiples=list(EXIT_MULTIPLE_SENSITIVITY),
    )
    basis_label = {
        ExitMultipleBasis.PRICE_TO_TBV: "P/TBV",
        ExitMultipleBasis.FORWARD_PE: "forward P/E",
    }[example.deal.irr.exit_multiple_basis]
    base_case = example.deal.irr.exit_multiple
    for multiple, irr in irr_by_multiple.items():
        marker = "  <-- base case" if multiple == base_case else ""
        typer.echo(f"  {multiple:.2f}x {basis_label} exit: IRR = {irr:.2%}{marker}")

    typer.echo(f"\n=== Stage 5: Stress Test (Fed severely adverse, {FED_COMPARISON_QUARTERS}Q) ===")
    acquirer_severely_adverse_projection = _build_credit_projection(
        example.acquirer,
        "severely_adverse",
        severely_adverse_scenario,
        training,
        macro_history,
        long_history_frames,
        categories,
        selected_family_by_category,
    )
    target_severely_adverse_projection = _build_credit_projection(
        example.target,
        "severely_adverse",
        severely_adverse_scenario,
        training,
        macro_history,
        long_history_frames,
        categories,
        selected_family_by_category,
    )
    stress_result = run_stress_test(
        results["acquirer"].opening,
        results["target"].opening,
        acquirer_severely_adverse_projection,
        target_severely_adverse_projection,
        bank_configs["acquirer"],
        bank_configs["target"],
        deal_result,
        example.deal,
        acquirer_unexplained_cet1_residual_mm=results[
            "acquirer"
        ].capital.unexplained_cet1_residual_mm,
    )
    for label, path in (
        ("Acquirer standalone", stress_result.acquirer_standalone),
        ("Pro forma combined", stress_result.pro_forma_combined),
    ):
        breach = "BREACHES" if path.breaches_4_5_pct_minimum else "does not breach"
        typer.echo(
            f"  {label}: starting CET1={path.starting_cet1_ratio:.4%}  "
            f"minimum CET1={path.minimum_cet1_ratio:.4%} ({breach} the 4.5% minimum)  "
            f"illustrative SCB={path.illustrative_stress_capital_buffer:.4%}"
        )
    typer.echo(
        f"  Deal impact: minimum stressed CET1 ratio changes by "
        f"{stress_result.minimum_cet1_ratio_change_pp:+.4%} (pro forma combined vs. "
        f"acquirer standalone)"
    )


if __name__ == "__main__":
    app()
