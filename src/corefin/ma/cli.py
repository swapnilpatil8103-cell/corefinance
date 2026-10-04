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

import numpy as np
import pandas as pd
import typer
import yaml
from pydantic import BaseModel, ConfigDict

from corefin.bank import ppnr
from corefin.bank.model import BankModelResult
from corefin.bank.schema import DEFAULT_NIM_BETA, BankConfig, PpnrStressConfig
from corefin.bank.sources.capital_parse import parse_bank_capital_zip
from corefin.bank.sources.fr_y9c import parse_y9c_bulk_zip
from corefin.bank.sources.real_data import DataQualityFlags, build_opening_balance_from_real_data
from corefin.credit import backtest, interface
from corefin.credit.cli import _load_long_history_frames
from corefin.credit.projection import FED_COMPARISON_QUARTERS
from corefin.credit.sources.fed_stress_test_results import DFAST_2026_SEVERELY_ADVERSE_LOSS_RATES
from corefin.ma.accretion import compute_acquirer_irr_sensitivity
from corefin.ma.horizon import (
    build_deal_horizon_bank_result,
    extend_credit_loss_projection,
    truncate_credit_loss_projection,
)
from corefin.ma.model import (
    check_cet1_bridge_matches_balance_sheet,
    check_goodwill_equals_consideration_less_fair_value,
    check_pcd_has_no_net_effect_at_close,
    check_pro_forma_balance_sheet_balances,
    run_deal_model,
)
from corefin.ma.schema import DealConfig, ExitMultipleBasis
from corefin.ma.sensitivity import (
    DEFAULT_MONTE_CARLO_DRAWS,
    DealContext,
    compute_price_cost_save_grid,
    compute_tornado,
    run_monte_carlo,
)
from corefin.ma.stress import StressTestResult, run_stress_test

app = typer.Typer(add_completion=False)

QUARTER_Q4 = pd.Period("2025Q4", freq="Q")
QUARTER_Q3 = pd.Period("2025Q3", freq="Q")
EXIT_MULTIPLE_SENSITIVITY = (1.0, 1.25, 1.5, 1.75, 2.0)

# NIM-beta calibration window (corefin.bank.ppnr) -- the Call Report quarterly bulk ZIP
# filenames for 2020Q1-2021Q4, matching DEFAULT_RAW_DIR's own naming convention
# (credit/cli.py's "{MM-DD-YYYY}.zip"). All 8 are expected to already be cached locally
# (the same raw directory every other real-data workflow in this project uses); a missing
# quarter just falls back to the illustrative default beta for that bank, not a hard error.
CALIBRATION_QUARTER_FILENAMES: dict[pd.Period, str] = {
    pd.Period("2020Q1", freq="Q"): "03-31-2020.zip",
    pd.Period("2020Q2", freq="Q"): "06-30-2020.zip",
    pd.Period("2020Q3", freq="Q"): "09-30-2020.zip",
    pd.Period("2020Q4", freq="Q"): "12-31-2020.zip",
    pd.Period("2021Q1", freq="Q"): "03-31-2021.zip",
    pd.Period("2021Q2", freq="Q"): "06-30-2021.zip",
    pd.Period("2021Q3", freq="Q"): "09-30-2021.zip",
    pd.Period("2021Q4", freq="Q"): "12-31-2021.zip",
}

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


class LocalBankIdentityOverride(BaseModel):
    """`configs/local/*.yaml`'s own schema (gitignored -- see
    `configs/local/example_bank_deal.local.yaml.template`): REPLACES
    `ExampleDealConfig.acquirer`/`target` wholesale, for anyone adapting
    this example to their own (possibly sensitive) real deal without
    committing their own bank RSSD IDs."""

    model_config = ConfigDict(extra="forbid")
    acquirer: BankIdentity
    target: BankIdentity


def load_example_deal_config(
    path: Path, local_config_path: Path | None = None
) -> ExampleDealConfig:
    example = ExampleDealConfig.model_validate(yaml.safe_load(path.read_text()))
    if local_config_path is None:
        return example
    local = LocalBankIdentityOverride.model_validate(yaml.safe_load(local_config_path.read_text()))
    return example.model_copy(update={"acquirer": local.acquirer, "target": local.target})


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


def _load_calibration_call_report_frames(raw_dir: Path) -> dict[pd.Period, pd.DataFrame]:
    """One nationwide Call Report frame per 2020Q1-2021Q4 quarter found
    in `raw_dir` (missing quarters just aren't in the result -- see
    `CALIBRATION_QUARTER_FILENAMES`), read ONCE and shared across both
    banks (not re-parsed per bank)."""
    frames: dict[pd.Period, pd.DataFrame] = {}
    for quarter, filename in CALIBRATION_QUARTER_FILENAMES.items():
        path = raw_dir / filename
        if not path.exists():
            continue
        with open(path, "rb") as f:
            frames[quarter], _ = parse_bank_capital_zip(f.read(), quarter)
    return frames


def _resolve_bank_config(
    identity: BankIdentity,
    calibration_frames_by_quarter: dict[pd.Period, pd.DataFrame],
    rate_by_quarter: pd.Series,
) -> tuple[BankConfig, ppnr.CalibratedNimBeta | None]:
    """Calibrates this bank's own NIM beta (`corefin.bank.ppnr.
    calibrate_nim_beta`) from the 2020Q1-2021Q4 Call Report frames if
    available and estimable, falling back to the illustrative default
    otherwise. Returns (bank_config with ppnr_stress set, the
    calibration result or None)."""
    rows_by_quarter: dict[pd.Period, pd.Series] = {}
    for quarter, frame in calibration_frames_by_quarter.items():
        bank_rows = frame[frame["bank_id"] == identity.bank_id]
        if not bank_rows.empty:
            rows_by_quarter[quarter] = bank_rows.iloc[0]
    calibration = ppnr.calibrate_nim_beta(identity.bank_id, rows_by_quarter, rate_by_quarter)
    nim_beta = (
        calibration.nim_beta
        if calibration is not None and calibration.nim_beta is not None
        else DEFAULT_NIM_BETA
    )
    bank_config = BankConfig(ppnr_stress=PpnrStressConfig(nim_beta=nim_beta))
    return bank_config, calibration


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
    bank_config: BankConfig,
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
) -> tuple[BankModelResult, DataQualityFlags, float, float]:
    """Returns (result, flags, realized_total_nco_rate,
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
    jumpoff_rate_pp = float(macro_history.loc[QUARTER_Q4, ppnr.RATE_VARIABLE])
    projected_rate_path_pp = scenario[ppnr.RATE_VARIABLE].sort_index().to_numpy()
    result = build_deal_horizon_bank_result(
        opening,
        credit_projection,
        bank_config,
        deal_horizon_quarters=deal_horizon_quarters,
        jumpoff_rate_pp=jumpoff_rate_pp,
        projected_rate_path_pp=projected_rate_path_pp,
    )
    return result, flags, realized_total, projected_total


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, help="Path to an ExampleDealConfig YAML."
    ),
    local_config: Path | None = typer.Option(
        None,
        "--local-config",
        exists=True,
        dir_okay=False,
        help="Optional LOCAL, gitignored YAML (see configs/local/example_bank_deal.local."
        "yaml.template) that REPLACES --config's own acquirer/target RSSD IDs.",
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
    calibration_raw_dir: Path = typer.Option(
        Path("data/raw/ffiec"),
        "--calibration-raw-dir",
        help="Directory holding the 2020Q1-2021Q4 Call Report bulk ZIPs "
        "(CALIBRATION_QUARTER_FILENAMES) used to calibrate each bank's own NIM beta. A "
        "missing quarter falls back to the illustrative default for that bank, not an error.",
    ),
    monte_carlo_draws: int = typer.Option(
        DEFAULT_MONTE_CARLO_DRAWS, "--monte-carlo-draws", help="Stage 6 Monte Carlo draw count."
    ),
    monte_carlo_seed: int = typer.Option(
        0, "--monte-carlo-seed", help="Stage 6 Monte Carlo RNG seed (reproducible draws)."
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

    example = load_example_deal_config(config, local_config)

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

    calibration_frames_by_quarter = _load_calibration_call_report_frames(calibration_raw_dir)
    rate_by_quarter = macro_history[ppnr.RATE_VARIABLE]

    results: dict[str, BankModelResult] = {}
    bank_configs: dict[str, BankConfig] = {}
    for key, identity in (("acquirer", example.acquirer), ("target", example.target)):
        bank_config, calibration = _resolve_bank_config(
            identity, calibration_frames_by_quarter, rate_by_quarter
        )
        if calibration is not None and calibration.nim_beta is not None:
            typer.echo(
                f"{identity.label}: calibrated NIM beta = {calibration.nim_beta:+.3f} "
                f"(illustrative default: {DEFAULT_NIM_BETA:+.3f}) -- realized NIM "
                f"{calibration.start_nim:.4%} -> {calibration.end_nim:.4%} vs. realized rate "
                f"change {calibration.realized_rate_change_pp:+.2f}pp, "
                f"{calibration.start_quarter}-{calibration.end_quarter}. CAVEAT: calibrated on "
                "2020Q1-2021Q4, when margins also moved from the deposit surge and PPP loans, "
                "not rates alone -- likely overstates true rate sensitivity (see README)."
            )
        else:
            typer.echo(
                f"{identity.label}: NIM beta calibration unavailable -- using the illustrative "
                f"default ({DEFAULT_NIM_BETA:+.3f})"
            )

        result, flags, realized_total, projected_total = _build_bank_result(
            identity,
            bank_config,
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
            f"  realized NCO rate = {realized_total:.4%}, "
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

    typer.echo(f"\n=== Stage 5: Stress Test ({FED_COMPARISON_QUARTERS}Q DFAST window) ===")
    scenario_projections: dict[str, dict[str, interface.CreditLossProjection]] = {}
    scenario_rate_paths: dict[str, np.ndarray] = {}
    for scenario_name, scenario_df in (
        ("baseline", scenario),
        ("severely_adverse", severely_adverse_scenario),
    ):
        scenario_projections[scenario_name] = {
            key: _build_credit_projection(
                identity,
                scenario_name,
                scenario_df,
                training,
                macro_history,
                long_history_frames,
                categories,
                selected_family_by_category,
            )
            for key, identity in (("acquirer", example.acquirer), ("target", example.target))
        }
        jumpoff_rate_pp = float(macro_history.loc[QUARTER_Q4, ppnr.RATE_VARIABLE])
        projected_rate_path_pp = scenario_df[ppnr.RATE_VARIABLE].sort_index().to_numpy()
        scenario_rate_paths[scenario_name] = ppnr.align_rate_path_to_timeline(
            jumpoff_rate_pp, projected_rate_path_pp, FED_COMPARISON_QUARTERS + 1
        )

    stress_results: dict[str, StressTestResult] = {}
    for scenario_name in ("baseline", "severely_adverse"):
        rate_path = scenario_rate_paths[scenario_name]
        stress_results[scenario_name] = run_stress_test(
            results["acquirer"].opening,
            results["target"].opening,
            scenario_projections[scenario_name]["acquirer"],
            scenario_projections[scenario_name]["target"],
            bank_configs["acquirer"],
            bank_configs["target"],
            deal_result,
            example.deal,
            acquirer_unexplained_cet1_residual_mm=results[
                "acquirer"
            ].capital.unexplained_cet1_residual_mm,
            jumpoff_rate_pp=float(rate_path[0]),
            projected_rate_path_pp=rate_path[1:],
        )

    for scenario_name in ("baseline", "severely_adverse"):
        stress_result = stress_results[scenario_name]
        typer.echo(f"\n  --- {scenario_name} ---")
        for label, path in (
            ("Acquirer standalone", stress_result.acquirer_standalone),
            ("Pro forma combined", stress_result.pro_forma_combined),
        ):
            breach = "BREACHES" if path.breaches_4_5_pct_minimum else "does not breach"
            typer.echo(
                f"  {label}: starting CET1={path.starting_cet1_ratio:.4%}  "
                f"minimum CET1={path.minimum_cet1_ratio:.4%} ({breach} the 4.5% minimum)  "
                f"peak-to-trough={path.peak_to_trough_cet1_change_pp:.2f}pp  "
                f"illustrative SCB={path.illustrative_stress_capital_buffer:.4%}"
            )
            typer.echo(
                f"    cumulative {FED_COMPARISON_QUARTERS}Q: "
                f"PPNR=${path.cumulative_ppnr_mm:,.1f}mm  "
                f"provisions=${path.cumulative_provision_mm:,.1f}mm  "
                f"net income=${path.cumulative_net_income_mm:,.1f}mm"
            )
        typer.echo(
            f"  Deal impact: minimum stressed CET1 ratio changes by "
            f"{stress_result.minimum_cet1_ratio_change_pp:+.4%} (pro forma combined vs. "
            f"acquirer standalone)"
        )

    typer.echo("\n  --- 9-quarter cumulative loss rate vs. Fed DFAST published average ---")
    fed_total_loans_pct = DFAST_2026_SEVERELY_ADVERSE_LOSS_RATES[
        "total_loans"
    ].severely_adverse_9q_loss_rate_percent
    for key, identity in (("acquirer", example.acquirer), ("target", example.target)):
        severely_adverse_proj = scenario_projections["severely_adverse"][key]
        total_balance = severely_adverse_proj.balance_mm[:, 0].sum()
        blended_nco_rate = (
            severely_adverse_proj.nco_rate[:, 1 : FED_COMPARISON_QUARTERS + 1]
            * severely_adverse_proj.balance_mm[:, 1 : FED_COMPARISON_QUARTERS + 1]
        ).sum(axis=0) / total_balance
        our_cumulative_loss_rate_pct = float(np.sum(blended_nco_rate / 4.0)) * 100.0
        gap_pp = our_cumulative_loss_rate_pct - fed_total_loans_pct
        typer.echo(
            f"  {identity.label}: our {our_cumulative_loss_rate_pct:.2f}% vs. Fed DFAST "
            f"{fed_total_loans_pct:.2f}% (all loan types, published average) -- "
            f"gap {gap_pp:+.2f}pp"
        )

    typer.echo("\n=== Stage 6: Sensitivities & Monte Carlo ===")
    deal_horizon_rate_path = ppnr.align_rate_path_to_timeline(
        float(macro_history.loc[QUARTER_Q4, ppnr.RATE_VARIABLE]),
        scenario[ppnr.RATE_VARIABLE].sort_index().to_numpy(),
        example.deal.deal_horizon_quarters + 1,
    )
    ctx = DealContext(
        acquirer_opening=results["acquirer"].opening,
        target_opening=results["target"].opening,
        acquirer_bank_config=bank_configs["acquirer"],
        target_bank_config=bank_configs["target"],
        acquirer_baseline_result=results["acquirer"],
        target_baseline_result=results["target"],
        acquirer_baseline_projection=extend_credit_loss_projection(
            scenario_projections["baseline"]["acquirer"], example.deal.deal_horizon_quarters + 1
        ),
        target_baseline_projection=extend_credit_loss_projection(
            scenario_projections["baseline"]["target"], example.deal.deal_horizon_quarters + 1
        ),
        acquirer_severely_adverse_projection=truncate_credit_loss_projection(
            scenario_projections["severely_adverse"]["acquirer"], FED_COMPARISON_QUARTERS + 1
        ),
        target_severely_adverse_projection=truncate_credit_loss_projection(
            scenario_projections["severely_adverse"]["target"], FED_COMPARISON_QUARTERS + 1
        ),
        baseline_rate_path_pp=deal_horizon_rate_path,
        severely_adverse_rate_path_pp=scenario_rate_paths["severely_adverse"],
        base_config=example.deal,
        acquirer_unexplained_cet1_residual_mm=results[
            "acquirer"
        ].capital.unexplained_cet1_residual_mm,
    )

    typer.echo("\n  --- Tornado (base case vs. low/high, holding everything else fixed) ---")
    tornado = compute_tornado(ctx)
    base = tornado.base_metrics
    base_accretion_str = (
        f"{base.year2_accretion_pct_excl_one_time:+.2%}"
        if base.year2_accretion_pct_excl_one_time is not None
        else "n.m."
    )
    base_earnback_str = base.earnback_years if base.earnback_years is not None else "beyond horizon"
    typer.echo(
        f"  Base case: Year-2 accretion (excl. one-time)={base_accretion_str}  "
        f"TBV dilution={base.tbv_dilution_at_close_pct:+.2%}  earnback={base_earnback_str}  "
        f"min stressed CET1={base.minimum_stressed_cet1_ratio:.4%}"
    )
    for metric_name, metric_attr, fmt, not_meaningful_label in (
        (
            "Year-2 accretion (excl. one-time)",
            "year2_accretion_pct_excl_one_time",
            "{:+.2%}",
            "n.m.",
        ),
        ("TBV dilution at close", "tbv_dilution_at_close_pct", "{:+.2%}", "n.m."),
        ("Earnback (years)", "earnback_years", "{:.2f}", "beyond horizon"),
        ("Minimum stressed CET1", "minimum_stressed_cet1_ratio", "{:.4%}", "n.m."),
    ):
        typer.echo(f"\n  {metric_name}:")
        for row in tornado.sorted_by(metric_attr):
            low_val = getattr(row.low_metrics, metric_attr)
            high_val = getattr(row.high_metrics, metric_attr)
            low_str = fmt.format(low_val) if low_val is not None else not_meaningful_label
            high_str = fmt.format(high_val) if high_val is not None else not_meaningful_label
            typer.echo(
                f"    {row.driver_name:<28} [{row.low_value:.3f}, {row.high_value:.3f}] -> "
                f"{low_str} .. {high_str}"
            )

    typer.echo("\n  --- Grid: Year-2 accretion (excl. one-time) & earnback, price x cost saves ---")
    price_values = np.array(
        [
            max(example.deal.consideration.price_to_tbv - 0.3, 0.1),
            example.deal.consideration.price_to_tbv,
            example.deal.consideration.price_to_tbv + 0.3,
        ]
    )
    cost_save_values = np.array(
        [
            max(example.deal.cost_saves.cost_save_pct_of_target_noninterest_expense - 0.10, 0.0),
            example.deal.cost_saves.cost_save_pct_of_target_noninterest_expense,
            min(example.deal.cost_saves.cost_save_pct_of_target_noninterest_expense + 0.10, 1.0),
        ]
    )
    grid = compute_price_cost_save_grid(ctx, price_values, cost_save_values)
    header = "".join(f"{p:>10.2f}x" for p in price_values)
    typer.echo(f"  accretion{'':<4}{header}")
    for i, cs in enumerate(cost_save_values):
        row = "".join(
            f"{v:>10.1%}" if np.isfinite(v) else f"{'n.m.':>10}"
            for v in grid.year2_accretion_pct[i]
        )
        typer.echo(f"  cost save={cs:.0%}  {row}")
    typer.echo(f"  earnback (yrs){'':<0}{header}")
    for i, cs in enumerate(cost_save_values):
        row = "".join(
            f"{v:>10.2f}" if np.isfinite(v) else f"{'beyond':>10}" for v in grid.earnback_years[i]
        )
        typer.echo(f"  cost save={cs:.0%}  {row}")

    typer.echo("\n  --- Monte Carlo (credit mark, cost saves, rate mark, NIM beta) ---")
    mc = run_monte_carlo(ctx, n_draws=monte_carlo_draws, seed=monte_carlo_seed)
    typer.echo(f"  {mc.n_draws} draws, seed={mc.seed}")
    for metric_name, metric_attr, fmt in (
        ("Year-2 accretion (excl. one-time)", "year2_accretion_pct", "{:+.2%}"),
        ("Earnback (years)", "earnback_years", "{:.2f}"),
        ("Minimum stressed CET1", "minimum_stressed_cet1_ratio", "{:.4%}"),
    ):
        pct = mc.percentiles(metric_attr, percentiles=(5, 25, 50, 75, 95))
        pct_str = "  ".join(f"p{p}={fmt.format(v)}" for p, v in pct.items())
        typer.echo(f"    {metric_name}: {pct_str}")
    typer.echo(f"    P(TBV earnback beyond horizon) = {mc.probability_beyond_horizon():.1%}")


if __name__ == "__main__":
    app()
