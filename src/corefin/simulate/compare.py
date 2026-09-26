"""Structure comparison: runs the Monte Carlo engine, downside analytics
and named stress scenarios on multiple structures with the SAME random
draws (common random numbers -- financing structure never affects
operating performance, so differences across structures reflect the
structure, not fresh sampling noise; this falls out for free by calling
`run_simulation` with the same seed and n_scenarios for every structure,
exactly as the optimizer's grid search already relies on), so returns and
risk can be compared side by side -- e.g. the input config's own
structure vs. the optimizer's recommendation vs. a more levered variant.

Reuses `optimize/pricing.build_priced_structure` (the optimizer's own
structure builder) rather than reimplementing it.
"""

from __future__ import annotations

from dataclasses import dataclass

from corefin.assumptions.schema import RootConfig
from corefin.optimize.pricing import build_priced_structure
from corefin.optimize.structure import CandidateStructure, compute_entry_ebitda_mm
from corefin.simulate.analytics import DownsideAnalytics, compute_downside_analytics
from corefin.simulate.engine import SimulationResult, run_simulation
from corefin.simulate.stress import StressScenarioResult, run_stress_scenarios
from corefin.timeline import Timeline


def input_config_decision_values(root_config: RootConfig) -> dict[str, float]:
    """The decision-variable tranches' actual as-configured sizes,
    expressed as multiples of entry EBITDA -- "the structure already in
    the config," for comparison against the optimizer's recommendation."""
    if root_config.optimizer is None:
        raise ValueError("root_config.optimizer must be set to derive input-config decision values")
    entry_ebitda_mm = compute_entry_ebitda_mm(root_config)
    tranche_by_name = {t.name: t for t in root_config.tranches}
    return {
        dv.tranche_name: tranche_by_name[dv.tranche_name].size_mm / entry_ebitda_mm
        for dv in root_config.optimizer.decision_variables
    }


def bump_decision_values(
    decision_values: dict[str, float], tranche_name: str, delta: float
) -> dict[str, float]:
    """A copy of `decision_values` with `tranche_name` shifted by `delta`
    turns of entry EBITDA -- e.g. a "+1.0x TLB" more-levered variant of the
    optimizer's recommendation. Not clipped to any configured decision-
    variable bound: the whole point is to explore beyond what the
    optimizer's search grid would have tried."""
    if tranche_name not in decision_values:
        raise ValueError(
            f"'{tranche_name}' is not one of the decision variables: {list(decision_values)}"
        )
    bumped = dict(decision_values)
    bumped[tranche_name] += delta
    return bumped


@dataclass(frozen=True)
class StructureComparisonEntry:
    name: str
    candidate: CandidateStructure
    simulation: SimulationResult
    downside: DownsideAnalytics
    stress_results: list[StressScenarioResult]


def compare_structures(
    root_config: RootConfig,
    timeline: Timeline,
    named_decision_values: dict[str, dict[str, float]],
    n_scenarios: int,
    seed: int | None = None,
    irr_hurdle: float = 0.15,
) -> list[StructureComparisonEntry]:
    stress_configs = (
        root_config.simulate.stress_scenarios if root_config.simulate is not None else []
    )
    entries = []
    for name, decision_values in named_decision_values.items():
        candidate = build_priced_structure(root_config, decision_values)
        simulation = run_simulation(root_config, candidate, timeline, n_scenarios, seed=seed)
        downside = compute_downside_analytics(simulation, irr_hurdle)
        stress_results = run_stress_scenarios(root_config, candidate, timeline, stress_configs)
        entries.append(
            StructureComparisonEntry(
                name=name,
                candidate=candidate,
                simulation=simulation,
                downside=downside,
                stress_results=stress_results,
            )
        )
    return entries
