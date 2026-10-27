"""Virtual replicas of APRIL- and complement-targeting IgAN trials.

Each :class:`Trial` holds the design (arms, duration, approximate baseline
population) and the published read-outs used as calibration targets.

UPCR targets are expressed as the *ratio of geometric-mean change vs placebo*
at the primary time point, i.e. ``(1 + change_drug) / (1 + change_placebo)``,
which is how "placebo-adjusted % reduction" is reported (reduction = 1 - ratio).

Published numbers were checked against the sources in ``refs``; baseline
population summaries are approximate (rounded from the publications'
baseline tables / inclusion criteria) and should be refined if you have the
primary tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import SimResult, simulate
from .parameters import ModelParams
from .pk import (
    ATACICEPT_150_QW,
    IPTACOPAN_200_BID,
    NARSOPLIMAB_IV_12W,
    RAVULIZUMAB_IV,
    SIBEPRENLIMAB_SC_400,
    Regimen,
    sibeprenlimab_iv,
)
from .population import PatientParams, PopulationSpec, patient_params, virtual_population


@dataclass(frozen=True)
class Target:
    kind: str  # "upcr_ratio" | "egfr_change" | "biomarker_ratio"
    arm: str
    day: float
    value: float
    sd: float  # residual scale (log-scale for ratios, mL/min for eGFR)
    note: str = ""
    marker: str = "UPCR"


@dataclass
class Trial:
    name: str
    drug_class: str
    arms: dict[str, list[Regimen]]
    duration: float
    primary_day: float
    placebo_change: float  # reported geometric-mean UPCR change in the placebo arm
    population: PopulationSpec
    targets: list[Target]
    refs: list[str] = field(default_factory=list)
    calibrate: bool = True

    @property
    def arm_names(self) -> list[str]:
        return list(self.arms)


def _r(drug_change: float, placebo_change: float) -> float:
    return (1.0 + drug_change) / (1.0 + placebo_change)


TRIALS: dict[str, Trial] = {
    "ENVISION": Trial(
        name="ENVISION (phase 2, sibeprenlimab IV Q4W)",
        drug_class="anti-APRIL",
        arms={
            "placebo": [],
            "sib 2 mg/kg": [sibeprenlimab_iv(2)],
            "sib 4 mg/kg": [sibeprenlimab_iv(4)],
            "sib 8 mg/kg": [sibeprenlimab_iv(8)],
        },
        duration=365,
        primary_day=365,
        placebo_change=-0.20,
        population=PopulationSpec(egfr_mean=63, upcr_gm=1.3, upcr_min=0.5),
        targets=[
            Target("upcr_ratio", "sib 2 mg/kg", 365, _r(-0.472, -0.20), 0.08, "GM change -47.2% vs -20.0%"),
            Target("upcr_ratio", "sib 4 mg/kg", 365, _r(-0.588, -0.20), 0.08, "-58.8% vs -20.0%"),
            Target("upcr_ratio", "sib 8 mg/kg", 365, _r(-0.620, -0.20), 0.08, "-62.0% vs -20.0%"),
            Target("egfr_change", "placebo", 365, -7.4, 1.8, "LS mean eGFR change"),
            Target("egfr_change", "sib 2 mg/kg", 365, -2.7, 1.8),
            Target("egfr_change", "sib 4 mg/kg", 365, 0.2, 1.7),
            Target("egfr_change", "sib 8 mg/kg", 365, -1.5, 1.8),
            Target(
                "biomarker_ratio", "sib 8 mg/kg", 365, 0.35, 0.25,
                "serum IgA/Gd-IgA1 ~-65% (secondary summary; approximate)", marker="GD",
            ),
        ],
        refs=[
            "Mathur M et al. A Phase 2 Trial of Sibeprenlimab in Patients with IgA Nephropathy. NEJM 2024;390:20-31. PMID 37916620",
        ],
    ),
    "VISIONARY": Trial(
        name="VISIONARY (phase 3, sibeprenlimab 400 mg SC Q4W)",
        drug_class="anti-APRIL",
        arms={"placebo": [], "sibeprenlimab 400 mg SC": [SIBEPRENLIMAB_SC_400]},
        duration=365,
        primary_day=274,
        placebo_change=+0.021,
        population=PopulationSpec(egfr_mean=63, upcr_gm=1.3, upcr_min=0.5),
        targets=[
            Target("upcr_ratio", "sibeprenlimab 400 mg SC", 274, _r(-0.502, 0.021), 0.06, "interim: -50.2% vs +2.1% (51.2% adj.)"),
            Target("upcr_ratio", "sibeprenlimab 400 mg SC", 365, _r(-0.566, -0.051), 0.08, "12 mo: -56.6% vs -5.1%"),
        ],
        refs=[
            "Otsuka press release / ERA 2025 late-breaker, 6 Jun 2025 (prespecified interim, n=320)",
            "UKKW 2026 report (12-month data), EMJ Nephrology",
        ],
    ),
    "ORIGIN3": Trial(
        name="ORIGIN 3 (phase 3, atacicept 150 mg SC QW)",
        drug_class="APRIL+BAFF",
        arms={"placebo": [], "atacicept 150 mg QW": [ATACICEPT_150_QW]},
        duration=252,
        primary_day=252,
        placebo_change=-0.068,
        population=PopulationSpec(egfr_mean=63, upcr_gm=1.5),
        targets=[
            Target("upcr_ratio", "atacicept 150 mg QW", 252, _r(-0.457, -0.068), 0.06, "-45.7% vs -6.8% at week 36"),
            Target(
                "biomarker_ratio", "atacicept 150 mg QW", 252, 0.40, 0.25,
                "Gd-IgA1 ~-60% (phase 2b ORIGIN; approximate)", marker="GD",
            ),
        ],
        refs=["Lafayette R et al. A Phase 3 Trial of Atacicept in Patients with IgA Nephropathy. NEJM 2025 (interim, n=203)"],
    ),
    "APPLAUSE": Trial(
        name="APPLAUSE-IgAN (phase 3, iptacopan 200 mg BID)",
        drug_class="factor B",
        arms={"placebo": [], "iptacopan 200 mg BID": [IPTACOPAN_200_BID]},
        duration=274,
        primary_day=274,
        placebo_change=-0.09,
        population=PopulationSpec(egfr_mean=64, upcr_gm=1.9, upcr_min=1.0),
        targets=[Target("upcr_ratio", "iptacopan 200 mg BID", 274, 1 - 0.383, 0.06, "38.3% placebo-adjusted at month 9")],
        refs=["Perkovic V et al. APPLAUSE-IgAN interim analysis, NEJM 2025; Novartis release 15 Apr 2024"],
    ),
    "SANCTUARY": Trial(
        name="SANCTUARY (phase 2, ravulizumab IV)",
        drug_class="C5",
        arms={"placebo": [], "ravulizumab": [RAVULIZUMAB_IV]},
        duration=182,
        primary_day=182,
        placebo_change=-0.168,
        population=PopulationSpec(egfr_mean=70, upcr_gm=2.0, upcr_min=1.0),
        targets=[
            Target("upcr_ratio", "ravulizumab", 182, _r(-0.419, -0.168), 0.10, "-41.9% vs -16.8% at week 26"),
            Target("egfr_change", "ravulizumab", 182, 0.2, 2.5),
            Target("egfr_change", "placebo", 182, -4.5, 3.0),
        ],
        refs=["Lafayette R et al. Efficacy and Safety of Ravulizumab in IgA Nephropathy. JASN 2025. PMID 39455063"],
    ),
    "ARTEMIS": Trial(
        name="ARTEMIS-IGAN (phase 3, narsoplimab IV QW x12)",
        drug_class="MASP-2",
        arms={"placebo": [], "narsoplimab": [NARSOPLIMAB_IV_12W]},
        duration=252,
        primary_day=252,
        placebo_change=-0.30,  # "substantially greater than expected" — exact value not public
        population=PopulationSpec(egfr_mean=60, upcr_gm=2.5, upcr_min=1.3),
        targets=[
            Target(
                "upcr_ratio", "narsoplimab", 252, 0.92, 0.12,
                "no significant difference vs placebo (interim, trial stopped Oct 2023)",
            )
        ],
        refs=["Omeros press release 16 Oct 2023; EU CTR 2018-000075-33 results"],
    ),
}


def build_trial_rows(trial: Trial, params: ModelParams, n: int, seed: int):
    """Same virtual patients replicated in every arm (common random numbers)."""
    cov, eta, _ = virtual_population(trial.population, n, params, seed)
    pp = patient_params(cov, params, eta)
    k = len(trial.arms)
    rows = PatientParams.concat([pp] * k)
    arm_of_row = np.repeat(np.arange(k), n)
    return rows, arm_of_row, cov, eta


@dataclass
class TrialResult:
    trial: Trial
    sim: SimResult
    placebo_factor: float

    def arm(self, name: str) -> int:
        return self.trial.arm_names.index(name)

    def upcr_ratio_vs_placebo(self, arm: str, day: float) -> float:
        return self.sim.gm_ratio(self.arm(arm), day) / self.sim.gm_ratio(self.arm("placebo"), day)

    def predicted(self, tgt: Target) -> float:
        a = self.arm(tgt.arm)
        if tgt.kind == "upcr_ratio":
            return self.upcr_ratio_vs_placebo(tgt.arm, tgt.day)
        if tgt.kind == "egfr_change":
            return self.sim.mean_change(a, tgt.day, "eGFR")
        if tgt.kind == "biomarker_ratio":
            return self.sim.gm_ratio(a, tgt.day, tgt.marker)
        raise ValueError(tgt.kind)


def run_trial(trial: Trial, params: ModelParams, n: int = 150, seed: int = 1, record_every: float = 7.0) -> TrialResult:
    """Simulate a trial; the trial-level placebo response is set so the virtual
    placebo arm reproduces the reported placebo UPCR change at the primary time."""
    rows, arm_of_row, _, _ = build_trial_rows(trial, params, n, seed)
    arms = list(trial.arms.values())
    pbo = trial.arm_names.index("placebo")
    sel = np.flatnonzero(arm_of_row == pbo)
    natural = simulate(params, rows.take(sel), [[]], None, trial.primary_day, record_every=trial.primary_day)
    r0 = natural.gm_ratio(0, trial.primary_day)
    pl_end = (1.0 + trial.placebo_change) / r0
    sim = simulate(
        params, rows, arms, arm_of_row, trial.duration, record_every=record_every,
        placebo_end=pl_end, placebo_T=trial.primary_day,
    )
    return TrialResult(trial, sim, pl_end)
