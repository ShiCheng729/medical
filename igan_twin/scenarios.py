"""Long-term virtual-population projections and precision-medicine analyses."""

from __future__ import annotations

from dataclasses import dataclass

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
)
from .population import Covariates, PatientParams, PopulationSpec, patient_params, virtual_population

# Treatment strategies on top of optimised supportive care (RASi +/- SGLT2i).
STRATEGIES: dict[str, list[Regimen]] = {
    "Supportive care": [],
    "Sibeprenlimab 400 mg Q4W": [SIBEPRENLIMAB_SC_400],
    "Atacicept 150 mg QW": [ATACICEPT_150_QW],
    "Iptacopan 200 mg BID": [IPTACOPAN_200_BID],
    "Ravulizumab Q8W": [RAVULIZUMAB_IV],
    "Narsoplimab 12-wk course": [NARSOPLIMAB_IV_12W],
    "Sibeprenlimab + iptacopan": [SIBEPRENLIMAB_SC_400, IPTACOPAN_200_BID],
}

HIGH_RISK = PopulationSpec(egfr_mean=60, egfr_sd=20, upcr_gm=1.6, upcr_gsd=1.6, upcr_min=1.0)


@dataclass
class Projection:
    strategies: dict[str, list[Regimen]]
    sim: SimResult
    cov: Covariates
    pp: PatientParams
    n: int

    def arm(self, name: str) -> int:
        return list(self.strategies).index(name)

    def kf_free(self, name: str, threshold: float = 15.0) -> np.ndarray:
        """Kidney-failure-free proportion (eGFR >= threshold) over time."""
        e = self.sim.out["eGFR"][:, self.sim.rows(self.arm(name))]
        return 1.0 - np.mean(np.minimum.accumulate(e, axis=0) < threshold, axis=1)

    def table(self, months: float = 9.0) -> list[dict]:
        d = 30.44 * months
        out = []
        base = self.arm("Supportive care")
        for s in self.strategies:
            a = self.arm(s)
            ratio = self.sim.gm_ratio(a, d) / self.sim.gm_ratio(base, d)
            slope = (self.sim.at("eGFR", 730)[self.sim.rows(a)] - self.sim.at("eGFR", 365)[self.sim.rows(a)])
            out.append({
                "strategy": s,
                f"UPCR reduction vs SoC at {months:g} mo": 1 - ratio,
                "Gd-IgA1 vs baseline": self.sim.gm_ratio(a, d, "GD"),
                "mesangial deposits vs baseline": self.sim.gm_ratio(a, d, "D"),
                "C3 deposition vs baseline": self.sim.gm_ratio(a, d, "C3"),
                "C5b-9 vs baseline": self.sim.gm_ratio(a, d, "MAC"),
                "eGFR slope yr1-2 (mL/min/yr)": float(np.mean(slope)),
                "kidney failure-free at 10 y": float(self.kf_free(s)[-1]) if self.sim.t[-1] >= 3650 else float("nan"),
            })
        return out


def project(params: ModelParams, strategies=None, n: int = 300, years: float = 10.0, seed: int = 11,
            spec: PopulationSpec = HIGH_RISK) -> Projection:
    strategies = strategies or STRATEGIES
    cov, eta, pp = virtual_population(spec, n, params, seed)
    k = len(strategies)
    rows = PatientParams.concat([pp] * k)
    arm_of_row = np.repeat(np.arange(k), n)
    sim = simulate(params, rows, list(strategies.values()), arm_of_row, t_end=365 * years, record_every=14)
    return Projection(strategies, sim, cov, pp, n)


def responder_analysis(params: ModelParams, n: int = 400, seed: int = 5, day: float = 274):
    """Per-patient 9-month UPCR reduction on anti-APRIL vs factor-B inhibition."""
    strategies = {
        "Supportive care": [],
        "Sibeprenlimab 400 mg Q4W": [SIBEPRENLIMAB_SC_400],
        "Iptacopan 200 mg BID": [IPTACOPAN_200_BID],
    }
    cov, eta, pp = virtual_population(HIGH_RISK, n, params, seed)
    rows = PatientParams.concat([pp] * 3)
    sim = simulate(params, rows, list(strategies.values()), np.repeat(np.arange(3), n), t_end=day, record_every=day)
    u = sim.out["UPCR"][-1].reshape(3, n)
    red_sib = 1 - u[1] / u[0]
    red_ipt = 1 - u[2] / u[0]
    return {
        "cov": cov,
        "pp": pp,
        "reduction_sib": red_sib,
        "reduction_ipt": red_ipt,
        "complement_share": 1 - pp.w_direct,
        "april_independent": pp.f_ind_gd,
    }
