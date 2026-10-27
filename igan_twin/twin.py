"""Individual digital twins: personalise the model to one patient and forecast.

A twin starts from the patient's baseline covariates (labs, biopsy, complement
biomarkers). If longitudinal measurements under a known treatment history are
available, the five individual random effects (``ETA_NAMES``) are estimated by
maximum a-posteriori fitting with a standard-normal prior; the Laplace
(Gauss-Newton) covariance gives a posterior used to propagate uncertainty
into counterfactual treatment forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .model import SimResult, simulate
from .parameters import ModelParams
from .pk import Regimen
from .population import ETA_NAMES, Patient, PatientParams, patient_params

# Residual error model (intra-individual + assay variability)
SIGMA = {"upcr": 0.25, "gd": 0.15, "egfr": 4.0}  # log-SD, log-SD, mL/min


@dataclass
class Observation:
    day: float
    upcr: float | None = None  # g/g
    egfr: float | None = None  # mL/min/1.73m2
    gd_rel: float | None = None  # serum Gd-IgA1 relative to this patient's baseline


@dataclass
class TwinFit:
    patient: Patient
    params: ModelParams
    history: list[Regimen]
    observations: list[Observation]
    eta: np.ndarray
    cov: np.ndarray
    success: bool = True
    info: dict = field(default_factory=dict)

    def eta_table(self) -> dict:
        sd = np.sqrt(np.diag(self.cov))
        return {k: (float(m), float(s)) for k, m, s in zip(ETA_NAMES, self.eta, sd)}

    def mechanism(self) -> dict:
        """Patient-level mechanism readout at the MAP estimate."""
        pp = patient_params(self.patient.covariates(), self.params, self.eta[None, :])
        return {
            "APRIL/BAFF-independent Gd-IgA1 fraction": float(pp.f_ind_gd[0]),
            "complement share of glomerular injury": float(1 - pp.w_direct[0]),
            "terminal-pathway (C5) share of injury": float(pp.w_c5[0]),
            "lectin-pathway share of initiation": float(pp.f_lectin[0]),
            "AP amplification loop gain": float(pp.loop_gain[0]),
            "reversible share of proteinuria": float(pp.phi[0]),
            "deposit clearance half-life (days)": float(np.log(2) / pp.kD[0]),
            "progression multiplier": float(pp.prog[0]),
        }


def _simulate_etas(patient: Patient, params: ModelParams, etas: np.ndarray, arms: list[list[Regimen]],
                   arm_of_row=None, t_end: float = 365.0, record_every: float = 1.0) -> SimResult:
    etas = np.atleast_2d(etas)
    cov = patient.covariates(len(etas))
    pp = patient_params(cov, params, etas)
    if arm_of_row is not None and len(arm_of_row) != pp.n:
        raise ValueError("arm_of_row length mismatch")
    return simulate(params, pp, arms, arm_of_row, t_end=t_end, record_every=record_every)


def _predict_obs(sim: SimResult, obs: list[Observation], row_slice=slice(None)):
    out = []
    for o in obs:
        i = int(np.argmin(np.abs(sim.t - o.day)))
        if o.upcr is not None:
            out.append(("upcr", np.log(sim.out["UPCR"][i, row_slice]), np.log(o.upcr)))
        if o.egfr is not None:
            out.append(("egfr", sim.out["eGFR"][i, row_slice], o.egfr))
        if o.gd_rel is not None:
            out.append(("gd", np.log(sim.out["GD"][i, row_slice]), np.log(o.gd_rel)))
    return out


def fit_twin(patient: Patient, history: list[Regimen], observations: list[Observation],
             params: ModelParams, prior_sd: float = 1.0) -> TwinFit:
    """MAP estimate of the patient's random effects from longitudinal data.

    Baseline UPCR and eGFR are taken from ``patient`` (the model reproduces
    them exactly at day 0); observations are later measurements.
    """
    k = len(ETA_NAMES)
    obs = [o for o in observations if o.day > 0]
    if not obs:
        return TwinFit(patient, params, history, observations, np.zeros(k), np.eye(k) * prior_sd**2, info={"note": "prior only"})
    t_end = max(o.day for o in obs)
    h = 1e-3

    def residuals_for(etas):
        sim = _simulate_etas(patient, params, etas, [history], t_end=t_end)
        res = []
        for kind, pred, y in _predict_obs(sim, obs):
            res.append((pred - y) / SIGMA[kind])
        res = np.array(res)  # (n_obs, n_rows)
        prior = (etas / prior_sd).T  # (k, n_rows)
        return np.vstack([res, prior])

    def fun(x):
        return residuals_for(x[None, :])[:, 0]

    def jac(x):
        etas = np.vstack([x, x + h * np.eye(k)])  # one vectorised run for all perturbations
        r = residuals_for(etas)
        return (r[:, 1:] - r[:, :1]) / h

    sol = least_squares(fun, np.zeros(k), jac=jac, bounds=(-4, 4), method="trf")
    J = sol.jac
    cov = np.linalg.pinv(J.T @ J)
    return TwinFit(patient, params, history, observations, sol.x, cov, bool(sol.success),
                   {"cost": float(sol.cost), "nfev": int(sol.nfev)})


@dataclass
class Forecast:
    scenarios: dict[str, list[Regimen]]
    sim: SimResult
    n_samples: int

    def bands(self, scenario: str, name: str = "eGFR", q=(0.05, 0.5, 0.95)) -> np.ndarray:
        a = list(self.scenarios).index(scenario)
        return self.sim.quantiles(name, a, q)

    def kidney_failure_prob(self, scenario: str, threshold: float = 15.0) -> np.ndarray:
        """Cumulative probability (over posterior samples) that eGFR < threshold, by time."""
        a = list(self.scenarios).index(scenario)
        egfr = self.sim.out["eGFR"][:, self.sim.rows(a)]
        return np.mean(np.minimum.accumulate(egfr, axis=0) < threshold, axis=1)

    def summary(self, years=(1, 5, 10)) -> dict:
        out = {}
        for s in self.scenarios:
            a = list(self.scenarios).index(s)
            row = {}
            for y in years:
                d = 365 * y
                if d > self.sim.t[-1]:
                    continue
                row[f"UPCR_{y}y_median"] = float(np.median(self.sim.at("UPCR", d)[self.sim.rows(a)]))
                row[f"eGFR_{y}y_median"] = float(np.median(self.sim.at("eGFR", d)[self.sim.rows(a)]))
                i = int(np.argmin(np.abs(self.sim.t - d)))
                row[f"P_kidney_failure_{y}y"] = float(self.kidney_failure_prob(s)[i])
            out[s] = row
        return out


def forecast(fit: TwinFit, scenarios: dict[str, list[Regimen]], years: float = 10.0,
             n_samples: int = 200, seed: int = 0, record_every: float = 14.0) -> Forecast:
    """Simulate posterior samples of the twin under each treatment scenario.

    Each scenario's regimen list should include the treatment already given
    (``fit.history``) if it continues; use ``start_day`` / ``stop_day`` to
    express switches.
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(fit.cov + 1e-9 * np.eye(len(fit.eta)))
    etas = fit.eta + rng.standard_normal((n_samples, len(fit.eta))) @ L.T
    etas = np.clip(etas, -4, 4)
    arms = list(scenarios.values())
    all_etas = np.vstack([etas] * len(arms))
    arm_of_row = np.repeat(np.arange(len(arms)), n_samples)
    sim = _simulate_etas(fit.patient, fit.params, all_etas, arms, arm_of_row, t_end=365 * years, record_every=record_every)
    return Forecast(scenarios, sim, n_samples)


def synthetic_observations(patient: Patient, params: ModelParams, eta_true: np.ndarray, history: list[Regimen],
                           days, seed: int = 0) -> list[Observation]:
    """Noisy measurements from a 'ground-truth' patient, for validating twin fitting."""
    rng = np.random.default_rng(seed)
    sim = _simulate_etas(patient, params, eta_true[None, :], [history], t_end=max(days))
    obs = []
    for d in days:
        i = int(np.argmin(np.abs(sim.t - d)))
        obs.append(Observation(
            day=d,
            upcr=float(sim.out["UPCR"][i, 0] * np.exp(rng.normal(0, SIGMA["upcr"]))),
            egfr=float(sim.out["eGFR"][i, 0] + rng.normal(0, SIGMA["egfr"])),
            gd_rel=float(sim.out["GD"][i, 0] * np.exp(rng.normal(0, SIGMA["gd"]))),
        ))
    return obs
