"""Patients, virtual populations and the covariate -> mechanism mapping.

A *digital twin* here is one row of patient-level parameters. Those parameters
come from three layers:

1. population typical values (:class:`~igan_twin.parameters.ModelParams`),
2. covariate effects linking measurable biology (serum APRIL, Gd-IgA1,
   FHR-5/FH ratio, mesangial C3, glomerular C4d, Oxford MEST-C) to mechanism,
3. individual random effects ``eta`` (standard-normal), sampled for virtual
   patients or estimated from a real patient's longitudinal data
   (:mod:`igan_twin.twin`).

Covariate effect sizes are mechanistic hypotheses drawn from the IgAN
literature on direction of effect (e.g. FHR-5 competes with factor H and
raises alternative-pathway activity; C4d positivity marks lectin-pathway
activation; tubular atrophy T1/T2 marks irreversible damage); their
magnitudes are not estimated from individual-level data and should be treated
as priors.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from .parameters import LN2, ModelParams

ETA_NAMES = ("f_ind_gd", "w_comp", "phi", "kD", "prog")


def logit(p):
    return np.log(p / (1.0 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Covariates:
    """Baseline characteristics, one array entry per patient."""

    egfr0: np.ndarray  # mL/min/1.73m2
    upcr0: np.ndarray  # g/g (24 h UPCR)
    weight: np.ndarray  # kg
    age: np.ndarray
    april_rel: np.ndarray  # serum APRIL relative to IgAN median
    gd_rel: np.ndarray  # serum Gd-IgA1 relative to IgAN median
    fhr5_rel: np.ndarray  # FHR-5 / factor H ratio relative to IgAN median
    c3_int: np.ndarray  # mesangial C3 immunofluorescence 0-3
    c4d: np.ndarray  # glomerular C4d positive (0/1)
    M: np.ndarray
    E: np.ndarray
    S: np.ndarray
    T: np.ndarray  # 0/1/2
    C: np.ndarray  # 0/1/2

    @property
    def n(self) -> int:
        return len(self.egfr0)

    def take(self, idx) -> "Covariates":
        return Covariates(**{f.name: np.asarray(getattr(self, f.name))[idx] for f in fields(self)})

    @staticmethod
    def concat(items: list["Covariates"]) -> "Covariates":
        return Covariates(
            **{f.name: np.concatenate([np.atleast_1d(getattr(c, f.name)) for c in items]) for f in fields(Covariates)}
        )


@dataclass
class Patient:
    """A single real (or hypothetical) patient's baseline data."""

    egfr0: float
    upcr0: float
    weight: float = 75.0
    age: float = 40.0
    april_rel: float = 1.0
    gd_rel: float = 1.0
    fhr5_rel: float = 1.0
    c3_int: float = 2.0
    c4d: int = 0
    M: int = 1
    E: int = 0
    S: int = 1
    T: int = 0
    C: int = 0
    name: str = "patient"

    def covariates(self, n: int = 1) -> Covariates:
        vals = {f.name: np.full(n, float(getattr(self, f.name))) for f in fields(Covariates)}
        return Covariates(**vals)


@dataclass(frozen=True)
class PopulationSpec:
    """Baseline distribution of a trial population (approximate published values)."""

    egfr_mean: float = 63.0
    egfr_sd: float = 22.0
    egfr_min: float = 30.0
    upcr_gm: float = 1.4
    upcr_gsd: float = 1.7
    upcr_min: float = 0.75
    weight_mean: float = 75.0
    weight_sd: float = 15.0
    age_mean: float = 42.0
    prev: dict = field(
        default_factory=lambda: {"M": 0.45, "E": 0.2, "S": 0.7, "T1": 0.25, "T2": 0.05, "C1": 0.25, "C2": 0.03, "c4d": 0.3}
    )


def sample_covariates(spec: PopulationSpec, n: int, rng: np.random.Generator) -> Covariates:
    egfr = np.clip(rng.normal(spec.egfr_mean, spec.egfr_sd, n), spec.egfr_min, 130)
    upcr = np.maximum(spec.upcr_gm * np.exp(rng.normal(0, np.log(spec.upcr_gsd), n)), spec.upcr_min)
    p = spec.prev
    u = rng.random(n)
    T = np.where(u < p["T2"], 2, np.where(u < p["T2"] + p["T1"], 1, 0))
    u = rng.random(n)
    C = np.where(u < p["C2"], 2, np.where(u < p["C2"] + p["C1"], 1, 0))
    return Covariates(
        egfr0=egfr,
        upcr0=upcr,
        weight=np.clip(rng.normal(spec.weight_mean, spec.weight_sd, n), 45, 140),
        age=np.clip(rng.normal(spec.age_mean, 12, n), 18, 80),
        april_rel=np.exp(rng.normal(0, 0.4, n)),
        gd_rel=np.exp(rng.normal(0, 0.45, n)),
        fhr5_rel=np.exp(rng.normal(0, 0.35, n)),
        c3_int=rng.choice([0, 1, 2, 3], size=n, p=[0.05, 0.25, 0.45, 0.25]).astype(float),
        c4d=(rng.random(n) < p["c4d"]).astype(float),
        M=(rng.random(n) < p["M"]).astype(float),
        E=(rng.random(n) < p["E"]).astype(float),
        S=(rng.random(n) < p["S"]).astype(float),
        T=T.astype(float),
        C=C.astype(float),
    )


@dataclass
class PatientParams:
    """Vectorised patient-level model parameters (one entry per simulated row)."""

    f_ind_gd: np.ndarray
    f_ind_g: np.ndarray
    kD: np.ndarray
    loop_gain: np.ndarray
    f_lectin: np.ndarray
    f_classical: np.ndarray
    w_direct: np.ndarray
    w_c3: np.ndarray
    w_c5: np.ndarray
    phi: np.ndarray
    prog: np.ndarray
    upcr0: np.ndarray
    egfr0: np.ndarray
    weight: np.ndarray

    @property
    def n(self) -> int:
        return len(self.upcr0)

    def take(self, idx) -> "PatientParams":
        return PatientParams(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    @staticmethod
    def concat(items: list["PatientParams"]) -> "PatientParams":
        return PatientParams(**{f.name: np.concatenate([getattr(p, f.name) for p in items]) for f in fields(PatientParams)})


def patient_params(cov: Covariates, params: ModelParams, eta: np.ndarray | None = None) -> PatientParams:
    """Map covariates (+ random effects ``eta`` of shape (n, 5)) to model parameters."""
    n = cov.n
    eta = np.zeros((n, len(ETA_NAMES))) if eta is None else np.asarray(eta, float).reshape(n, len(ETA_NAMES))
    om = params.omega
    e = {name: eta[:, i] * om[name] for i, name in enumerate(ETA_NAMES)}

    # Hit 1: higher serum APRIL -> Gd-IgA1 output more APRIL dependent.
    f_ind_gd = expit(logit(params.f_ind_gd) - 0.5 * np.log(cov.april_rel) + e["f_ind_gd"])
    f_ind_g = np.full(n, params.f_ind_g)

    # Hit 4 / complement: share of injury that is complement mediated rises with
    # mesangial C3 intensity and FHR-5/FH ratio (factor H de-regulation).
    w_comp = expit(logit(1.0 - params.w_direct) + 0.4 * (cov.c3_int - 2.0) + 0.6 * np.log(cov.fhr5_rel) + e["w_comp"])
    w_direct = 1.0 - w_comp
    w_c3 = w_comp * params.c3_share
    w_c5 = w_comp * (1.0 - params.c3_share)
    loop_gain = expit(logit(params.loop_gain) + 0.7 * np.log(cov.fhr5_rel))

    # Lectin pathway initiation is concentrated in C4d-positive biopsies.
    f_lectin = np.clip(np.where(cov.c4d > 0, 3.0, 0.5) * params.f_lectin / 1.25, 0.0, 0.9)  # mean ~f_lectin at 30% C4d+
    f_classical = np.full(n, params.f_classical)

    # Reversible (active) share of proteinuria: endocapillary/crescentic activity
    # raises it, tubulointerstitial fibrosis lowers it.
    phi = expit(logit(params.phi) + 0.5 * cov.E + 0.3 * (cov.C > 0) - 0.7 * cov.T + e["phi"])

    kD = LN2 / params.t_half_deposit * np.exp(e["kD"])
    prog = np.exp(0.3 * cov.T + 0.15 * cov.S + 0.2 * np.log(cov.gd_rel) + e["prog"])

    return PatientParams(
        f_ind_gd=f_ind_gd,
        f_ind_g=f_ind_g,
        kD=kD,
        loop_gain=loop_gain,
        f_lectin=f_lectin,
        f_classical=f_classical,
        w_direct=w_direct,
        w_c3=w_c3,
        w_c5=w_c5,
        phi=phi,
        prog=prog,
        upcr0=np.asarray(cov.upcr0, float),
        egfr0=np.asarray(cov.egfr0, float),
        weight=np.asarray(cov.weight, float),
    )


def virtual_population(spec: PopulationSpec, n: int, params: ModelParams, seed: int = 0):
    """Sample ``n`` virtual patients. Returns (covariates, eta, PatientParams)."""
    rng = np.random.default_rng(seed)
    cov = sample_covariates(spec, n, rng)
    eta = rng.standard_normal((n, len(ETA_NAMES)))
    return cov, eta, patient_params(cov, params, eta)
