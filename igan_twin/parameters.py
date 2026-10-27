"""Model parameters for the IgAN digital twin.

Two kinds of parameters live here:

* ``CALIBRATED`` parameters are estimated against published trial read-outs
  (see :mod:`igan_twin.trials`). Their defaults are a reasonable starting
  point; after ``python -m igan_twin calibrate`` the fitted values are stored in
  ``results/calibrated_params.json`` and picked up by :func:`load_params`.
* Fixed parameters are kinetic constants taken from general immunology /
  nephrology knowledge (half-lives of immunoglobulins, plasma-cell turnover,
  etc.). They are not identifiable from trial summary data, so they are held
  fixed and documented.

Time is in days throughout. Biological states are normalised so that an
untreated patient sits at 1.0 at baseline.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

LN2 = math.log(2.0)

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATED_PATH = REPO_ROOT / "results" / "calibrated_params.json"

# Names of the parameters estimated by igan_twin.calibration, with the
# transform used during optimisation ("log", "logit") and bounds on the
# natural scale.
CALIBRATED = {
    "ec50_sib": ("log", 0.05, 50.0),
    "ec50_ata": ("log", 0.05, 50.0),
    "f_ind_gd": ("logit", 0.02, 0.8),
    "f_ind_g": ("logit", 0.2, 0.95),
    "t_half_deposit": ("log", 10.0, 240.0),
    "phi": ("logit", 0.3, 0.98),
    "w_direct": ("logit", 0.05, 0.9),
    "c3_share": ("logit", 0.02, 0.9),
    "loop_gain": ("logit", 0.3, 0.95),
    "f_lectin": ("logit", 0.01, 0.6),
    "k_upcr": ("log", 0.005, 0.5),
    "psi": ("logit", 0.001, 0.3),
}


@dataclass(frozen=True)
class ModelParams:
    # ------------------------------------------------------------------
    # Calibrated (population typical values)
    # ------------------------------------------------------------------
    ec50_sib: float = 2.0
    """In-vivo sibeprenlimab concentration (mg/L) giving 50% APRIL neutralisation."""
    ec50_ata: float = 1.0
    """In-vivo atacicept concentration (mg/L) giving 50% APRIL and BAFF neutralisation."""
    f_ind_gd: float = 0.25
    """Fraction of Gd-IgA1 output from APRIL/BAFF-independent long-lived plasma cells."""
    f_ind_g: float = 0.65
    """Same for the IgG (anti-Gd-IgA1 autoantibody) plasma-cell compartment."""
    t_half_deposit: float = 60.0
    """Half-life (days) of mesangial IgA-immune-complex deposits."""
    phi: float = 0.8
    """Fraction of baseline proteinuria driven by active immune injury (reversible)."""
    w_direct: float = 0.4
    """Share of glomerular injury driven by deposits independent of complement
    (mesangial CD71/TfR & integrin signalling, IL-6/PDGF release)."""
    c3_share: float = 0.25
    """Share of complement-mediated injury carried by C3 fragments (C3a/C3b/iC3b)
    rather than terminal pathway (C5a / C5b-9)."""
    loop_gain: float = 0.8
    """Alternative-pathway amplification loop gain g (0<g<1); amplification = 1/(1-g)."""
    f_lectin: float = 0.15
    """Share of complement initiation on deposits via the lectin pathway (MBL/ficolin-MASP-2)."""
    k_upcr: float = 0.07
    """Fractional nephron loss per year per 1 g/g of UPCR."""
    psi: float = 0.05
    """Reversible (haemodynamic/inflammatory) fraction of eGFR lost to active injury."""

    # ------------------------------------------------------------------
    # Fixed immunology
    # ------------------------------------------------------------------
    april_K: float = 0.3
    """Receptor saturation constant for APRIL/BAFF signalling (normalised ligand)."""
    wA_gd: float = 0.85
    """APRIL (vs BAFF) share of TNF-ligand support for IgA1 plasma cells (TACI/BCMA)."""
    wA_g: float = 0.35
    """APRIL share for IgG plasma cells/plasmablasts (more BAFF/BAFF-R dependent)."""
    t_half_pc: float = 21.0
    """Turnover half-life of APRIL-dependent IgA1 plasma cells/plasmablasts."""
    t_half_pc_g: float = 30.0
    t_half_iga: float = 5.5
    """Serum IgA1 half-life (days)."""
    t_half_igg: float = 21.0
    t_half_ic: float = 1.0
    """Circulating immune complex half-life (hepatic/splenic clearance)."""
    f_classical: float = 0.05
    """Classical-pathway share of initiation (C1q is usually absent in IgAN)."""
    t_half_c3: float = 7.0
    """Turnover of mesangial C3 fragment deposition."""
    t_half_mac: float = 2.0
    t_half_inj: float = 30.0
    """Resolution half-life of mesangial proliferation / glomerular inflammation."""

    # ------------------------------------------------------------------
    # Fixed kidney
    # ------------------------------------------------------------------
    k_bg: float = 0.008
    """Background (age-related) fractional nephron loss per year."""
    eta_struct: float = 1.0
    """Structural proteinuria scales as (1/N)**eta_struct with nephron mass N."""
    placebo_tau: float = 60.0
    """Time constant of trial (placebo / regression-to-the-mean) response."""

    # ------------------------------------------------------------------
    # Inter-individual variability (SD on the transformed scale)
    # ------------------------------------------------------------------
    omega: dict = field(
        default_factory=lambda: {
            "f_ind_gd": 0.6,
            "w_comp": 0.6,
            "phi": 0.7,
            "kD": 0.4,
            "prog": 0.35,
        }
    )

    def kinetic(self, name: str) -> float:
        """First-order rate constant (1/day) for a ``t_half_*`` parameter."""
        return LN2 / getattr(self, f"t_half_{name}")

    def calibrated_dict(self) -> dict:
        return {k: getattr(self, k) for k in CALIBRATED}

    def with_values(self, **kwargs) -> "ModelParams":
        return replace(self, **kwargs)

    def to_json(self) -> dict:
        return asdict(self)


def load_params(path: Path | None = None) -> ModelParams:
    """Return default parameters overridden by the calibrated JSON if present."""
    path = CALIBRATED_PATH if path is None else Path(path)
    base = ModelParams()
    if not path.exists():
        return base
    data = json.loads(path.read_text())
    values = data.get("params", data)
    known = {f.name for f in fields(ModelParams)}
    return replace(base, **{k: v for k, v in values.items() if k in known})
