"""Pharmacokinetics and target engagement of APRIL- and complement-directed drugs.

Each drug is a linear one-compartment model with optional first-order
absorption (SC) or zero-order input (oral, treated as continuous because the
twin runs on a 1-day step). Doses are applied as events by the integrator in
:mod:`igan_twin.model`.

PK values are *effective* parameters chosen to reproduce published half-lives
and the degree of target suppression reported in trials; they are not a
replacement for the sponsors' population-PK models. Potencies flagged
``calibrated`` are overridden from :class:`~igan_twin.parameters.ModelParams`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TARGETS = ("APRIL", "BAFF", "FB", "C3", "C5", "MASP2")


@dataclass(frozen=True)
class DrugPK:
    name: str
    mechanism: str
    V: float  # L at 70 kg
    CL: float  # L/day at 70 kg
    ka: float = 0.25  # 1/day, SC absorption
    F_sc: float = 0.7
    F_oral: float = 1.0
    # target -> (EC50 mg/L, Hill, Imax)
    targets: dict = field(default_factory=dict)
    calibrated_ec50: str | None = None  # ModelParams attribute overriding EC50

    @property
    def t_half(self) -> float:
        return math.log(2) * self.V / self.CL


DRUGS: dict[str, DrugPK] = {
    "sibeprenlimab": DrugPK(
        "sibeprenlimab",
        "anti-APRIL IgG2 mAb",
        V=5.5,
        CL=0.2,  # t1/2 ~19 d
        ka=0.25,
        F_sc=0.7,
        targets={"APRIL": (2.0, 1.0, 1.0)},
        calibrated_ec50="ec50_sib",
    ),
    "zigakibart": DrugPK(
        "zigakibart",
        "humanised anti-APRIL mAb (BION-1301)",
        V=5.5,
        CL=0.25,  # t1/2 ~15 d assumed (typical IgG); exposure enters via the fitted EC50
        ka=0.25,
        F_sc=0.7,
        targets={"APRIL": (1.0, 1.0, 1.0)},
        calibrated_ec50="ec50_ziga",
    ),
    "atacicept": DrugPK(
        "atacicept",
        "TACI-Fc fusion, neutralises APRIL and BAFF",
        V=8.0,
        CL=1.1,  # t1/2 ~5 d
        ka=0.3,
        F_sc=0.5,
        targets={"APRIL": (1.0, 1.0, 1.0), "BAFF": (1.0, 1.0, 1.0)},
        calibrated_ec50="ec50_ata",
    ),
    "iptacopan": DrugPK(
        "iptacopan",
        "oral factor B inhibitor (alternative pathway)",
        V=100.0,
        CL=70.0,  # t1/2 ~1 d
        targets={"FB": (0.6, 1.0, 0.95)},
    ),
    "ravulizumab": DrugPK(
        "ravulizumab",
        "long-acting anti-C5 mAb",
        V=5.3,
        CL=0.073,  # t1/2 ~50 d
        targets={"C5": (60.0, 4.0, 0.99)},
    ),
    "narsoplimab": DrugPK(
        "narsoplimab",
        "anti-MASP-2 mAb (lectin pathway)",
        V=4.0,
        CL=0.3,  # t1/2 ~9 d
        targets={"MASP2": (0.5, 1.0, 0.95)},
    ),
    "pegcetacoplan": DrugPK(
        "pegcetacoplan",
        "pegylated C3 inhibitor (scenario only, not calibrated in IgAN)",
        V=4.0,
        CL=0.35,
        ka=0.2,
        F_sc=0.8,
        targets={"C3": (50.0, 1.0, 0.9)},
    ),
}

DRUG_INDEX = {name: i for i, name in enumerate(DRUGS)}


@dataclass(frozen=True)
class Regimen:
    """A dosing regimen.

    ``route``: ``"iv"`` bolus into central, ``"sc"`` into depot, ``"oral"``
    continuous zero-order input of ``dose_mg`` every ``interval_days``.
    """

    drug: str
    dose_mg: float | None = None
    dose_mg_per_kg: float | None = None
    interval_days: float = 28.0
    route: str = "iv"
    start_day: float = 0.0
    stop_day: float | None = None  # last dose strictly before this day
    n_doses: int | None = None
    loading: tuple = ()  # extra (day, mg) doses, e.g. ravulizumab loading

    def __post_init__(self):
        if self.drug not in DRUGS:
            raise ValueError(f"unknown drug {self.drug!r}; choose from {list(DRUGS)}")
        if self.route not in ("iv", "sc", "oral"):
            raise ValueError("route must be iv, sc or oral")
        if self.dose_mg is None and self.dose_mg_per_kg is None:
            raise ValueError("give dose_mg or dose_mg_per_kg")

    def label(self) -> str:
        dose = f"{self.dose_mg:g} mg" if self.dose_mg is not None else f"{self.dose_mg_per_kg:g} mg/kg"
        freq = {0.5: "BID", 1: "QD", 7: "QW", 14: "Q2W", 28: "Q4W", 56: "Q8W"}.get(
            self.interval_days, f"q{self.interval_days:g}d"
        )
        return f"{self.drug} {dose} {self.route.upper()} {freq}"

    def dose_events(self, t_end: float) -> list[tuple[float, float | None, float | None]]:
        """(day, mg, mg_per_kg) bolus events up to ``t_end`` (non-oral routes)."""
        if self.route == "oral":
            return []
        stop = t_end if self.stop_day is None else min(self.stop_day, t_end)
        events = [(float(d), float(mg), None) for d, mg in self.loading]
        first = self.start_day + (max(d for d, _ in self.loading) + self.interval_days if self.loading else 0.0)
        k, t = 0, first
        while t < stop and (self.n_doses is None or k < self.n_doses):
            events.append((t, self.dose_mg, self.dose_mg_per_kg))
            k += 1
            t = first + k * self.interval_days
        return events

    def oral_rate(self) -> float:
        """mg/day for oral regimens."""
        return self.dose_mg / self.interval_days if self.route == "oral" else 0.0


def engagement_params(params) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Drug -> target -> (EC50, Hill, Imax) with calibrated potencies applied."""
    out = {}
    for name, drug in DRUGS.items():
        tg = dict(drug.targets)
        if drug.calibrated_ec50 and hasattr(params, drug.calibrated_ec50):
            ec50 = getattr(params, drug.calibrated_ec50)
            tg = {t: (ec50, h, imax) for t, (_, h, imax) in tg.items()}
        out[name] = tg
    return out


# Common regimens used in trials ---------------------------------------------

def sibeprenlimab_iv(mg_per_kg: float) -> Regimen:
    return Regimen("sibeprenlimab", dose_mg_per_kg=mg_per_kg, interval_days=28, route="iv")


ZIGAKIBART_600_Q2W = Regimen("zigakibart", dose_mg=600, interval_days=14, route="sc")
SIBEPRENLIMAB_SC_400 = Regimen("sibeprenlimab", dose_mg=400, interval_days=28, route="sc")
ATACICEPT_150_QW = Regimen("atacicept", dose_mg=150, interval_days=7, route="sc")
IPTACOPAN_200_BID = Regimen("iptacopan", dose_mg=200, interval_days=0.5, route="oral")
RAVULIZUMAB_IV = Regimen(
    "ravulizumab", dose_mg=3300, interval_days=56, route="iv", loading=((0, 2700), (14, 3300))
)
NARSOPLIMAB_IV_12W = Regimen("narsoplimab", dose_mg=370, interval_days=7, route="iv", n_doses=12)
PEGCETACOPLAN_SC = Regimen("pegcetacoplan", dose_mg=1080, interval_days=3.5, route="sc")
