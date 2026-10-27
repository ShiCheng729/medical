"""Mechanistic IgAN model: the four-hit cascade with APRIL and complement biology.

Disease cascade (all states normalised to 1.0 at untreated baseline)::

    APRIL / BAFF ──TACI/BCMA──> IgA1 plasma cells (PA) ──> Gd-IgA1 (GD)      hit 1
                 └──BAFF-R/TACI─> IgG plasma cells (PG) ──> anti-Gd-IgA1 IgG (AB)  hit 2
    GD x AB ──> circulating immune complexes (IC)                            hit 3
    IC ──> mesangial deposits (D)                                            hit 4
    D ──lectin (MASP-2) / classical / alternative (factor B) initiation──┐
          alternative-pathway amplification loop (gain g, FH/FHR-5) ─────┴> C3 fragments (C3)
    C3 ──C5 convertase──> C5a / C5b-9 (MAC)
    w_direct*D + w_c3*C3 + w_c5*MAC ──> glomerular injury / mesangial activation (INJ)
    INJ ──> proteinuria (UPCR) ──> nephron loss (N) ──> eGFR

Drug targets: APRIL (sibeprenlimab, atacicept), BAFF (atacicept), factor B
(iptacopan), MASP-2 (narsoplimab), C5 (ravulizumab), C3 (pegcetacoplan).

The model is integrated with a fixed-step RK4 scheme vectorised across
patients ("rows"), so a whole virtual trial is one integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import ModelParams
from .pk import DRUG_INDEX, DRUGS, TARGETS, Regimen, engagement_params
from .population import PatientParams

STATES = ("PA", "PG", "GD", "AB", "IC", "D", "C3", "MAC", "INJ", "N")
S = {name: i for i, name in enumerate(STATES)}
NS = len(STATES)
ND = len(DRUGS)

OUTPUTS = ("UPCR", "eGFR") + STATES + tuple(f"e_{t}" for t in TARGETS)


@dataclass
class SimResult:
    t: np.ndarray  # recorded days
    out: dict  # name -> array (n_times, n_rows)
    arm_of_row: np.ndarray
    arms: list

    def at(self, name: str, day: float) -> np.ndarray:
        i = int(np.argmin(np.abs(self.t - day)))
        return self.out[name][i]

    def rows(self, arm: int) -> np.ndarray:
        return np.flatnonzero(self.arm_of_row == arm)

    def gm_ratio(self, arm: int, day: float, name: str = "UPCR") -> float:
        """Geometric-mean ratio to baseline of ``name`` in ``arm`` at ``day``."""
        r = self.rows(arm)
        return float(np.exp(np.mean(np.log(self.at(name, day)[r] / self.out[name][0][r]))))

    def mean_change(self, arm: int, day: float, name: str = "eGFR") -> float:
        r = self.rows(arm)
        return float(np.mean(self.at(name, day)[r] - self.out[name][0][r]))

    def quantiles(self, name: str, arm: int, q=(0.05, 0.5, 0.95), relative: bool = False) -> np.ndarray:
        x = self.out[name][:, self.rows(arm)]
        if relative:
            x = x / x[0]
        return np.quantile(x, q, axis=1)


class _Dosing:
    """Pre-computed dosing events and oral input rates for all rows."""

    def __init__(self, pp: PatientParams, arms: list[list[Regimen]], arm_of_row: np.ndarray, t_end: float, dt: float):
        n = pp.n
        w = pp.weight
        self.V = np.empty((ND, n))
        self.k = np.empty((ND, n))
        self.ka = np.empty((ND, 1))
        for name, d in DRUGS.items():
            i = DRUG_INDEX[name]
            self.V[i] = d.V * (w / 70.0)
            self.k[i] = d.CL * (w / 70.0) ** 0.75 / self.V[i]
            self.ka[i] = d.ka
        self.rate = np.zeros((ND, n))
        self.r_start = np.full((ND, n), np.inf)
        self.r_stop = np.full((ND, n), np.inf)
        self.events: dict[int, list] = {}
        for a, regimens in enumerate(arms):
            rows = np.flatnonzero(arm_of_row == a)
            if rows.size == 0:
                continue
            for reg in regimens:
                d = DRUG_INDEX[reg.drug]
                drug = DRUGS[reg.drug]
                if reg.route == "oral":
                    self.rate[d, rows] = drug.F_oral * reg.oral_rate()
                    self.r_start[d, rows] = reg.start_day
                    self.r_stop[d, rows] = np.inf if reg.stop_day is None else reg.stop_day
                    continue
                for day, mg, mgkg in reg.dose_events(t_end):
                    amount = np.full(rows.size, mg) if mg is not None else mgkg * w[rows]
                    if reg.route == "sc":
                        amount = amount * drug.F_sc
                    comp = 0 if reg.route == "sc" else 1
                    step = int(round(day / dt))
                    self.events.setdefault(step, []).append((d, rows, amount, comp))

    def oral_input(self, t: float) -> np.ndarray:
        on = (t >= self.r_start) & (t < self.r_stop)
        return np.where(on, self.rate, 0.0)


def _engagement_tables(params: ModelParams):
    """Per target: list of (drug_index, EC50, Hill, Imax)."""
    eng = engagement_params(params)
    table = {t: [] for t in TARGETS}
    for name, tg in eng.items():
        for target, (ec50, h, imax) in tg.items():
            table[target].append((DRUG_INDEX[name], ec50, h, imax))
    return table


def _engage(conc: np.ndarray, table) -> dict:
    e = {}
    for target, entries in table.items():
        free = 1.0
        for d, ec50, h, imax in entries:
            c = conc[d]
            if not np.any(c > 0):
                continue
            ch = c**h if h != 1.0 else c
            free = free * (1.0 - imax * ch / (ch + ec50**h))
        e[target] = 1.0 - free if not np.isscalar(free) else np.zeros(conc.shape[1])
    return e


def initial_state(n: int) -> np.ndarray:
    y = np.zeros((NS + 2 * ND, n))
    y[:NS] = 1.0
    return y


def simulate(
    params: ModelParams,
    pp: PatientParams,
    arms: list[list[Regimen]],
    arm_of_row: np.ndarray | None = None,
    t_end: float = 365.0,
    dt: float = 1.0,
    record_every: float = 7.0,
    placebo_end: np.ndarray | float = 1.0,
    placebo_T: np.ndarray | float = 365.0,
    extra_records: tuple = (),
) -> SimResult:
    """Simulate every row of ``pp`` under the regimens of its arm.

    ``placebo_end`` is the trial (placebo / regression-to-the-mean) UPCR
    multiplier reached at ``placebo_T`` days; it applies to every arm equally.
    """
    n = pp.n
    arm_of_row = np.zeros(n, int) if arm_of_row is None else np.asarray(arm_of_row)
    dosing = _Dosing(pp, arms, arm_of_row, t_end, dt)
    table = _engagement_tables(params)

    kpa, kpg = params.kinetic("pc"), params.kinetic("pc_g")
    kiga, kigg, kic = params.kinetic("iga"), params.kinetic("igg"), params.kinetic("ic")
    kc3, kmac, kinj = params.kinetic("c3"), params.kinetic("mac"), params.kinetic("inj")
    K = params.april_K
    g = pp.loop_gain
    f_ap = 1.0 - pp.f_lectin - pp.f_classical
    pl_end = np.broadcast_to(np.asarray(placebo_end, float), (n,))
    pl_T = np.broadcast_to(np.asarray(placebo_T, float), (n,))
    tau = params.placebo_tau
    pl_norm = 1.0 - np.exp(-pl_T / tau)

    def h(s):  # saturable ligand signalling, h(1) = 1
        return s * (1.0 + K) / (s + K)

    def upcr(t, y):
        pl = 1.0 + (pl_end - 1.0) * (1.0 - np.exp(-t / tau)) / pl_norm
        # hyperfiltration of remaining nephrons; capped so end-stage kidneys stay finite
        struct = np.maximum(y[S["N"]], 0.15) ** (-params.eta_struct)
        return pp.upcr0 * pl * (pp.phi * y[S["INJ"]] + (1.0 - pp.phi) * struct)

    def rhs(t, y):
        depot, cent = y[NS : NS + ND], y[NS + ND :]
        conc = cent / dosing.V
        e = _engage(conc, table)
        dy = np.empty_like(y)
        dy[NS : NS + ND] = -dosing.ka * depot
        dy[NS + ND :] = dosing.ka * depot - dosing.k * cent + dosing.oral_input(t)

        sA, sB = h(1.0 - e["APRIL"]), h(1.0 - e["BAFF"])
        drive_a = pp.f_ind_gd + (1.0 - pp.f_ind_gd) * (params.wA_gd * sA + (1.0 - params.wA_gd) * sB)
        drive_g = pp.f_ind_g + (1.0 - pp.f_ind_g) * (params.wA_g * sA + (1.0 - params.wA_g) * sB)
        dy[S["PA"]] = kpa * (drive_a - y[S["PA"]])
        dy[S["PG"]] = kpg * (drive_g - y[S["PG"]])
        dy[S["GD"]] = kiga * (y[S["PA"]] - y[S["GD"]])
        dy[S["AB"]] = kigg * (y[S["PG"]] - y[S["AB"]])
        dy[S["IC"]] = kic * (y[S["GD"]] * y[S["AB"]] - y[S["IC"]])
        dy[S["D"]] = pp.kD * (y[S["IC"]] - y[S["D"]])

        iFB, iC3, iM = e["FB"], e["C3"], e["MASP2"]
        init = y[S["D"]] * (pp.f_lectin * (1.0 - iM) + pp.f_classical + f_ap * (1.0 - iFB))
        amp = (1.0 - g) / (1.0 - g * (1.0 - iFB) * (1.0 - iC3))
        dy[S["C3"]] = kc3 * (init * (1.0 - iC3) * amp - y[S["C3"]])
        dy[S["MAC"]] = kmac * (y[S["C3"]] * (1.0 - e["C5"]) - y[S["MAC"]])
        drive_inj = pp.w_direct * y[S["D"]] + pp.w_c3 * y[S["C3"]] + pp.w_c5 * y[S["MAC"]]
        dy[S["INJ"]] = kinj * (drive_inj - y[S["INJ"]])
        dy[S["N"]] = -y[S["N"]] * (params.k_bg + params.k_upcr * pp.prog * upcr(t, y)) / 365.0
        return dy, e

    def record(t, y, e, buf):
        u = upcr(t, y)
        buf["UPCR"].append(u)
        buf["eGFR"].append(pp.egfr0 * y[S["N"]] * (1.0 - params.psi * (y[S["INJ"]] - 1.0)))
        for name in STATES:
            buf[name].append(y[S[name]].copy())
        for tg in TARGETS:
            buf[f"e_{tg}"].append(np.asarray(e[tg]).copy())

    n_steps = int(round(t_end / dt))
    rec_stride = max(1, int(round(record_every / dt)))
    y = initial_state(n)
    buf = {k: [] for k in OUTPUTS}
    times = []
    for step in range(n_steps + 1):
        t = step * dt
        for d, rows, amount, comp in dosing.events.get(step, ()):
            y[NS + comp * ND + d, rows] += amount
        if step % rec_stride == 0 or step == n_steps:
            _, e = rhs(t, y)
            record(t, y, e, buf)
            times.append(t)
        if step == n_steps:
            break
        k1, _ = rhs(t, y)
        k2, _ = rhs(t + dt / 2, y + dt / 2 * k1)
        k3, _ = rhs(t + dt / 2, y + dt / 2 * k2)
        k4, _ = rhs(t + dt, y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        y[: NS] = np.maximum(y[:NS], 1e-9)

    out = {k: np.array(v) for k, v in buf.items()}
    return SimResult(t=np.array(times), out=out, arm_of_row=arm_of_row, arms=arms)
