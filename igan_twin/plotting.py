"""Static figures (matplotlib) for calibration, mechanism and forecasts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette (light mode), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
NEUTRAL = "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "xtick.color": TEXT2,
    "ytick.color": TEXT2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 2.0,
    "font.size": 9,
})


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def calibration_figure(table: list[dict], path):
    ratios = [r for r in table if r["kind"] == "upcr_ratio"]
    egfr = [r for r in table if r["kind"] == "egfr_change"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, rows, fmt, xlabel, title in (
        (axes[0], ratios, lambda v: 100 * (1 - v), "Placebo-adjusted UPCR reduction (%)", "Proteinuria response"),
        (axes[1], egfr, lambda v: v, "Mean eGFR change (mL/min/1.73 m²)", "Kidney function"),
    ):
        labels = [f"{r['trial']} · {r['arm']} · d{r['day']:g}" for r in rows]
        y = np.arange(len(rows))[::-1]
        obs = np.array([fmt(r["observed"]) for r in rows])
        pred = np.array([fmt(r["predicted"]) for r in rows])
        for yi, o, p in zip(y, obs, pred):
            ax.plot([o, p], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.scatter(obs, y, s=46, facecolor=SURFACE, edgecolor=TEXT, lw=1.5, zorder=3, label="Published")
        ax.scatter(pred, y, s=46, color=SERIES[0], edgecolor=SURFACE, lw=1.5, zorder=4, label="Digital twin (virtual trial)")
        ax.set_yticks(y, labels, fontsize=7.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[1].axvline(0, color=MUTED, lw=1)
    axes[0].legend(loc="lower right")
    fig.suptitle("Calibration against APRIL- and complement-targeting IgAN trials", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def trials_figure(results: dict, path):
    """UPCR geometric-mean change over time per trial arm, with published endpoints."""
    keys = list(results)
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.6 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, keys):
        res = results[key]
        trial, sim = res.trial, res.sim
        for i, arm in enumerate(trial.arm_names):
            color = NEUTRAL if arm == "placebo" else SERIES[(i - 1) % len(SERIES)]
            r = sim.rows(i)
            u = sim.out["UPCR"][:, r] / sim.out["UPCR"][0, r]
            gm = 100 * (np.exp(np.mean(np.log(u), axis=1)) - 1)
            ax.plot(sim.t / 30.44, gm, color=color, label=arm, lw=2)
            ax.annotate(arm, (sim.t[-1] / 30.44, gm[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=TEXT2, va="center")
        pbo_label = False
        for tgt in trial.targets:
            if tgt.kind != "upcr_ratio":
                continue
            # recover arm-level published change from ratio and reported placebo change
            pbo = trial.placebo_change if tgt.day == trial.primary_day else None
            if pbo is None:
                continue
            drug_change = tgt.value * (1 + pbo) - 1
            m = tgt.day / 30.44
            ax.scatter([m], [100 * drug_change], s=60, facecolor=SURFACE, edgecolor=TEXT, lw=1.6, zorder=5,
                       label="published" if not pbo_label else None)
            if not pbo_label:
                ax.scatter([m], [100 * pbo], s=60, facecolor=SURFACE, edgecolor=TEXT, lw=1.6, zorder=5)
                pbo_label = True
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(trial.name, loc="left", fontsize=9)
        ax.set_xlabel("Months")
        ax.set_xlim(0, trial.duration / 30.44 * 1.35)
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    axes[0].set_ylabel("UPCR change from baseline (%, GM)")
    axes[ncol].set_ylabel("UPCR change from baseline (%, GM)") if nrow > 1 else None
    fig.suptitle("Virtual trial replicas — open circles: published drug and placebo arm results",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def mechanism_figure(sim, labels, path, months=12):
    panels = [
        ("e_APRIL", "APRIL neutralised (fraction)", False),
        ("GD", "Serum Gd-IgA1 (hit 1)", True),
        ("AB", "Anti-Gd-IgA1 IgG (hit 2)", True),
        ("IC", "Circulating immune complexes (hit 3)", True),
        ("D", "Mesangial IgA deposits (hit 4)", True),
        ("C3", "Mesangial C3 fragments", True),
        ("MAC", "C5a / C5b-9", True),
        ("INJ", "Glomerular injury", True),
        ("UPCR", "UPCR (ratio to baseline)", True),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True)
    for ax, (key, title, rel) in zip(axes.ravel(), panels):
        for a, lab in enumerate(labels):
            r = sim.rows(a)
            x = sim.out[key][:, r]
            if rel:
                x = x / x[0]
            combo = "+" in lab  # combinations overlay their components: draw dashed
            ax.plot(sim.t / 30.44, np.median(x, axis=1), color=SERIES[a - 1] if a else NEUTRAL, label=lab,
                    ls="--" if combo else "-")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel("Months")
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Mechanism read-out: where each therapy acts in the four-hit / complement cascade (median virtual patient)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=TEXT)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, path)


def longterm_figure(proj, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    years = proj.sim.t / 365.0
    for a, s in enumerate(proj.strategies):
        color = NEUTRAL if a == 0 else SERIES[(a - 1) % len(SERIES)]
        r = proj.sim.rows(a)
        axes[0].plot(years, np.median(proj.sim.out["eGFR"][:, r], axis=1), color=color, label=s)
        kf = proj.kf_free(s)
        axes[1].plot(years, 100 * kf, color=color, label=s, drawstyle="steps-post")
        axes[1].annotate(f"{100 * kf[-1]:.0f}%", (years[-1], 100 * kf[-1]), xytext=(4, 0), textcoords="offset points",
                         fontsize=7, color=TEXT2, va="center")
    axes[0].set_title("Median eGFR", loc="left")
    axes[0].set_ylabel("mL/min/1.73 m²")
    axes[1].set_title("Kidney-failure-free (eGFR ≥ 15)", loc="left")
    axes[1].set_ylabel("% of virtual patients")
    for ax in axes:
        ax.set_xlabel("Years")
    axes[1].set_xlim(0, years[-1] * 1.08)
    axes[0].legend(loc="lower left")
    fig.suptitle(f"10-year projection, high-risk IgAN virtual cohort (n={proj.n} per strategy, UPCR ≥ 1 g/g)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def precision_figure(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    cs = res["complement_share"]
    ai = res["april_independent"]
    axes[0].scatter(cs, 100 * res["reduction_ipt"], s=10, color=SERIES[2], alpha=0.6, edgecolor="none")
    axes[0].set_xlabel("Complement share of glomerular injury")
    axes[0].set_ylabel("UPCR reduction vs SoC at 9 mo (%)")
    axes[0].set_title("Iptacopan response", loc="left")
    axes[1].scatter(ai, 100 * res["reduction_sib"], s=10, color=SERIES[0], alpha=0.6, edgecolor="none")
    axes[1].set_xlabel("APRIL/BAFF-independent Gd-IgA1 fraction")
    axes[1].set_title("Sibeprenlimab response", loc="left")
    diff = 100 * (res["reduction_ipt"] - res["reduction_sib"])
    axes[2].hist(diff, bins=30, color=SERIES[6], edgecolor=SURFACE, lw=1)
    axes[2].grid(axis="x", visible=False)
    axes[2].axvline(0, color=TEXT2, lw=1)
    axes[2].set_xlabel("Iptacopan − sibeprenlimab benefit (pct-points)")
    axes[2].set_ylabel("Virtual patients")
    frac = np.mean(diff > 0)
    axes[2].set_title(f"Favours iptacopan in {100 * frac:.0f}% of patients", loc="left")
    fig.suptitle("Precision medicine: which mechanism dominates in whom?", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def twin_figure(fit, fc, truth_sim, path, horizon_years=5):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    obs = fit.observations
    od = np.array([o.day for o in obs]) / 30.44
    for a, s in enumerate(fc.scenarios):
        color = SERIES[a % len(SERIES)]
        t = fc.sim.t / 30.44
        mask = fc.sim.t <= 365 * horizon_years
        for ax, key in ((axes[0], "UPCR"), (axes[1], "eGFR")):
            q = fc.bands(s, key)
            ax.fill_between(t[mask], q[0][mask], q[2][mask], color=color, alpha=0.12, lw=0)
            ax.plot(t[mask], q[1][mask], color=color, label=s)
            if truth_sim is not None:
                ax.plot(truth_sim.t[truth_sim.t <= 365 * horizon_years] / 30.44,
                        truth_sim.out[key][truth_sim.t <= 365 * horizon_years, a], color=color, lw=1, ls="--")
        kf = fc.kidney_failure_prob(s)
        axes[2].plot(fc.sim.t / 365, 100 * kf, color=color, label=s)
    axes[0].scatter(od, [o.upcr for o in obs], s=28, color=TEXT, zorder=5, label="observed")
    axes[1].scatter(od, [o.egfr for o in obs], s=28, color=TEXT, zorder=5)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_title("UPCR (g/g)", loc="left")
    axes[1].set_title("eGFR (mL/min/1.73 m²)", loc="left")
    axes[2].set_title("Probability of kidney failure", loc="left")
    axes[0].set_xlabel("Months")
    axes[1].set_xlabel("Months")
    axes[2].set_xlabel("Years")
    axes[2].set_ylabel("%")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=len(labs), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Individual digital twin — {fit.patient.name}: median & 90% band; dashed = ground truth",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)
