"""Command line entry point: ``python -m igan_twin <command>``.

Commands
--------
calibrate   fit population parameters to published trial read-outs
trials      re-run the virtual trial replicas and plot them
project     10-year strategy comparison + responder analysis
twin-demo   personalise a twin from synthetic follow-up data and forecast
all         every step above, then write results/REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import plotting
from .calibration import calibrate, save, summarise
from .model import simulate
from .parameters import REPO_ROOT, load_params
from .pk import IPTACOPAN_200_BID, SIBEPRENLIMAB_SC_400, Regimen
from .population import Patient, PatientParams, patient_params
from .scenarios import STRATEGIES, project, responder_analysis
from .trials import TRIALS, run_trial
from .twin import _simulate_etas, fit_twin, forecast, synthetic_observations

OUT = REPO_ROOT / "results"


def cmd_calibrate(args):
    print("Calibrating against:", ", ".join(k for k, t in TRIALS.items() if t.calibrate))
    params, info = calibrate(n=args.n, max_nfev=args.max_nfev)
    table = summarise(params)
    path = save(params, info, table)
    print(f"saved {path}")
    for k, v in params.calibrated_dict().items():
        print(f"  {k:16s} {v:.4g}")
    plotting.calibration_figure(table, OUT / "fig_calibration.png")
    return params, table


def cmd_trials(args, params=None):
    params = params or load_params()
    results = {k: run_trial(t, params, n=args.n, seed=7) for k, t in TRIALS.items()}
    plotting.trials_figure(results, OUT / "fig_trials.png")
    table = summarise(params, n=args.n)
    plotting.calibration_figure(table, OUT / "fig_calibration.png")
    for r in table:
        print(f"{r['trial']:10s} {r['kind']:15s} {r['arm']:26s} d{r['day']:<4g} obs {r['observed']:7.3f}  model {r['predicted']:7.3f}")
    return results, table


def mechanism_sim(params, n=200, seed=3):
    from .scenarios import HIGH_RISK
    from .population import virtual_population

    labels = ["Supportive care", "Sibeprenlimab 400 mg Q4W", "Atacicept 150 mg QW", "Iptacopan 200 mg BID",
              "Ravulizumab Q8W", "Narsoplimab 12-wk course", "Sibeprenlimab + iptacopan"]
    cov, eta, pp = virtual_population(HIGH_RISK, n, params, seed)
    rows = PatientParams.concat([pp] * len(labels))
    sim = simulate(params, rows, [STRATEGIES[l] for l in labels], np.repeat(np.arange(len(labels)), n),
                   t_end=365, record_every=7)
    return sim, labels


def cmd_project(args, params=None):
    params = params or load_params()
    sim, labels = mechanism_sim(params)
    plotting.mechanism_figure(sim, labels, OUT / "fig_mechanism.png")
    proj = project(params, n=args.n)
    plotting.longterm_figure(proj, OUT / "fig_longterm.png")
    table = proj.table()
    resp = responder_analysis(params)
    plotting.precision_figure(resp, OUT / "fig_precision.png")
    (OUT / "projection.json").write_text(json.dumps(table, indent=2))
    for row in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    return table, resp


DEMO_PATIENT = Patient(
    name="Patient A (synthetic)", egfr0=58, upcr0=1.8, weight=72, age=36,
    april_rel=1.3, gd_rel=1.5, fhr5_rel=1.6, c3_int=3, c4d=0, M=1, E=1, S=1, T=0, C=0,
)
DEMO_TRUE_ETA = np.array([0.8, 0.9, 0.3, 0.2, 0.4])  # hidden ground truth for validation


def cmd_twin_demo(args, params=None):
    params = params or load_params()
    patient = DEMO_PATIENT
    history = [SIBEPRENLIMAB_SC_400]
    days = [30, 60, 91, 122, 152, 182]
    obs = synthetic_observations(patient, params, DEMO_TRUE_ETA, history, days, seed=4)
    prior_fit = fit_twin(patient, history, [], params)
    fit = fit_twin(patient, history, obs, params)
    t6 = 182
    scenarios = {
        "Continue sibeprenlimab": [SIBEPRENLIMAB_SC_400],
        "Add iptacopan at 6 mo": [SIBEPRENLIMAB_SC_400, Regimen("iptacopan", dose_mg=200, interval_days=0.5, route="oral", start_day=t6)],
        "Switch to iptacopan at 6 mo": [Regimen("sibeprenlimab", dose_mg=400, interval_days=28, route="sc", stop_day=t6),
                                         Regimen("iptacopan", dose_mg=200, interval_days=0.5, route="oral", start_day=t6)],
        "Stop therapy at 6 mo": [Regimen("sibeprenlimab", dose_mg=400, interval_days=28, route="sc", stop_day=t6)],
    }
    fc = forecast(fit, scenarios, years=10, n_samples=args.samples)
    fc_prior = forecast(prior_fit, scenarios, years=10, n_samples=args.samples)
    truth = _simulate_etas(patient, params, np.vstack([DEMO_TRUE_ETA] * len(scenarios)), list(scenarios.values()),
                           np.arange(len(scenarios)), t_end=3650, record_every=14)
    plotting.twin_figure(fit, fc, truth, OUT / "fig_twin.png")
    truth_kf = {s: float(np.min(truth.out["eGFR"][:, i]) < 15) for i, s in enumerate(scenarios)}
    truth_egfr5 = {s: float(truth.at("eGFR", 1825)[i]) for i, s in enumerate(scenarios)}
    summary = {
        "patient": patient.__dict__,
        "true_eta": DEMO_TRUE_ETA.tolist(),
        "estimated_eta": fit.eta_table(),
        "mechanism_estimate": fit.mechanism(),
        "forecast_personalised": fc.summary(),
        "forecast_covariates_only": fc_prior.summary(),
        "truth_eGFR_5y": truth_egfr5,
        "truth_kidney_failure_10y": truth_kf,
    }
    (OUT / "twin_demo.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("estimated_eta", "mechanism_estimate", "truth_eGFR_5y")}, indent=1, default=float))
    print(json.dumps(summary["forecast_personalised"], indent=1))
    return summary


def write_report(params, cal_table, proj_table, twin):
    from .report import render

    path = OUT / "REPORT.md"
    path.write_text(render(params, cal_table, proj_table, twin))
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="igan_twin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--n", type=int, default=60, help="virtual patients per arm during fitting")
    c.add_argument("--max-nfev", type=int, default=300)
    t = sub.add_parser("trials")
    t.add_argument("--n", type=int, default=200)
    p = sub.add_parser("project")
    p.add_argument("--n", type=int, default=300)
    d = sub.add_parser("twin-demo")
    d.add_argument("--samples", type=int, default=200)
    a = sub.add_parser("all")
    a.add_argument("--skip-calibration", action="store_true")
    a.add_argument("--n", type=int, default=200)
    a.add_argument("--samples", type=int, default=200)
    a.add_argument("--max-nfev", type=int, default=300)
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    if args.cmd == "calibrate":
        cmd_calibrate(args)
    elif args.cmd == "trials":
        cmd_trials(args)
    elif args.cmd == "project":
        cmd_project(args)
    elif args.cmd == "twin-demo":
        cmd_twin_demo(args)
    elif args.cmd == "all":
        if not args.skip_calibration:
            cmd_calibrate(argparse.Namespace(n=60, max_nfev=args.max_nfev))
        params = load_params()
        _, cal_table = cmd_trials(argparse.Namespace(n=args.n), params)
        proj_table, _ = cmd_project(argparse.Namespace(n=300), params)
        twin = cmd_twin_demo(argparse.Namespace(samples=args.samples), params)
        write_report(params, cal_table, proj_table, twin)


if __name__ == "__main__":
    main()
