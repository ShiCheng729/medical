import numpy as np
import pytest

from igan_twin.model import S, simulate
from igan_twin.parameters import ModelParams, load_params
from igan_twin.pk import (
    ATACICEPT_150_QW,
    IPTACOPAN_200_BID,
    NARSOPLIMAB_IV_12W,
    RAVULIZUMAB_IV,
    SIBEPRENLIMAB_SC_400,
    Regimen,
)
from igan_twin.population import Patient, patient_params
from igan_twin.trials import TRIALS, run_trial
from igan_twin.twin import fit_twin, forecast, synthetic_observations

P = ModelParams()


def one(regimens, t_end=365, patient=None, params=P):
    patient = patient or Patient(egfr0=60, upcr0=1.5)
    pp = patient_params(patient.covariates(), params)
    return simulate(params, pp, [regimens], t_end=t_end, record_every=7)


def test_untreated_immunology_is_at_steady_state():
    sim = one([])
    for name in ("PA", "PG", "GD", "AB", "IC", "D", "C3", "MAC", "INJ"):
        assert np.allclose(sim.out[name], 1.0, atol=1e-9), name
    # kidney progresses without treatment
    assert sim.out["eGFR"][-1, 0] < sim.out["eGFR"][0, 0]
    assert sim.out["UPCR"][0, 0] == pytest.approx(1.5)


def test_anti_april_acts_upstream_of_deposits():
    sim = one([SIBEPRENLIMAB_SC_400])
    assert sim.out["e_APRIL"][-1, 0] > 0.8
    assert sim.out["GD"][-1, 0] < 0.6
    assert sim.out["D"][-1, 0] < sim.out["IC"][0, 0]
    assert sim.out["UPCR"][-1, 0] < 1.0


def test_iptacopan_spares_hits_1_to_4_but_blocks_complement():
    sim = one([IPTACOPAN_200_BID])
    assert np.allclose(sim.out["GD"][-1], 1.0)
    assert np.allclose(sim.out["D"][-1], 1.0)
    assert sim.out["C3"][-1, 0] < 0.3
    assert sim.out["MAC"][-1, 0] < 0.3


def test_ravulizumab_blocks_terminal_pathway_only():
    sim = one([RAVULIZUMAB_IV], t_end=182)
    assert np.allclose(sim.out["C3"][-1], 1.0)
    assert sim.out["MAC"][-1, 0] < 0.1


def test_narsoplimab_effect_depends_on_lectin_pathway():
    neg = one([NARSOPLIMAB_IV_12W], t_end=84, patient=Patient(egfr0=60, upcr0=2, c4d=0))
    pos = one([NARSOPLIMAB_IV_12W], t_end=84, patient=Patient(egfr0=60, upcr0=2, c4d=1))
    assert pos.out["C3"][-1, 0] < neg.out["C3"][-1, 0]


def test_atacicept_also_lowers_autoantibody_more_than_anti_april():
    ata = one([ATACICEPT_150_QW])
    sib = one([SIBEPRENLIMAB_SC_400])
    assert ata.out["AB"][-1, 0] < sib.out["AB"][-1, 0]


def test_regimen_schedule():
    r = Regimen("sibeprenlimab", dose_mg=400, interval_days=28, route="sc", stop_day=100)
    assert [d for d, _, _ in r.dose_events(365)] == [0, 28, 56, 84]
    assert [d for d, _, _ in RAVULIZUMAB_IV.dose_events(200)] == [0, 14, 70, 126, 182]
    assert len(NARSOPLIMAB_IV_12W.dose_events(365)) == 12


def test_placebo_factor_reproduces_reported_placebo_change():
    trial = TRIALS["APPLAUSE"]
    res = run_trial(trial, P, n=40, seed=2)
    assert res.sim.gm_ratio(0, trial.primary_day) == pytest.approx(1 + trial.placebo_change, rel=0.02)


def test_calibrated_model_reproduces_primary_endpoints():
    params = load_params()
    if params == ModelParams():
        pytest.skip("no calibrated parameters yet")
    for key in ("VISIONARY", "ORIGIN3", "APPLAUSE", "SANCTUARY"):
        trial = TRIALS[key]
        res = run_trial(trial, params, n=100, seed=7)
        tgt = trial.targets[0]
        # within ~8 percentage points of published placebo-adjusted reduction
        assert abs((1 - res.predicted(tgt)) - (1 - tgt.value)) < 0.08, key


def test_twin_fit_recovers_complement_signal():
    patient = Patient(egfr0=55, upcr0=2.0, c3_int=2)
    true_eta = np.array([0.0, 1.2, 0.0, 0.0, 0.0])
    history = [IPTACOPAN_200_BID]
    obs = synthetic_observations(patient, P, true_eta, history, [30, 60, 90, 120, 150, 180], seed=1)
    fit = fit_twin(patient, history, obs, P)
    assert fit.eta[1] > 0.3  # complement share above population typical
    fc = forecast(fit, {"ipt": history, "none": []}, years=2, n_samples=30)
    assert np.median(fc.bands("ipt", "UPCR")[1][-1]) < np.median(fc.bands("none", "UPCR")[1][-1])


def test_zigakibart_fit_hits_biomarker_target_and_acts_on_april():
    from igan_twin.pk import ZIGAKIBART_600_Q2W
    from igan_twin.zigakibart import fit_ec50

    ec50, gd = fit_ec50(P, target=0.35, n=20)
    assert gd == pytest.approx(0.35, rel=0.05)
    sim = one([ZIGAKIBART_600_Q2W], params=P.with_values(ec50_ziga=ec50))
    assert sim.out["e_APRIL"][-1, 0] > 0.8
    assert sim.out["C3"][-1, 0] < 1.0  # downstream complement falls with deposits
