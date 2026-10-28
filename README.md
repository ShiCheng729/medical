# IgAN digital twin — APRIL & complement

A mechanistic, trial-calibrated **digital twin of IgA nephropathy** (IgAN). It models the
multi-hit pathogenesis from APRIL/BAFF-driven production of galactose-deficient IgA1 through
immune-complex deposition, lectin/alternative-pathway complement activation, glomerular injury,
proteinuria and nephron loss. It is calibrated to published results of the main
**anti-APRIL** (sibeprenlimab, atacicept) and **complement-inhibitor** (iptacopan, ravulizumab,
narsoplimab) trials.

It works at two levels:

* **Virtual population / virtual trials.** You can re-run ENVISION, VISIONARY, ORIGIN 3,
  APPLAUSE-IgAN, SANCTUARY and ARTEMIS-IGAN in silico, compare strategies over 10 years, and
  run responder analyses.
* **Individual twins.** You can personalise the model to one patient from baseline labs, biopsy
  findings and follow-up visits (Bayesian MAP fit), then forecast counterfactual treatments with
  uncertainty bands, e.g. "continue anti-APRIL" vs "add iptacopan" vs "switch".

> Research software. Outputs are model-based hypotheses calibrated to aggregate trial data.
> They are not validated for clinical decision-making.

## Biology encoded

```
            APRIL ──TACI/BCMA──┐            BAFF ──BAFF-R/TACI──┐
   (sibeprenlimab, atacicept)  │            (atacicept)         │
                               ▼                                ▼
   Hit 1  IgA1 plasma cells ──> Gd-IgA1        IgG plasma cells ──> anti-Gd-IgA1 IgG   Hit 2
                                   └──────────────┬───────────────────┘
   Hit 3                          circulating immune complexes
   Hit 4                          mesangial IgA deposits ───────────────┐ direct (CD71, integrins,
                                   │                                    │ IL-6 / PDGF signalling)
            lectin (MASP-2) ◄──────┤ classical (minor)                  │
            (narsoplimab)          │ alternative (factor B) ◄─ iptacopan │
                                   ▼                                    │
                     C3 convertase + AP amplification loop (FH vs FHR-5)│
                                   ▼                                    │
                      C3a/C3b ──> C5 convertase ──> C5a / C5b-9         │
                                   │           (ravulizumab)            │
                                   ▼                                    ▼
                         glomerular injury / mesangial activation ◄─────┘
                                   ▼
                proteinuria (reversible + structural) ──> nephron loss ──> eGFR
```

State variables (all at 1.0 at untreated baseline) are listed in `igan_twin/model.py`. Drug
PK is a one-compartment model with SC/IV/oral dosing, and target engagement uses Emax
functions (`igan_twin/pk.py`).

### Patient covariates that change mechanism (`igan_twin/population.py`)

| Measurement | Mechanistic effect in the twin |
|---|---|
| Serum APRIL | higher APRIL → Gd-IgA1 output more APRIL-dependent (better anti-APRIL response) |
| FHR-5 / factor H ratio | higher → stronger AP amplification loop and larger complement share of injury |
| Mesangial C3 intensity | higher → larger complement share of injury |
| Glomerular C4d | positive → lectin pathway dominates initiation (narsoplimab target) |
| Oxford E1 / C1 | more active, reversible proteinuria |
| Oxford T1/T2, S1 | less reversible proteinuria, faster progression |
| Serum Gd-IgA1 | faster progression |
| eGFR, UPCR, weight | baseline kidney state, weight-based PK |

The effect sizes are priors chosen from direction-of-effect evidence. Individual twins refine them
through five random effects: APRIL-independence, complement share, reversibility, deposit
clearance and progression.

## Calibration data

| Trial | Drug (target) | Published endpoint used |
|---|---|---|
| ENVISION (ph 2) | sibeprenlimab IV 2/4/8 mg/kg Q4W (APRIL) | 12-mo UPCR −47.2/−58.8/−62.0% vs −20.0% placebo; eGFR −2.7/+0.2/−1.5 vs −7.4 |
| VISIONARY (ph 3) | sibeprenlimab 400 mg SC Q4W | 9-mo UPCR −50.2% vs +2.1% (51.2% adj.); 12-mo −56.6% vs −5.1% |
| ORIGIN 3 (ph 3) | atacicept 150 mg SC QW (APRIL + BAFF) | wk-36 UPCR −45.7% vs −6.8% |
| APPLAUSE-IgAN (ph 3) | iptacopan 200 mg BID (factor B) | 9-mo UPCR 38.3% placebo-adjusted |
| SANCTUARY (ph 2) | ravulizumab IV (C5) | wk-26 UPCR −41.9% vs −16.8%; eGFR +0.2 vs −4.5 |
| ARTEMIS-IGAN (ph 3) | narsoplimab (MASP-2) | no significant benefit (weak null target) |

Full references, and notes on which numbers are approximate, are in `igan_twin/trials.py`.

### Prospective prediction: zigakibart (Novartis) and the BEYOND phase 3

`python -m igan_twin zigakibart` fits zigakibart's potency to its phase 1/2 IgA/Gd-IgA1 suppression
only. It then checks the model against the phase 1/2 proteinuria curve, which is not used in the
fit, and predicts the not-yet-reported BEYOND trial (600 mg SC Q2W vs placebo; week-40 UPCR,
week-104 eGFR). The result includes posterior and trial-sampling uncertainty. Output is in
`results/ZIGAKIBART.md` (Chinese) and `results/fig_zigakibart.png`.

## Quick start

```bash
pip install -r requirements.txt
python -m igan_twin all            # calibrate → virtual trials → projections → twin demo → results/REPORT.md
python -m igan_twin trials         # re-use results/calibrated_params.json
python -m pytest -q
```

### Build a twin for your own patient

```python
from igan_twin import Patient, Observation, fit_twin, forecast, load_params
from igan_twin.pk import SIBEPRENLIMAB_SC_400, IPTACOPAN_200_BID, Regimen

params = load_params()
pt = Patient(egfr0=52, upcr0=2.1, weight=80, april_rel=1.2, fhr5_rel=1.4,
             c3_int=3, c4d=0, M=1, E=0, S=1, T=1, C=0, name="clinic-017")
visits = [Observation(day=90, upcr=1.4, egfr=51, gd_rel=0.55),
          Observation(day=180, upcr=1.1, egfr=52, gd_rel=0.45)]
twin = fit_twin(pt, history=[SIBEPRENLIMAB_SC_400], observations=visits, params=params)
print(twin.mechanism())

fc = forecast(twin, {
    "continue": [SIBEPRENLIMAB_SC_400],
    "add iptacopan": [SIBEPRENLIMAB_SC_400,
                      Regimen("iptacopan", dose_mg=200, interval_days=0.5, route="oral", start_day=180)],
}, years=10)
print(fc.summary())
```

## Repository layout

| Path | Purpose |
|---|---|
| `igan_twin/model.py` | ODE system (vectorised RK4 across patients) |
| `igan_twin/pk.py` | drug PK, target engagement, standard regimens |
| `igan_twin/population.py` | covariates → mechanism mapping, virtual populations |
| `igan_twin/trials.py` | trial designs and published targets |
| `igan_twin/calibration.py` | least-squares calibration to the trials |
| `igan_twin/twin.py` | individual twin fitting and counterfactual forecasting |
| `igan_twin/scenarios.py` | 10-year strategy comparison, responder analysis |
| `results/` | calibrated parameters, figures, `REPORT.md` |

## Main limitations

* The model is calibrated to arm-level summaries, not individual patient data. Covariate effect
  sizes and variability are priors.
* PK is approximate (no target-mediated disposition), and potencies are effective in-vivo values.
* Long-term eGFR depends on the proteinuria → nephron-loss link. Check it against the 24-month
  eGFR-slope read-outs of VISIONARY, ORIGIN 3 and APPLAUSE-IgAN when they are published.
* Supportive care (RASi, SGLT2i, sparsentan) is treated as background, not modelled mechanistically.
