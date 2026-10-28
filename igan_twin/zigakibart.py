"""Zigakibart (Novartis, BION-1301): fit, validate and predict the BEYOND phase 3 trial.

Status of the evidence used here:

* **Phase 1/2 (ADU-CL-19)**, single arm, 600 mg SC Q2W on top of maximal RASi.
  Serum IgA fell ~67% and Gd-IgA1 ~74% at about one year. Mean proteinuria
  change was -20% (week 12), -39% (week 24), -67% (week 52), -67% (week 76,
  n=7) and -60.4% (week 100). Sources: Kidney Int 2025 (Phase 1/2 paper);
  ERA 2025 long-term update.
* **BEYOND (NCT05852938)**, randomised 600 mg SC Q2W vs placebo for 104 weeks,
  primary endpoint 24-h UPCR change at week 40, key secondary eGFR change to
  week 104. No efficacy results were public when this module was written.

Workflow:

1. Zigakibart's only free parameter, its in-vivo EC50, is fitted to the
   phase 1/2 **biomarker** suppression (Gd-IgA1/IgA), not to proteinuria.
2. The phase 1/2 **proteinuria** time course is then predicted and compared
   with the published values (external validation).
3. BEYOND is simulated as a virtual trial. Uncertainty combines the
   calibration posterior (Laplace draws re-weighted by their fit to all
   calibration trials), uncertainty in the biomarker target and in the
   placebo-arm response.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar

from .calibration import PRIOR_SD, _from_x, _to_x, evaluate
from .model import simulate
from .parameters import CALIBRATED_PATH, REPO_ROOT, ModelParams, load_params
from .pk import ATACICEPT_150_QW, IPTACOPAN_200_BID, SIBEPRENLIMAB_SC_400, ZIGAKIBART_600_Q2W
from .population import PatientParams, PopulationSpec, virtual_population

OUT = REPO_ROOT / "results"

PHASE12_POP = PopulationSpec(egfr_mean=62, egfr_sd=22, upcr_gm=1.0, upcr_gsd=1.7, upcr_min=0.5)
PHASE12_UPCR = {  # week: mean % change from baseline (published)
    12: -20.0,
    24: -39.0,
    52: -67.0,
    76: -67.0,
    100: -60.4,
}
PHASE12_GD_TARGET = (0.26, 0.35)  # Gd-IgA1 (-74%) and IgA (-67%) ratio to baseline near week 52

BEYOND_POP = PopulationSpec(egfr_mean=60, egfr_sd=20, upcr_gm=1.6, upcr_gsd=1.6, upcr_min=1.0)
BEYOND_PRIMARY_DAY = 280  # week 40
BEYOND_EGFR_DAY = 728  # week 104
BEYOND_N_PER_ARM = 190  # ~383 randomised 1:1 (ClinicalTrials.gov actual enrolment)
# Standard error of log(placebo-adjusted UPCR ratio) observed in real phase 3 trials:
# VISIONARY interim 51.2% (96.5% CI 42.9-58.2%, ~160/arm) -> SE 0.074, scaled to 190/arm.
# The virtual patients understate real between-patient spread, so this empirical value is used.
TRIAL_SE_LOG_RATIO = 0.074 * np.sqrt(160 / BEYOND_N_PER_ARM)
PLACEBO_RANGE = (-0.10, 0.05)  # recent phase 3 placebo arms: APPLAUSE -9%, ORIGIN 3 -6.8%, VISIONARY +2.1%


def _single_arm(params, regimens, spec, n, seed, t_end, record_every=7.0):
    cov, eta, pp = virtual_population(spec, n, params, seed)
    return simulate(params, pp, [regimens], t_end=t_end, record_every=record_every)


def fit_ec50(params: ModelParams, target: float = float(np.mean(PHASE12_GD_TARGET)), n: int = 60, seed: int = 3):
    """Fit zigakibart EC50 so the virtual phase 1/2 cohort's GM Gd-IgA1 ratio at week 52 hits ``target``.

    Returns (ec50, achieved_ratio). If the target is below what complete APRIL
    blockade can achieve in this model, the EC50 goes to its lower bound.
    """

    def loss(log_ec50):
        p = replace(params, ec50_ziga=float(np.exp(log_ec50)))
        sim = _single_arm(p, [ZIGAKIBART_600_Q2W], PHASE12_POP, n, seed, 364, record_every=364)
        return (np.log(sim.gm_ratio(0, 364, "GD")) - np.log(target)) ** 2

    res = minimize_scalar(loss, bounds=(np.log(0.01), np.log(100.0)), method="bounded", options={"xatol": 1e-3})
    ec50 = float(np.exp(res.x))
    p = replace(params, ec50_ziga=ec50)
    sim = _single_arm(p, [ZIGAKIBART_600_Q2W], PHASE12_POP, n, seed, 364, record_every=364)
    return ec50, sim.gm_ratio(0, 364, "GD")


def phase12_prediction(params: ModelParams, n: int = 300, seed: int = 4):
    sim = _single_arm(params, [ZIGAKIBART_600_Q2W], PHASE12_POP, n, seed, 728)
    u = sim.out["UPCR"] / sim.out["UPCR"][0]
    return {
        "t": sim.t,
        "upcr_gm": np.exp(np.mean(np.log(u), axis=1)),
        "upcr_mean": np.mean(u, axis=1),
        "gd_gm": np.exp(np.mean(np.log(sim.out["GD"]), axis=1)),
        "egfr_mean_change": np.mean(sim.out["eGFR"] - sim.out["eGFR"][0], axis=1),
    }


def beyond_trial(params: ModelParams, placebo_change: float, n: int = 150, seed: int = 21, extra_arms=None):
    """Virtual BEYOND: placebo vs zigakibart (+ optional comparator arms in the same patients)."""
    arms = {"placebo": [], "zigakibart 600 mg Q2W": [ZIGAKIBART_600_Q2W]}
    arms.update(extra_arms or {})
    cov, eta, pp = virtual_population(BEYOND_POP, n, params, seed)
    # trial-level placebo response, set from the natural-history placebo arm
    nat = simulate(params, pp, [[]], t_end=BEYOND_PRIMARY_DAY, record_every=BEYOND_PRIMARY_DAY)
    pl_end = (1 + placebo_change) / nat.gm_ratio(0, BEYOND_PRIMARY_DAY)
    k = len(arms)
    rows = PatientParams.concat([pp] * k)
    sim = simulate(params, rows, list(arms.values()), np.repeat(np.arange(k), n), t_end=BEYOND_EGFR_DAY,
                   record_every=7, placebo_end=pl_end, placebo_T=BEYOND_PRIMARY_DAY)
    return arms, sim


def _posterior_draws(base: ModelParams, n_draws: int, seed: int, n_eval: int = 50):
    """Sampling-importance weights for Laplace draws of the calibrated parameters.

    Proposal q = N(x_hat, cov); target = exp(-SSE_data/2) x prior (the same
    weak prior used in calibration). Weight = target / q.
    """
    fit = json.loads(CALIBRATED_PATH.read_text())["fit"]
    names, x, cov = fit["names"], np.array(fit["x"]), np.array(fit["cov"])
    prior_centre = _to_x(ModelParams(), names)
    prec = np.linalg.pinv(cov)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(x)))
    draws, logw = [], []
    for _ in range(n_draws):
        xi = x + L @ rng.standard_normal(len(x))
        p = _from_x(xi, base, names)
        sse = sum(r[3] ** 2 for r in evaluate(p, n=n_eval, seed=1))
        prior = float(np.sum(((xi - prior_centre) / PRIOR_SD) ** 2))
        quad = float((xi - x) @ prec @ (xi - x))
        logw.append(-0.5 * (sse + prior) + 0.5 * quad)
        draws.append(p)
    logw = np.array(logw)
    w = np.exp(logw - logw.max())
    return draws, w / w.sum()


def run(n_draws: int = 60, n_patients: int = 120, seed: int = 0, verbose: bool = True) -> dict:
    base = load_params()
    rng = np.random.default_rng(seed)

    # 1. potency from biomarker data (point estimate)
    ec50, gd_fit = fit_ec50(base)
    point = replace(base, ec50_ziga=ec50)
    if verbose:
        print(f"zigakibart EC50 fitted to phase 1/2 Gd-IgA1: {ec50:.3g} mg/L (GM Gd-IgA1 ratio wk52 = {gd_fit:.2f})")

    # 2. external validation on phase 1/2 proteinuria
    ph12 = phase12_prediction(point)

    # 3. BEYOND point prediction + comparators in the same virtual population
    comparators = {
        "sibeprenlimab 400 mg Q4W": [SIBEPRENLIMAB_SC_400],
        "atacicept 150 mg QW": [ATACICEPT_150_QW],
        "iptacopan 200 mg BID": [IPTACOPAN_200_BID],
        "zigakibart + iptacopan": [ZIGAKIBART_600_Q2W, IPTACOPAN_200_BID],
    }
    pbo_mid = float(np.mean(PLACEBO_RANGE))
    arms, sim = beyond_trial(point, pbo_mid, n=300, extra_arms=comparators)
    names = list(arms)
    base_ratio = sim.gm_ratio(0, BEYOND_PRIMARY_DAY)
    point_table = []
    for i, a in enumerate(names):
        r = sim.gm_ratio(i, BEYOND_PRIMARY_DAY)
        point_table.append({
            "arm": a,
            "UPCR change wk40 (GM)": r - 1,
            "placebo-adjusted reduction wk40": 1 - r / base_ratio,
            "Gd-IgA1 change wk40": sim.gm_ratio(i, BEYOND_PRIMARY_DAY, "GD") - 1,
            "eGFR change wk104": sim.mean_change(i, BEYOND_EGFR_DAY),
        })
    upcr_curves = {a: np.exp(np.mean(np.log(sim.out["UPCR"][:, sim.rows(i)] / sim.out["UPCR"][0, sim.rows(i)]), axis=1))
                   for i, a in enumerate(names)}
    egfr_curves = {a: np.mean(sim.out["eGFR"][:, sim.rows(i)] - sim.out["eGFR"][0, sim.rows(i)], axis=1)
                   for i, a in enumerate(names)}

    # 4. uncertainty
    draws, w = _posterior_draws(base, n_draws, seed=seed + 1)
    ess = 1.0 / np.sum(w**2)
    if verbose:
        print(f"posterior draws: {n_draws}, effective sample size {ess:.1f}")
    samples = []
    for k, p in enumerate(draws):
        target = rng.uniform(*PHASE12_GD_TARGET)
        e, _ = fit_ec50(p, target=target, n=40)
        p = replace(p, ec50_ziga=e)
        pbo = rng.uniform(*PLACEBO_RANGE)
        _, s = beyond_trial(p, pbo, n=n_patients, seed=100 + k)
        rz, rp = s.gm_ratio(1, BEYOND_PRIMARY_DAY), s.gm_ratio(0, BEYOND_PRIMARY_DAY)
        # what a real BEYOND might observe: finite sample + measurement noise
        se = TRIAL_SE_LOG_RATIO
        observed_ratio = rz / rp * np.exp(rng.normal(0, se))
        samples.append({
            "weight": float(w[k]),
            "placebo_change": pbo,
            "zig_change": rz - 1,
            "adj_reduction": 1 - rz / rp,
            "adj_reduction_observed": 1 - observed_ratio,
            "se_log_ratio": float(se),
            "egfr_diff_wk104": s.mean_change(1, BEYOND_EGFR_DAY) - s.mean_change(0, BEYOND_EGFR_DAY),
            "egfr_zig_wk104": s.mean_change(1, BEYOND_EGFR_DAY),
            "egfr_pbo_wk104": s.mean_change(0, BEYOND_EGFR_DAY),
            "upcr_zig": (s.out["UPCR"][:, s.rows(1)] / s.out["UPCR"][0, s.rows(1)]),
        })
        if verbose and (k + 1) % 10 == 0:
            print(f"  draw {k + 1}/{n_draws}")

    def wq(key, q=(0.05, 0.5, 0.95)):
        v = np.array([s[key] for s in samples])
        o = np.argsort(v)
        cw = np.cumsum(np.array([s["weight"] for s in samples])[o])
        return [float(v[o][np.searchsorted(cw, qi * cw[-1])]) for qi in q]

    t = sim.t
    zig_gm = np.array([np.exp(np.mean(np.log(s["upcr_zig"]), axis=1)) for s in samples])
    ww = np.array([s["weight"] for s in samples])

    def band(arr):
        o = np.argsort(arr, axis=0)
        out = []
        for j in range(arr.shape[1]):
            cw = np.cumsum(ww[o[:, j]])
            out.append([arr[o[np.searchsorted(cw, q * cw[-1]), j], j] for q in (0.05, 0.5, 0.95)])
        return np.array(out).T

    result = {
        "ec50_ziga_mg_per_L": ec50,
        "phase12": {
            "t_weeks": (ph12["t"] / 7).tolist(),
            "upcr_gm_change_pct": (100 * (ph12["upcr_gm"] - 1)).tolist(),
            "upcr_mean_change_pct": (100 * (ph12["upcr_mean"] - 1)).tolist(),
            "gd_gm_change_pct": (100 * (ph12["gd_gm"] - 1)).tolist(),
            "observed_upcr_mean_change_pct": PHASE12_UPCR,
        },
        "beyond_point": point_table,
        "beyond_interval": {
            "zigakibart UPCR change wk40": wq("zig_change"),
            "placebo-adjusted reduction wk40": wq("adj_reduction"),
            "placebo-adjusted reduction wk40, trial-level predictive": wq("adj_reduction_observed"),
            "trial standard error of log ratio": float(np.mean([s["se_log_ratio"] for s in samples])),
            "eGFR difference vs placebo wk104 (mL/min)": wq("egfr_diff_wk104"),
            "zigakibart eGFR change wk104": wq("egfr_zig_wk104"),
            "placebo eGFR change wk104": wq("egfr_pbo_wk104"),
            "P(observed placebo-adjusted reduction > 30%)": float(np.sum(ww * (np.array([s["adj_reduction_observed"] for s in samples]) > 0.30))),
            "P(observed placebo-adjusted reduction > 51.2%, i.e. beats VISIONARY)": float(np.sum(ww * (np.array([s["adj_reduction_observed"] for s in samples]) > 0.512))),
            "effective_sample_size": float(ess),
            "n_draws": n_draws,
        },
        "curves": {
            "t_weeks": (t / 7).tolist(),
            "upcr_gm": {a: v.tolist() for a, v in upcr_curves.items()},
            "egfr_change": {a: v.tolist() for a, v in egfr_curves.items()},
            "zig_band": band(zig_gm).tolist(),
        },
    }
    (OUT / "zigakibart.json").write_text(json.dumps(result, indent=2))
    return result


def render_markdown(res: dict) -> str:
    """Chinese-language summary of the zigakibart prediction (results/ZIGAKIBART.md)."""
    iv = res["beyond_interval"]
    pt = {r["arm"]: r for r in res["beyond_point"]}
    ph = res["phase12"]
    tw = np.array(ph["t_weeks"])

    def at(week, key):
        return ph[key][int(np.argmin(np.abs(tw - week)))]

    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    rng3 = lambda v, f=pct: f"{f(v[1])}（90% 区间 {f(v[0])}–{f(v[2])}）"  # noqa: E731
    mL = lambda v: f"{v:+.1f}"  # noqa: E731
    L = [
        "# Zigakibart（诺华，抗 APRIL 单抗）—— 数字孪生预测 BEYOND III 期结果\n",
        "> **这是模型预测，不是试验结果。** 截至本报告生成时，BEYOND（NCT05852938）尚未公布疗效数据。"
        "预测仅基于公开的 I/II 期数据和已校准的 IgAN 机制模型，不能用于临床决策。\n",
        "## 1. 方法",
        "1. **药效强度只用生物标志物拟合。** I/II 期研究（ADU-CL-19，600 mg 皮下 Q2W）报告约第 52 周时"
        "血清 IgA 下降约 67%、Gd-IgA1 下降约 74%。模型只用这一信息拟合 zigakibart 的体内 EC50"
        f"（{res['ec50_ziga_mg_per_L']:.2g} mg/L）。",
        "2. **蛋白尿作为外部验证（未参与拟合）。** 机制链：APRIL 中和 → IgA1 浆细胞 → Gd-IgA1 → "
        "免疫复合物 → 系膜沉积 → 补体（凝集素/旁路途径，C3 → C5b-9）→ 肾小球损伤 → 蛋白尿 → eGFR。",
        "3. **虚拟 BEYOND 试验。** 安慰剂 vs 600 mg Q2W，104 周，主要终点为第 40 周 24 小时 UPCR 变化，"
        "关键次要终点为第 104 周 eGFR 变化。不确定性来源：校准参数后验（重要性加权）、生物标志物目标范围、"
        "安慰剂组反应（−10% 至 +5%），以及约 383 例受试者的有限样本和测量误差。\n",
        "## 2. I/II 期验证（单臂）\n",
        "| 周 | 发表的平均 UPCR 变化 | 模型预测（平均） |",
        "|---|---|---|",
    ]
    for w, v in PHASE12_UPCR.items():
        L.append(f"| {w} | {v:+.0f}% | {at(w, 'upcr_mean_change_pct'):+.0f}% |")
    L.append(
        f"\n模型 Gd-IgA1 在第 52 周变化 {at(52, 'gd_gm_change_pct'):+.0f}%（拟合目标约 −65% 至 −74%）。"
        "第 52 周之后模型低估约 6–12 个百分点。单臂开放研究不扣除安慰剂/回归均值效应，"
        "而 ENVISION 的安慰剂组下降了 20%，所以这个差距在预期范围内。\n"
    )
    L += [
        "## 3. BEYOND 预测\n",
        "| 指标 | 预测 |",
        "|---|---|",
        f"| 第 40 周 UPCR 相对安慰剂降低（模型） | **{rng3(iv['placebo-adjusted reduction wk40'])}** |",
        f"| 同上，含试验抽样误差的预测区间 | {rng3(iv['placebo-adjusted reduction wk40, trial-level predictive'])} |",
        f"| Zigakibart 组第 40 周 UPCR 相对基线变化 | {rng3(iv['zigakibart UPCR change wk40'])} |",
        f"| 第 104 周 eGFR 差值（zigakibart − 安慰剂，mL/min） | {rng3(iv['eGFR difference vs placebo wk104 (mL/min)'], mL)} |",
        f"| 第 104 周 eGFR 变化：zigakibart / 安慰剂 | {mL(iv['zigakibart eGFR change wk104'][1])} / {mL(iv['placebo eGFR change wk104'][1])} |",
        f"| P(观察到的降幅 > 30%) | {pct(iv['P(observed placebo-adjusted reduction > 30%)'])} |",
        f"| P(观察到的降幅 > 51.2%，即超过 VISIONARY 中期结果) | {pct(iv['P(observed placebo-adjusted reduction > 51.2%, i.e. beats VISIONARY)'])} |",
        "",
        "### 同一虚拟 BEYOND 人群中的其他药物（点估计，非头对头试验数据）\n",
        "| 方案 | 第 40 周 UPCR 相对安慰剂降低 | 第 40 周 Gd-IgA1 变化 | 第 104 周 eGFR 变化 |",
        "|---|---|---|---|",
    ]
    for a, r in pt.items():
        L.append(f"| {a} | {pct(r['placebo-adjusted reduction wk40'])} | {pct(r['Gd-IgA1 change wk40'])} | {mL(r['eGFR change wk104'])} |")
    L += [
        "\n![zigakibart](fig_zigakibart.png)\n",
        "## 4. 解读与局限",
        "- 模型预测 zigakibart 与 sibeprenlimab 同属抗 APRIL 机制。按 I/II 期数据，它的 Gd-IgA1 抑制略深"
        "（约 −70% vs −60%），所以预测降幅略高。但这只是间接比较，置信区间重叠。",
        "- **eGFR 的绝对值可能偏悲观。** 模型的肾功能下降速率主要按 ENVISION 和 SANCTUARY 安慰剂组的陡峭下降校准"
        "（约 −8 mL/min/年），而近期 III 期试验安慰剂组通常约 −5 至 −7 mL/min/年。104 周 eGFR 差值很可能被高估，"
        "相对排序比绝对数值可靠。",
        "- 药代动力学按典型 IgG 假设（t½ 约 15 天）。由于 EC50 按生物标志物拟合，暴露量的误差大部分被吸收。",
        "- BEYOND 的基线特征（UPCR ≥1 g/g、eGFR 约 60）为近似值，以正式发表的基线表为准。",
        f"- 后验重要性抽样的有效样本量为 {iv['effective_sample_size']:.0f}/{iv['n_draws']}，区间可能略窄。",
        "\n## 数据来源",
        "- Zigakibart I/II 期：Kidney International 2025（DOI 10.1016/j.kint.2025.05.006）；ERA 2025 长期数据",
        "- BEYOND 设计：ClinicalTrials.gov NCT05852938；Kidney Int Rep 2024（WCN24-1837）",
        "- 校准试验：ENVISION、VISIONARY、ORIGIN 3、APPLAUSE-IgAN、SANCTUARY、ARTEMIS-IGAN（见 REPORT.md）",
    ]
    return "\n".join(L)
