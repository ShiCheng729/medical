"""
小鼠脑组织中一氧化氮 (NO) 信号通路的反应-扩散仿真模型
==========================================================

通路:  谷氨酸 → NMDA 受体 → Ca²⁺ 内流 → Ca²⁺/CaM → nNOS → NO
       NO 自由扩散 (D≈3300 μm²/s) → 组织消耗 + 血红蛋白清除
       NO → sGC → cGMP → PKG  → (a) 突触前释放概率增加 (逆行信使, LTP)
                              → (b) 平滑肌舒张 (神经血管耦联)
       病理: NO + O₂⁻ → ONOO⁻ (过氧亚硝酸盐);  NO 抑制线粒体 COX

单位: 空间 μm, 时间 s, NO/O₂⁻ 为 nM, Ca²⁺/cGMP 为 μM。
参数取自文献的典型数量级 (见 docs/NO_mouse_brain_analysis.md), 用于定性/半定量教学研究,
不是对某一具体实验的拟合。
"""
from dataclasses import dataclass, field, replace
import numpy as np


@dataclass
class Params:
    # --- 空间/时间 ---
    L: float = 240.0          # 方形组织切片边长 μm
    dx: float = 2.0           # 网格 μm
    T: float = 4.0            # 仿真时长 s
    dt: float = 2.5e-4        # 时间步 s  (稳定性: dt < dx²/4D = 3.0e-4)
    # --- NO 扩散与清除 ---
    D: float = 3300.0         # NO 扩散系数 μm²/s (37°C, 水相)
    k_tissue: float = 1.5     # 组织一级消耗 s⁻¹ (半衰期 ~0.5 s)
    k_hb: float = 800.0       # 血管腔内血红蛋白清除 s⁻¹ (强汇)
    # --- 血管 (穿通小动脉) ---
    vessel_xy: tuple = (160.0, 120.0)
    vessel_r: float = 9.0     # 内腔半径 μm (直径 ~18 μm)
    smc_width: float = 4.0    # 平滑肌层厚度 μm
    eNOS_rate: float = 20.0   # 内皮 eNOS 基础产率 nM/s (内皮像素)
    # --- 神经元 nNOS ---
    neurons: tuple = ((70, 120), (90, 95), (90, 145), (110, 120))
    stim_idx: tuple = (0, 1, 2, 3)   # 被刺激的 nNOS 神经元
    src_sigma: float = 4.0    # 源的空间尺度 μm (胞体/树突棘簇)
    P_max: float = 1400.0     # nNOS 最大产率 nM/s (源中心)
    nNOS_scale: float = 1.0   # 1=正常, 0.1=7-NI 抑制, 0=敲除
    # --- Ca²⁺ 与 CaM ---
    Ca0: float = 0.05         # 静息 Ca μM
    tau_Ca: float = 0.2       # Ca 清除 s
    dCa_pulse: float = 0.02   # 每个 NMDA 刺激脉冲 Ca 增量 μM
    K_CaM: float = 0.35       # nNOS 半激活 Ca μM
    n_CaM: float = 3.0
    stim_start: float = 0.5
    stim_dur: float = 1.0
    stim_freq: float = 100.0  # HFS 100 Hz
    sustained_Ca: float = 0.0 # >0 时为病理持续 Ca 平台 (μM)
    # --- sGC / cGMP / PKG ---
    EC50_sGC: float = 2.0     # nM
    n_sGC: float = 1.5
    V_sGC: float = 6.0        # μM/s
    V_PDE: float = 5.0        # μM/s
    K_PDE: float = 1.0        # μM
    K_PKG: float = 0.4        # μM
    # --- 下游效应 ---
    theta_LTP: float = 0.60   # PKG 激活阈值
    eta_LTP: float = 1.4      # 突触增强速率 s⁻¹
    dil_max: float = 0.30     # 最大扩张 30%
    tau_vessel: float = 1.2   # s
    # --- 病理 ---
    superoxide: float = 0.0   # 稳态 O₂⁻ nM
    k_onoo: float = 10.0      # NO+O₂⁻ 速率常数 nM⁻¹s⁻¹ (=1e10 M⁻¹s⁻¹)
    k_onoo_dec: float = 1.0   # ONOO⁻ 分解 s⁻¹
    K_COX: float = 120.0      # NO 抑制 COX 的 Ki nM (常氧; 与 O₂ 竞争)
    snapshot_every: float = 0.05
    warmup: float = 2.0       # 预平衡时间 s (eNOS 张力性 NO、基础 cGMP), 不记录


def stim_times(p: Params):
    n = int(round(p.stim_dur * p.stim_freq))
    return p.stim_start + np.arange(n) / p.stim_freq


def run(p: Params, probes=None, verbose=False):
    """运行仿真, 返回含时间序列与快照的字典。"""
    assert p.dt < p.dx**2 / (4 * p.D), "explicit scheme unstable"
    N = int(round(p.L / p.dx))
    xs = (np.arange(N) + 0.5) * p.dx
    X, Y = np.meshgrid(xs, xs, indexing="xy")

    # 几何
    rv = np.hypot(X - p.vessel_xy[0], Y - p.vessel_xy[1])
    lumen = rv <= p.vessel_r
    endo = (rv > p.vessel_r - p.dx) & (rv <= p.vessel_r + 0.5 * p.dx)
    smc = (rv > p.vessel_r) & (rv <= p.vessel_r + p.smc_width)
    k_field = np.where(lumen, p.k_hb, p.k_tissue) + p.k_onoo * p.superoxide
    k_field = np.where(lumen, p.k_hb, k_field)

    shapes = [np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * p.src_sigma**2))
              for (x, y) in p.neurons]
    nN = len(p.neurons)

    # 探针
    if probes is None:
        probes = {}
    pidx = {k: (int(v[1] // p.dx), int(v[0] // p.dx)) for k, v in probes.items()}

    # 状态
    NO = np.zeros((N, N))
    cG = np.zeros((N, N))
    w = np.zeros((N, N))           # 突触增强 (相对增量)
    onoo = np.zeros((N, N))
    onoo_cum = np.zeros((N, N))
    Ca = np.full(nN, p.Ca0)
    dil = 0.0

    # 初始稳态 cGMP: V_sGC*A0 = V_PDE*c/(K+c)  (A0≈0)
    steps = int(round(p.T / p.dt))
    rec_every = max(1, int(round(0.005 / p.dt)))
    snap_every = max(1, int(round(p.snapshot_every / p.dt)))
    stim = set(np.round(stim_times(p) / p.dt).astype(int).tolist())
    stimulated = np.zeros(nN, bool)
    stimulated[list(p.stim_idx)] = True

    ts, rec = [], {k: [] for k in
                   ["Ca", "nNOS", "dil", "NO_smc", "cG_smc", "NO_max", "onoo_max"]}
    prec = {k: {"NO": [], "cG": [], "w": []} for k in probes}
    snaps_t, snaps_NO, snaps_cG = [], [], []
    peak_NO = np.zeros((N, N))

    inv_dx2 = 1.0 / p.dx**2
    warm = int(round(p.warmup / p.dt))
    for s in range(-warm, steps):
        t = s * p.dt
        # Ca dynamics
        if s >= 0 and s in stim:
            Ca[stimulated] += p.dCa_pulse
        Ca += p.dt * (-(Ca - p.Ca0) / p.tau_Ca)
        if p.sustained_Ca > 0 and t >= p.stim_start:
            Ca[stimulated] = np.maximum(Ca[stimulated], p.sustained_Ca)
        act = Ca**p.n_CaM / (p.K_CaM**p.n_CaM + Ca**p.n_CaM)
        prod = np.zeros((N, N))
        for i in range(nN):
            prod += act[i] * shapes[i]
        prod *= p.P_max * p.nNOS_scale
        prod[endo] += p.eNOS_rate

        # 扩散 (零浓度远场边界)
        Pd = np.pad(NO, 1)
        lap = (Pd[2:, 1:-1] + Pd[:-2, 1:-1] + Pd[1:-1, 2:] + Pd[1:-1, :-2] - 4 * NO) * inv_dx2
        if p.superoxide > 0 and s >= 0:
            r_onoo = p.k_onoo * p.superoxide * NO
            onoo += p.dt * (r_onoo - p.k_onoo_dec * onoo)
            onoo_cum += p.dt * r_onoo
        NO += p.dt * (p.D * lap - k_field * NO + prod)
        np.maximum(NO, 0, out=NO)
        if s >= 0:
            np.maximum(peak_NO, NO, out=peak_NO)

        # sGC → cGMP → PKG
        A = NO**p.n_sGC / (p.EC50_sGC**p.n_sGC + NO**p.n_sGC)
        cG += p.dt * (p.V_sGC * A - p.V_PDE * cG / (p.K_PDE + cG))
        pkg = cG / (cG + p.K_PKG)
        if s >= 0:
            w += p.dt * p.eta_LTP * np.maximum(0.0, pkg - p.theta_LTP)

        pkg_smc = pkg[smc].mean()
        dil += p.dt * (p.dil_max * pkg_smc - dil) / p.tau_vessel

        if s < 0:
            continue
        if s % rec_every == 0:
            ts.append(t)
            rec["Ca"].append(Ca.copy())
            rec["nNOS"].append(act.copy())
            rec["dil"].append(dil)
            rec["NO_smc"].append(NO[smc].mean())
            rec["cG_smc"].append(cG[smc].mean())
            rec["NO_max"].append(NO.max())
            rec["onoo_max"].append(onoo.max())
            for k, (iy, ix) in pidx.items():
                prec[k]["NO"].append(NO[iy, ix])
                prec[k]["cG"].append(cG[iy, ix])
                prec[k]["w"].append(w[iy, ix])
        if s % snap_every == 0:
            snaps_t.append(t)
            snaps_NO.append(NO.copy())
            snaps_cG.append(cG.copy())

    out = {k: np.array(v) for k, v in rec.items()}
    out.update(t=np.array(ts), probes={k: {kk: np.array(vv) for kk, vv in v.items()}
                                       for k, v in prec.items()},
               snaps_t=np.array(snaps_t), snaps_NO=np.array(snaps_NO),
               snaps_cG=np.array(snaps_cG), peak_NO=peak_NO, w=w, cG=cG,
               onoo_cum=onoo_cum, X=X, Y=Y, xs=xs, lumen=lumen, smc=smc,
               cox_inhib=peak_NO / (peak_NO + p.K_COX), params=p)
    return out


# ---------------- 场景 ----------------
BASE = Params()
EXTRA_NEURONS = ((40, 60), (40, 180), (140, 50), (140, 190), (60, 210), (60, 30),
                 (130, 150), (195, 90), (195, 160))


def scenarios():
    allN = BASE.neurons + EXTRA_NEURONS
    return {
        "physio": replace(BASE),                              # 生理: 100 Hz HFS 1 s
        "7NI": replace(BASE, nNOS_scale=0.1),                 # nNOS 抑制剂 7-NI (~90%)
        "nNOS_KO": replace(BASE, nNOS_scale=0.0),             # nNOS⁻/⁻ 小鼠
        "ischemia": replace(BASE, neurons=allN, stim_idx=tuple(range(len(allN))),
                            sustained_Ca=2.0, stim_dur=0.0,
                            superoxide=0.1, k_tissue=0.8,
                            K_COX=40.0),                      # 缺血/兴奋毒性 (低氧, Ki 下降)
    }
