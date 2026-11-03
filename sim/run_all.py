"""运行全部场景并生成图表/汇总表 → ../results/"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation
from no_model import run, scenarios

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

COL = {"physio": "#1f6fd1", "7NI": "#e08a00", "nNOS_KO": "#7a7a7a", "ischemia": "#c62828"}
LAB = {"physio": "生理 (100 Hz HFS 1 s)", "7NI": "7-NI 抑制 nNOS 90%",
       "nNOS_KO": "nNOS⁻/⁻ 小鼠", "ischemia": "缺血 / 兴奋毒性"}
PROBES = {"syn": (72.0, 120.0),        # 紧邻 nNOS 神经元的突触前末梢
          "far": (90.0, 60.0)}         # 距簇中心 60 μm 的旁观突触

res = {k: run(p, PROBES) for k, p in scenarios().items()}
phy = res["physio"]; P = phy["params"]; ext = [0, P.L, 0, P.L]


def decorate(ax, p):
    for i, (x, y) in enumerate(p.neurons):
        ax.add_patch(plt.Circle((x, y), 5, fill=False, ec="w", lw=1.2,
                                ls="-" if i in p.stim_idx else ":"))
    ax.add_patch(plt.Circle(p.vessel_xy, p.vessel_r, fill=False, ec="#ff5252", lw=1.5))
    ax.add_patch(plt.Circle(p.vessel_xy, p.vessel_r + p.smc_width, fill=False, ec="#ff9e80", lw=0.8, ls="--"))
    ax.set_xlim(0, p.L); ax.set_ylim(0, p.L); ax.set_aspect("equal")


norm = LogNorm(0.05, 50)
# ---- 图1: 生理 NO 空间扩散快照 ----
times = [0.45, 0.6, 1.0, 1.5, 2.0, 3.0]
fig, axs = plt.subplots(2, 3, figsize=(12, 8.2), constrained_layout=True)
for ax, tt in zip(axs.flat, times):
    i = np.argmin(abs(phy["snaps_t"] - tt))
    im = ax.imshow(phy["snaps_NO"][i], origin="lower", extent=ext, cmap="inferno", norm=norm)
    decorate(ax, P)
    ax.set_title(f"t = {phy['snaps_t'][i]:.2f} s" + ("  (刺激前)" if tt < 0.5 else ""))
    ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")
fig.colorbar(im, ax=axs, shrink=0.8, label="[NO] (nM, 对数)")
fig.suptitle("图1  生理刺激下 NO 从 nNOS 神经元簇 (白圈) 向周围组织及小动脉 (红圈) 扩散", fontsize=13)
fig.savefig(f"{OUT}/fig1_NO_spatial_physio.png", dpi=130); plt.close(fig)

# ---- 图2: 通路时间过程 ----
t = phy["t"]
fig, axs = plt.subplots(2, 3, figsize=(14, 7.6), constrained_layout=True)
ax = axs[0, 0]
ax.plot(t, phy["Ca"][:, 0], c="k", label="[Ca²⁺] (μM)")
ax.plot(t, phy["nNOS"][:, 0], c=COL["physio"], label="nNOS 激活分数")
ax.axvspan(P.stim_start, P.stim_start + P.stim_dur, color="#ddd", zorder=0)
ax.set_title("① NMDA → Ca²⁺/CaM → nNOS"); ax.legend(); ax.set_xlabel("时间 (s)")
for k in ["physio", "7NI", "nNOS_KO"]:
    r = res[k]
    axs[0, 1].plot(t, r["probes"]["syn"]["NO"], c=COL[k], label=LAB[k])
    axs[0, 2].plot(t, r["probes"]["syn"]["cG"], c=COL[k], label=LAB[k])
    axs[1, 0].plot(t, r["NO_smc"], c=COL[k], label=LAB[k])
    axs[1, 1].plot(t, 100 * (r["dil"] - r["dil"][0]), c=COL[k], label=LAB[k])
    axs[1, 2].plot(t, 100 * r["probes"]["syn"]["w"], c=COL[k], label=LAB[k] + " 近端")
    axs[1, 2].plot(t, 100 * r["probes"]["far"]["w"], c=COL[k], ls="--", lw=1)
axs[0, 1].set_title("② 突触处 [NO] (nM)")
axs[0, 2].set_title("③ 突触前 sGC → [cGMP] (μM)")
axs[1, 0].set_title("④ 小动脉平滑肌处 [NO] (nM)")
axs[1, 1].set_title("⑤ 小动脉扩张 (% 相对刺激前)")
axs[1, 2].set_title("⑥ 突触增强 / LTP (%)  实线近端, 虚线 60 μm 外")
for a in axs.flat:
    a.set_xlabel("时间 (s)"); a.axvspan(P.stim_start, P.stim_start + P.stim_dur, color="#eee", zorder=0)
axs[0, 1].legend(fontsize=8)
fig.suptitle("图2  NO 信号级联的时间过程 (灰色区间为 100 Hz 刺激)", fontsize=13)
fig.savefig(f"{OUT}/fig2_pathway_timecourse.png", dpi=130); plt.close(fig)

# ---- 图3: 距离依赖 (体积传递) ----
cx, cy = 90.0, 120.0
ix = int(cx // P.dx); iy0 = int(cy // P.dx)
ys = phy["xs"][:iy0 + 1][::-1]; dist = cy - ys
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
for k in ["physio", "7NI", "ischemia"]:
    r = res[k]
    axs[0].semilogy(dist, np.maximum(r["peak_NO"][:iy0 + 1, ix][::-1], 1e-3), c=COL[k], label=LAB[k])
    if k != "ischemia": axs[1].plot(dist, 100 * r["w"][:iy0 + 1, ix][::-1], c=COL[k], label=LAB[k])
axs[0].axhline(P.EC50_sGC, c="g", ls=":", label=f"sGC EC50 ≈ {P.EC50_sGC} nM")
axs[0].axhline(P.K_COX, c="m", ls=":", label=f"COX Ki ≈ {P.K_COX:.0f} nM (常氧)")
axs[0].set_ylim(0.05, 300)
lam = np.sqrt(P.D / P.k_tissue)
axs[0].set_title(f"峰值 [NO] 随距离衰减  (扩散长度 √(D/k) ≈ {lam:.0f} μm)")
axs[1].set_title("突触增强随距离 (LTP 的空间扩散, 类似 Schuman & Madison 1994)")
for a in axs:
    a.set_xlabel("距 nNOS 神经元簇中心距离 (μm)"); a.legend(fontsize=8); a.axvspan(0, 25, color="#f2f2f2", zorder=0)
axs[0].set_ylabel("峰值 [NO] (nM)"); axs[1].set_ylabel("Δ 突触强度 (%)")
fig.suptitle("图3  NO 作为“体积传递”信使: 作用范围由扩散与消耗决定", fontsize=13)
fig.savefig(f"{OUT}/fig3_distance_dependence.png", dpi=130); plt.close(fig)

# ---- 图4: 病理 ----
isc = res["ischemia"]; Pi = isc["params"]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
for ax, r, ttl in [(axs[0], phy, "生理: 峰值 [NO]"), (axs[1], isc, "缺血: 峰值 [NO]")]:
    im = ax.imshow(r["peak_NO"], origin="lower", extent=ext, cmap="inferno", norm=LogNorm(0.1, 100))
    decorate(ax, r["params"]); ax.set_title(ttl)
fig.colorbar(im, ax=axs[:2], label="nM", shrink=0.85)
im = axs[2].imshow(isc["onoo_cum"], origin="lower", extent=ext, cmap="viridis")
decorate(axs[2], Pi); axs[2].set_title("缺血: 累计 ONOO⁻ 生成 (nM, 3.5 s)")
fig.colorbar(im, ax=axs[2], shrink=0.85)
keys = ["physio", "7NI", "nNOS_KO", "ischemia"]
cox = [100 * res[k]["cox_inhib"][int(120 // P.dx), int(90 // P.dx)] for k in keys]
axs[3].bar(range(4), cox, color=[COL[k] for k in keys])
axs[3].set_xticks(range(4)); axs[3].set_xticklabels(["生理", "7-NI", "KO", "缺血"])
axs[3].set_ylabel("线粒体 COX 抑制 (%)"); axs[3].set_title("簇中心峰值 COX 抑制")
for i, v in enumerate(cox):
    axs[3].text(i, v + 1, f"{v:.1f}%", ha="center")
fig.suptitle("图4  病理状态: NO 过量 + 超氧阴离子 → 过氧亚硝酸盐与线粒体抑制", fontsize=13)
fig.savefig(f"{OUT}/fig4_pathology.png", dpi=130); plt.close(fig)

# ---- 动画 ----
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), constrained_layout=True)
ims = []
for ax, r, ttl in [(axs[0], phy, "生理"), (axs[1], isc, "缺血")]:
    ims.append(ax.imshow(r["snaps_NO"][0], origin="lower", extent=ext, cmap="inferno", norm=LogNorm(0.05, 60)))
    decorate(ax, r["params"]); ax.set_title(ttl)
fig.colorbar(ims[0], ax=axs, label="[NO] nM", shrink=0.8)
ttl = fig.suptitle("")
def upd(i):
    for im, r in zip(ims, (phy, isc)):
        im.set_data(r["snaps_NO"][i])
    ttl.set_text(f"小鼠脑片 NO 扩散  t = {phy['snaps_t'][i]:.2f} s")
    return ims
animation.FuncAnimation(fig, upd, frames=len(phy["snaps_t"]), interval=80).save(
    f"{OUT}/NO_diffusion.gif", writer="pillow", fps=12, dpi=72)
plt.close(fig)

# ---- 汇总 ----
summary = {}
for k, r in res.items():
    b = int(120 // P.dx), int(90 // P.dx)
    summary[k] = {
        "峰值NO_突触(nM)": round(float(r["probes"]["syn"]["NO"].max()), 2),
        "峰值cGMP_突触(μM)": round(float(r["probes"]["syn"]["cG"].max()), 2),
        "LTP_近端(%)": round(100 * float(r["probes"]["syn"]["w"][-1]), 1) if k != "ischemia" else "不适用(病理性PKG过度激活)",
        "LTP_60μm(%)": round(100 * float(r["probes"]["far"]["w"][-1]), 1) if k != "ischemia" else "不适用",
        "平滑肌峰值NO(nM)": round(float(r["NO_smc"].max()), 2),
        "血管最大扩张(%)": round(100 * float((r["dil"] - r["dil"][0]).max()), 1),
        "COX抑制_簇中心(%)": round(100 * float(r["cox_inhib"][b]), 1),
        "累计ONOO-_最大(nM)": round(float(r["onoo_cum"].max()), 1),
    }
json.dump(summary, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
