# medical

## 一氧化氮 (NO) 小鼠脑神经通路仿真

- 分析报告：[docs/NO_mouse_brain_analysis.md](docs/NO_mouse_brain_analysis.md)
- 模型：`sim/no_model.py` — NMDA→Ca²⁺/CaM→nNOS→NO 反应-扩散→sGC/cGMP/PKG→LTP 与血管扩张；病理 ONOO⁻ 与 COX 抑制
- 运行：`pip install numpy matplotlib && python3 sim/run_all.py`，结果写入 `results/`
