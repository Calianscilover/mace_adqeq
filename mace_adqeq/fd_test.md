# 生产 MD 前的验证清单（MACE-adQEq）

> 目的：在正式跑生产 MD 之前，确认 QEq 的力、能量和实际使用的求解路径都可靠。
> 前提：[PR #1](https://github.com/Calianscilover/mace_adqeq/pull/1)（matrix 求解器 Hessian 修复）已合并或已检出 `dev/1001`。

---

## 0. 为什么有限差分测试不够

`fd_force_test.py` 通过是**必要条件，不是充分条件**。它只证明：在测试用的那套配置下，解析力等于**当前实现的**能量函数的负梯度。

| 能证明 | 不能证明 |
|---|---|
| 冻结路径：电荷是约束极小点，包络定理成立，PME / 高斯修正 / onsite / 偶极项对坐标的自动微分正确 | 能量函数本身是否正确（单位、修正项约定、PME 收敛、参数是否准确、与短程 MACE 是否重复计数） |
| 完整路径：参数响应（JAX 中 MLP 的 VJP，加上 PyTorch 中 MACE 的 VJP）拼接正确，cotangent 顺序和 dtype 正确 | 生产配置（hybrid + CG warm start、float32、恒电势）是否同样正确 |
| | 能量面在 cutoff、电极原子重新归属处的跳变，以及 MD 中的能量守恒 |

之前 Hessian 在 q = 0 处退化的 bug 能被有限差分抓到，是因为它破坏了驻点条件。如果电荷仍是某个**错误**能量的极小点，力和能量依然自洽，有限差分照样通过。

---

## 1. 前置准备

- [ ] 合并 PR #1，或在 `dev/1001` 上进行下面所有测试。
- [ ] 把 `fd_force_test.py`、`qeq_gradient_diagnose.py`、`pg_solver_check.py` 提交进仓库（PR #1 的描述提到了它们，但分支里没有）。
- [ ] 作废 `fd_results/20261001_141902/`：它是修复前跑的（`energy_whole = +75.42 eV`，修复后应为 −156.09 eV）。
- [ ] 决定生产用的 QEq dtype，写进 `calculator.py`（见第 7 节第 1 条）。

---

## 2. 测试 1：有限差分梯度一致性

### 2.1 参数固定的诊断（已通过，修改代码后需重跑）

```bash
python qeq_gradient_diagnose.py --device cpu --output qeq_gradient_diagnose.json
```

合格标准：T1 `rms_projected_grad < 1e-6` eV/e；T3 `charge_response < 1e-5` eV/Å；T5 `n_negative = 0`。

### 2.2 完整路径（含参数响应）

**必须在 CPU 上用 float64 MACE**。修复前那次运行中，GPU 上 float32 MACE 在同一构型下重复计算 3 次，能量相差 4.5e-4 eV。这在 δ = 0.001 Å 时会带来约 0.23 eV/Å 的差分噪声，与参数响应项本身同量级。

```bash
python fd_force_test.py --device cpu --mace-dtype float64 --dtype float64
python fd_force_test.py --device cpu --mace-dtype float64 --dtype float64 --const-potential
```

合格标准（建议值）：

- [ ] `repeat_spread` 中 `full` 和 `frozen` 都在 1e-10 eV 量级（确定性）。
- [ ] `frozen` 与 `fd_frozen` 的平均误差 ≤ 1e-5 eV/Å（修复后诊断给出 2.5e-7）。
- [ ] `whole` 与 `fd_full` 的误差随 δ 按约 δ² 下降，直到噪声底 `≈ 能量噪声 / δ`；在最佳步长下 ≤ 1e-3 eV/Å，并且远小于 `|whole − frozen|`（即 `response_fd_minus_analytic_mae ≪ response_analytic_mean_abs`）。
- [ ] 能量扫描的 `fit_full` 与 `whole`、`fit_frozen` 与 `frozen` 在噪声范围内一致。
- [ ] 开启恒电势时 `electrode_changes` 为空。
- [ ] 至少再取 2–3 帧（`--frame`），包括一帧离子靠近电极的构型。

### 2.3 `fd_force_test.py` 待改进

- [ ] 增加 `--solver-mode {matrix,hybrid}` 和 `--pg-method`，目前测试中写死 `solver_mode="matrix"`。
- [ ] 自动输出 PASS/FAIL：按上面的标准判断，而不是只打印数字。
- [ ] 输出 `ΣF`（力的总和），检查平移不变性（应约为 0）。

---

## 3. 测试 2：生产求解路径（hybrid + CG）

有限差分测试用的是 matrix 求解器；MD 每帧实际走的是 hybrid 模式下 warm start 的 CG。

```bash
python pg_solver_check.py --device cpu --pg-method cg --frames 20 --output pg_solver_cg.json
python pg_solver_check.py --device cpu --pg-method cg --frames 20 --const-potential --output pg_solver_cg_cp.json
```

- [ ] 所有帧都是 `projected_gradient`，没有 `matrix_fallback`。
- [ ] float64：`max_force_error < 1e-4` eV/Å。
- [ ] float32：确认误差可以接受。上次结果中每帧能量误差最大 3.2e-2 eV、力误差最大 1e-2 eV/Å，而且这部分误差来自 float32 能量函数本身，与求解器无关。
- [ ] 用更长的随机游走（`--frames` 加大），检查误差会不会随 warm start 累积。

---

## 4. 测试 3：能量本身是否正确

有限差分测试对这一类错误完全不敏感。

- [ ] **Ewald/PME 收敛**：分别改变 `pme_grid`（×1.5）、`kappa`、`cutoff`，QEq 能量与电荷的变化应 ≪ 目标精度（例如 < 1 meV/atom，Δq < 1e-3 e）。
- [ ] **独立参考**：在一个小体系（几十个原子）上，用直接 Ewald 求和（或其他 QEq 实现）计算同一组 χ/J/η 下的电荷和能量，与 `JAXQEqModel` 对比，覆盖高斯屏蔽修正、自能项和 slab 偶极修正。
- [ ] **偶极修正轴**：代码默认 `dipole_axis=2`（z），README 写的是 `1`（y）。确认与 slab 法向（非周期方向）一致，并更新 README。
- [ ] **物理合理性**：Σq = 总电荷；μ 落在 χ 范围内；电荷分布合理（例如水中 O/H、SO₄²⁻、Zn²⁺ 的符号和量级）；ΣF ≈ 0。

---

## 5. 测试 4：重新生成标签并重训

PR #1 之前的 matrix 求解器给出的电荷是错的，依赖它的数据都要重做。

- [ ] 用修复后的 QEq 重新生成 `E_short = E_DFT − E_QEq` 和 `F_short`（`qeq_extract_new.py`）。
- [ ] 如果 QEq 参数 MLP 的训练调用过 `solve_charges_matrix`，需要重训。
- [ ] 重训短程 MACE。
- [ ] 在留出的测试集上报告总能量和力的误差（MAE/RMSE），并与纯 MACE 基线对比。
- [ ] 重新计算基于旧电荷的分析数据（电荷统计、完整力与冻结力的比例等）。

---

## 6. 测试 5：NVE 能量守恒与能量面的不连续

单帧加小位移的有限差分几乎碰不到下面这些跳变，只有在 MD 中才会暴露。

- [ ] **NVE**：用生产配置（hybrid + CG、生产 dtype、恒电势开/关）、生产时间步长，跑至少 10–50 ps。统计总能量漂移（meV/atom/ps）和涨落，并与相同设置下纯短程 MACE 的漂移对比。
- [ ] **cutoff 截断**：实空间 PME 项和高斯修正在 cutoff 处直接截断。每对原子跨越 cutoff 时，能量跳变约为
  `ΔE ≈ 14.40 · q_i q_j · [erfc(κ r_c) − erfc(r_c / (√2 η_ij))] / r_c` eV。
  在一个典型构型上统计 cutoff 附近所有原子对的 |ΔE|。如果不可忽略，考虑加大 cutoff 或加平滑切换函数。
- [ ] **电极原子重新归属**（恒电势）：`determine_chi` 是离散判断，归属改变时 χ 突变 +10 或 −2 eV，能量跳变约为“偏置 × 该原子电荷”。在 MD 中记录每一步的电极原子集合，统计改变的次数；如果有改变，考虑用 `forced_bottom_indices` / `forced_upper_indices` 固定电极原子。
- [ ] 记录每步的 `qeq_solver`、`qeq_pg_iterations`、`qeq_pg_error`，统计 `matrix_fallback` 的频率和 CG 迭代数的变化趋势。
- [ ] 检查电荷有没有失控（例如 max|q| 随时间增长），检查结构稳定性（RDF、密度），有条件时与 AIMD 对比。

---

## 7. 生产前的配置确认

- [ ] **QEq dtype**：本地 `calculator.py` 是 `"dtype": "float32"`，`dev/1001` 是 `"float64"`。PR #1 的测试显示 float64 在该体系上不比 float32 慢，力误差小约 500 倍。二选一并提交。
- [ ] **总电荷**：`calculator.py` 中写死了 `total_charge=0.0`，不读取 `atoms.info["total_charge"]`（README 示例里设置了这个字段）。非电中性体系需要修改。
- [ ] **`max_pairs`**：calculator 中为 30000，测试中为 50000。确认 MD 中的最大原子对数量不会超过容量，否则会抛出 `ValueError`。
- [ ] **MACE 设备与精度**：参数预测器在 GPU float32 上是非确定性的（能量噪声约 5e-4 eV），确认它对 NVE 漂移的贡献可以接受。
- [ ] **恒电势参数**：`bottom_chi_bias=10.0`、`upper_chi_bias=-2.0`、`coordination_cutoff=3.5`、`minimum_coordination=8` 是否就是生产要用的值。

---

## 完成标准

第 2 到第 6 节全部打勾，第 7 节的配置都确定并提交之后，才开始生产 MD。
