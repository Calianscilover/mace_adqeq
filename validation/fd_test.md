# 生产 MD 前的验证清单（MACE-adQEq）

> 目的：在正式跑生产 MD 之前，确认 QEq 的力、能量和实际使用的求解路径都可靠。
> 前提：[PR #1](https://github.com/Calianscilover/mace_adqeq/pull/1)（matrix 求解器 Hessian 修复）已合并或已检出 `dev/1001`。
> 脚本位于 `validation/`。模型和结构文件不入库，按集群上的布局放在 `validation/jax_model/` 和 `validation/pair_revised_equilibrated.extxyz`，或用 `--params / --config / --structure` 指定。

**已确定的约定**

| 项 | 决定 |
|---|---|
| QEq 精度 | `float64`（`calculator.py` 中为 `"dtype": "float64"`） |
| 总电荷 | 只用于电中性体系；`MACEJAXQEqCalculator` 固定按 `total_charge=0` 求解 |
| 偶极修正轴 | z 轴（`dipole_axis=2`），代码、脚本和 README 全部统一 |

---

## 0. 为什么有限差分测试不够

`fd_force_test.py` 通过是**必要条件，不是充分条件**。它只证明：在测试用的那套配置下，解析力等于**当前实现的**能量函数的负梯度。

| 能证明 | 不能证明 |
|---|---|
| 冻结路径：电荷是约束极小点，包络定理成立，PME / 高斯修正 / onsite / 偶极项对坐标的自动微分正确 | 能量函数本身是否正确（单位、修正项约定、PME 收敛、参数是否准确、与短程 MACE 是否重复计数） |
| 完整路径：参数响应（JAX 中 MLP 的 VJP，加上 PyTorch 中 MACE 的 VJP）拼接正确，cotangent 顺序和 dtype 正确 | 生产配置（hybrid + CG warm start、恒电势）是否同样正确 |
| | 能量面在 cutoff、电极原子重新归属处的跳变，以及 MD 中的能量守恒 |

之前 Hessian 在 q = 0 处退化的 bug 能被有限差分抓到，是因为它破坏了驻点条件。如果电荷仍是某个**错误**能量的极小点，力和能量依然自洽，有限差分照样通过。

---

## 1. 前置准备

- [ ] 合并 PR #1，或在 `dev/1001` 上进行下面所有测试。
- [x] 把 `fd_force_test.py`、`qeq_gradient_diagnose.py`、`pg_solver_check.py` 提交进仓库（`validation/`）。
- [ ] 作废 `fd_results/20261001_141902/`，不再引用：它是修复前跑的（`energy_whole = +75.42 eV`，修复后应为 −156.09 eV），用新的判定逻辑检查会有 12/14 项 FAIL。
- [x] 确定生产用的 QEq dtype：`float64`。

---

## 2. 测试 1：有限差分梯度一致性

### 2.1 参数固定的诊断（已通过，修改代码后需重跑）

```bash
python validation/qeq_gradient_diagnose.py --device cpu
```

合格标准：T1 `rms_projected_grad < 1e-6` eV/e；T3 `charge_response < 1e-5` eV/Å；T5 `n_negative = 0`。

### 2.2 力与能量的一致性（constQ 和恒电势）

这一步只回答“力算得对不对”。脚本对选中的原子分量和两个全局方向做中心差分，同时比较两种解析力：冻结力（χ/J/η 固定）和完整力（含 MACE + MLP 的参数响应）。所有位移构型都沿用参考构型的 QEq 邻居对，避免 6 Å cutoff 处的能量台阶混入差分；MACE 自己的邻居表有光滑包络，照常重建。

默认配置是 CPU、float64 MACE、float64 QEq、matrix 求解器。**不要在 GPU 上做这项验证**：GPU 上 MACE 的结果不确定，重复计算的能量差约 5e-4 eV，会淹没参数响应项。

```bash
python validation/fd_force_test.py --output validation/fd_results/constQ_frame0
python validation/fd_force_test.py --const-potential --output validation/fd_results/constP_frame0
# 两种模式再各取一帧：--frame 1，并相应修改 --output
```

脚本最后输出判定结果，任何一项失败时退出码为 1：

| 判据 | 含义 | 默认阈值 |
|---|---|---|
| 冻结力 | 最佳步长下 \|frozen − fd_frozen\| 的平均值，分量和全局方向各判一次 | ≤ 1e-5 eV/Å（`--frozen-tol`） |
| 参数响应 | 加上参数响应项后，消除了冻结力与完整差分之间偏差的比例，即 1 − \|whole − fd_full\| / \|frozen − fd_full\| | ≥ 90%（`--explained`） |
| 电极归属（仅恒电势） | 所有位移构型的电极原子集合与参考构型相同 | 0 次变化 |

参数响应不用绝对误差判定：float32 参数 MLP 会给完整能量带来约 7e-4 eV 的舍入噪声，\|whole − fd_full\| 只能随步长按 1/δ 下降（δ = 0.01 时约 0.03 eV/Å），绝对阈值原则上达不到。`test_107`（constQ）中，冻结力误差为 1.7e-7 eV/Å，参数响应消除了 96%（分量）和 99.5%（全局方向）的偏差。

- [ ] constQ：两帧都输出 `forces consistent with the energy: YES`。
- [ ] 恒电势：两帧都输出 `YES`。如果只有“电极归属”失败，说明位移让某个表面 Zn 的配位数越过了阈值，属于电极判定的跳变而不是力公式的错误，应考虑把 `minimum_coordination` 从 8 降到 6。

---

## 3. 测试 2：生产求解路径（hybrid + CG）

有限差分测试用的是 matrix 求解器；MD 每帧实际走的是 hybrid 模式：第一帧用 matrix，之后从上一帧的电荷 warm start，用 CG 求解；只有 CG 误差超过 `max(10 × pg_tol, 1e-5)`（默认 1e-2）时才回退到 matrix。

脚本沿一条轨迹逐帧调用同一个 hybrid 模型，调用方式与 `calculator.py` 完全相同（`calculate(atoms, predictor=...)`）：每帧重新预测参数，恒电势下重新判定电极，比较的是含参数响应的完整力。每一帧都与同一构型上 float64 matrix 求解器的结果对比。轨迹默认是从 `--structure` 出发、每步每个坐标随机位移 0.01 Å 的 50 帧随机游走；有了真实 MD 轨迹后，可以用 `--trajectory` 直接读入。

```bash
python validation/pg_solver_check.py --output validation/pg_constQ.json
python validation/pg_solver_check.py --const-potential --output validation/pg_constP.json
# 误差会不会随 warm start 累积：加长轨迹
python validation/pg_solver_check.py --frames 200 --output validation/pg_constQ_long.json
```

判定（第 0 帧两边都是 matrix，不计入；任何一项失败时退出码为 1）：

| 判据 | 含义 | 默认阈值 |
|---|---|---|
| 输入相同 | 两次调用得到的 χ/J/η 完全一致，否则比较的是预测器的噪声而不是求解器。必须用 `--device cpu` | ≤ 1e-10（`--parameter-tol`） |
| 没有回退 | 第 1 帧起全部是 `projected_gradient` | 0 帧 `matrix_fallback` |
| 力 | 每一帧的 max \|F_hybrid − F_matrix\| | ≤ 1e-4 eV/Å（`--force-tol`） |

最后输出 `production solver matches the matrix solver: YES/NO`。汇总里另外列出三项供复核，它们不参与判定：

- 跑满 `--pg-maxiter` 的帧：CG 没收敛到 `pg_tol`，但误差不到 1e-2，会被直接采用而不回退。逐帧表里这些帧的 `max|dF|` 是最需要看的。
- 电极重新归属的帧（仅恒电势）：该帧 χ 跳变 10 eV，warm start 的起点离解最远。
- 逐帧表的 `max|Pg|`：在 float64 下、用求解器实际看到的 χ（含电极偏置）重新计算的投影梯度，是对 CG 自报误差 `pg_err` 的独立核对。

- [ ] constQ：50 帧输出 `YES`。
- [ ] 恒电势：50 帧输出 `YES`；如果有电极重新归属的帧，确认它们也满足力的阈值。
- [ ] 200 帧的长轨迹：后段的 `max|dF|` 没有比前段明显增大（误差不随 warm start 累积）。

---

## 4. 测试 3：能量本身是否正确

有限差分测试对这一类错误完全不敏感。

- [ ] **Ewald/PME 收敛**：分别改变 `pme_grid`（×1.5）、`kappa`、`cutoff`，QEq 能量与电荷的变化应 ≪ 目标精度（例如 < 1 meV/atom，Δq < 1e-3 e）。
- [ ] **独立参考**：在一个小体系（几十个原子）上，用直接 Ewald 求和（或其他 QEq 实现）计算同一组 χ/J/η 下的电荷和能量，与 `JAXQEqModel` 对比，覆盖高斯屏蔽修正、自能项和 slab 偶极修正。
- [ ] **偶极修正轴与 checkpoint 一致**：现在所有地方都用 z 轴（`2`），短程标签脚本 `qeq_extract_new` 也用默认的 z 轴。但旧 README 记录“原训练代码实际写的是 `positions[:, 1]`”。需要确认 QEq 参数 MLP 训练时用的是 z 轴；如果当时用的是 y 轴，就要用 z 轴重训参数 MLP（并入第 5 节）。
- [ ] **物理合理性**：Σq = 0；μ 落在 χ 范围内；电荷分布合理（例如水中 O/H、SO₄²⁻、Zn²⁺ 的符号和量级）；ΣF ≈ 0。

---

## 5. 测试 4：重新生成标签并重训

PR #1 之前的 matrix 求解器给出的电荷是错的，依赖它的数据都要重做。

- [ ] 用修复后的 QEq 重新生成 `E_short = E_DFT − E_QEq` 和 `F_short`（`qeq_extract_new.py`）。
- [ ] 如果 QEq 参数 MLP 的训练调用过 `solve_charges_matrix`，或者训练时偶极修正用的不是 z 轴，需要重训。
- [ ] 重训短程 MACE。
- [ ] 在留出的测试集上报告总能量和力的误差（MAE/RMSE），并与纯 MACE 基线对比。
- [ ] 重新计算基于旧电荷的分析数据（电荷统计、完整力与冻结力的比例等）。

---

## 6. 测试 5：NVE 能量守恒与能量面的不连续

单帧加小位移的有限差分几乎碰不到下面这些跳变，只有在 MD 中才会暴露。

- [ ] **NVE**：用生产配置（hybrid + CG、float64、恒电势开/关）、生产时间步长，跑至少 10–50 ps。统计总能量漂移（meV/atom/ps）和涨落，并与相同设置下纯短程 MACE 的漂移对比。
- [ ] **cutoff 截断**：实空间 PME 项和高斯修正在 cutoff 处直接截断。每对原子跨越 cutoff 时，能量跳变约为
  `ΔE ≈ 14.40 · q_i q_j · [erfc(κ r_c) − erfc(r_c / (√2 η_ij))] / r_c` eV。
  `test_106` / `test_107` 的有限差分测试（当时每个位移构型都重建邻居表）已经证实存在这种台阶：108 个分量行中有 33 行跨越 cutoff，单次跨越约 2e-6 到 2.5e-4 eV。需要在 NVE 中量化累积效应；如果不可忽略，考虑加大 cutoff 或加平滑切换函数。
- [ ] **电极原子重新归属**（恒电势）：`determine_chi` 是离散判断，归属改变时 χ 突变 +10 或 −2 eV，能量跳变约为“偏置 × 该原子电荷”。在 MD 中记录每一步的电极原子集合，统计改变的次数；如果有改变，考虑用 `forced_bottom_indices` / `forced_upper_indices` 固定电极原子。
- [ ] 记录每步的 `qeq_solver`、`qeq_pg_iterations`、`qeq_pg_error`，统计 `matrix_fallback` 的频率和 CG 迭代数的变化趋势。
- [ ] 检查电荷有没有失控（例如 max|q| 随时间增长），检查结构稳定性（RDF、密度），有条件时与 AIMD 对比。

---

## 7. 生产前的配置确认

- [x] **QEq dtype**：`float64`。
- [x] **总电荷**：只跑电中性体系，calculator 固定 `total_charge=0` 可以接受。如果以后要跑带电体系，需要让 calculator 读取 `atoms.info["total_charge"]`。
- [x] **偶极修正轴**：全部统一为 z 轴（`2`）。剩下的“checkpoint 训练时用的是哪个轴”见第 4 节。
- [ ] **`max_pairs`**：calculator 中为 30000，测试中为 50000。确认 MD 中的最大原子对数量不会超过容量，否则会抛出 `ValueError`。
- [ ] **MACE 设备与精度**：参数预测器在 GPU float32 上是非确定性的（能量噪声约 5e-4 eV），确认它对 NVE 漂移的贡献可以接受。
- [ ] **恒电势参数**：`bottom_chi_bias=10.0`、`upper_chi_bias=-2.0`、`coordination_cutoff=3.5`、`minimum_coordination=8` 是否就是生产要用的值。

---

## 完成标准

第 2 到第 6 节全部打勾，第 7 节的配置都确定之后，才开始生产 MD。
