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

### 2.2 完整路径（含参数响应）

脚本默认就是验证配置：CPU、float64 MACE、float64 QEq、matrix 求解器、z 轴偶极修正。**不要在 GPU 上用 float32 MACE 做这项验证**：修复前那次运行中，同一构型重复计算 3 次能量相差 4.5e-4 eV，在 δ = 0.001 Å 时会带来约 0.23 eV/Å 的差分噪声，与参数响应项本身同量级。

```bash
python validation/fd_force_test.py
python validation/fd_force_test.py --const-potential
python validation/fd_force_test.py --frame 1      # 再取 2–3 帧，包括离子靠近电极的构型
```

脚本最后会打印 PASS/FAIL 表，并写入 `fd_summary.json` 的 `checks` 和 `passed` 字段；任何一项失败时退出码为 1。

**cutoff 跨越的处理**：实空间 PME 项和高斯修正在 cutoff（6 Å）处直接截断，某个原子对在 ±δ 两个构型之间跨过 cutoff 时，能量会多出一个不随 δ 减小的台阶。`--neighbor-list` 有两种模式：

- `fixed`（默认）：所有位移构型都沿用参考构型的 QEq 邻居对，能量对坐标光滑，所有行都参与统计。脚本仍会统计“重建邻居表时会跨越 cutoff 的行数”，只作参考。这是梯度一致性测试应该用的模式。
- `rebuild`：与 MD 一样每次重建邻居表。脚本把邻居对发生变化的行标记为 `cutoff_crossed`，单独列出能量台阶（`implied_jump_frozen` / `implied_jump_full`），并从误差统计中排除；能量扫描的拟合对每次变化加一个台阶项。用来测量 cutoff 台阶本身。

`test_107`（`rebuild`）的结果：108 个分量行中有 33 行跨越 cutoff，台阶为 2e-6 到 2.5e-4 eV；排除后冻结路径误差为 1.7e-7 eV/Å。全局方向的 8 行全部跨越 cutoff（原子 446 与 H316 相距 5.99997 Å），相关检查只能 SKIP，所以改为默认 `fixed`。

**noise-limited**：如果某项 FAIL 的数值不超过实测能量噪声对应误差的 3 倍，会标注 `noise-limited`，表示在当前噪声下无法分辨，不代表梯度有错。完整路径受 float32 参数 MLP 的舍入噪声（约 5e-4 到 1e-3 eV）限制，`whole` 相关的检查通常是这种情况。

默认阈值（可用命令行参数修改）：

| 检查 | 默认阈值 | 参数 |
|---|---|---|
| 同一构型重复计算的能量差（确定性） | ≤ 1e-8 eV | `--repeat-tol` |
| `frozen` vs `fd_frozen`，最佳步长下的平均误差 | ≤ 1e-5 eV/Å | `--frozen-tol` |
| `whole` vs `fd_full`，最佳步长下的平均误差 | ≤ 1e-3 eV/Å | `--whole-tol` |
| 参数响应的差分误差 / 参数响应本身 | ≤ 1% | `--response-ratio` |
| 能量扫描拟合斜率 vs 解析力投影 | ≤ 1e-3 eV/Å | `--whole-tol` |
| 力的总和 \|ΣF\|（平移不变性；PME 网格离散会引入少量偏差） | ≤ 1e-2 eV/Å | `--net-force-tol` |
| 恒电势下电极原子归属的变化次数 | 0 | — |

人工复核：

- [ ] 冻结路径的检查全部 PASS（排除 cutoff 跨越的行之后）。
- [ ] `whole` 相关的 FAIL 全部是 `noise-limited`；参数响应方向的检查（`directional: response error / |response|`）在 1% 以内。
- [ ] 三条命令（含至少 2 帧不同构型）都满足以上两条。

### 2.3 生产噪声（仅供参考，不要求 PASS）

```bash
python validation/fd_force_test.py --solver-mode hybrid
python validation/fd_force_test.py --device cuda --mace-dtype float32 --solver-mode hybrid
```

用来量化生产配置下的能量噪声和力误差。脚本会提示“不是验证配置，阈值可能不适用”。

---

## 3. 测试 2：生产求解路径（hybrid + CG）

有限差分测试默认用 matrix 求解器；MD 每帧实际走的是 hybrid 模式下 warm start 的 CG（脚本默认 `--pg-method cg`）。

```bash
python validation/pg_solver_check.py --device cpu --frames 20 --output pg_solver_cg.json
python validation/pg_solver_check.py --device cpu --frames 20 --const-potential --output pg_solver_cg_cp.json
```

- [ ] 所有帧都是 `projected_gradient`，没有 `matrix_fallback`。
- [ ] float64：`max_force_error < 1e-4` eV/Å。
- [ ] 用更长的随机游走（`--frames` 加大），检查误差会不会随 warm start 累积。

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
  `test_106` 的有限差分测试已经证实存在这种台阶：单次跨越约 2e-6 到 2.5e-4 eV（见 `fd_force_test.py` 输出的 cutoff 跨越表）。需要在 NVE 中量化累积效应；如果不可忽略，考虑加大 cutoff 或加平滑切换函数。
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
