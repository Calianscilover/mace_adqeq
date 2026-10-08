# 通用 JAX-QEq 与 MACE 的组合 ASE Calculator

## 1. 当前实现的目标

本目录实现的是一个专用的 MACE + ADQEq ASE Calculator：

```text
ASE Atoms
   ├── 官方 MACECalculator ──────> E_short, F_short
   └── JAXQEqModel ──────────────> E_qeq, F_qeq, q, chi, J, eta
                         │
                         └── E = E_short + E_qeq
                             F = F_short + F_qeq
```

MACE 没有被嵌入 JAX，也不需要参与 QEq 求解。`MACEJAXQEqCalculator` 直接使用 `from mace.calculators import MACECalculator` 加载短程模型，并在同一个 ASE Calculator 中调用独立的 JAX-QEq 模型，最后对两部分能量和力求和。本实现不再提供通用 short-range Calculator 注入接口。

## 2. 文件结构

- `mace_adqeq/qeq.py`：恢复 Flax `.msgpack`，从 `.json` 重建 ACSF 与多头网络，并完成 PME-QEq、KKT 电荷约束求解、能量和力计算。
- `mace_adqeq/calculator.py`：直接组合官方 `MACECalculator` 与 JAX-QEq 的专用 ASE Calculator。
- `run_singlepoint.py`：组合模型单点计算，写出能量、力和逐原子电荷。
- `run_qeq_singlepoint.py`：不加载 MACE，独立检查 QEq 及指定原子组中的负电荷。
- `example_md.py`：ASE Langevin MD 最小示例。
- `tests/test_calculator.py`：不依赖真实模型的组合逻辑测试。

当前默认模型路径对应：

- 短程模型：`../interface.model`
- QEq 参数：`/path/to/qeq_multihead_params.msgpack`
- QEq 配置：`/path/to/qeq_multihead_config.json`

## 3. QEq 推理过程

`JAXQEqModel` 复现了 `main_multi.py` 的推理路径：

1. 根据 JSON 中的元素、截断半径和 G2/G4 参数生成 ACSF。
2. 拼接元素 one-hot，得到当前 checkpoint 所需的 68 维逐原子特征。
3. Flax 多头 MLP 预测 `delta_chi`、`delta_log_hardness` 和 `delta_log_eta`。
4. 与元素基线组合得到电负性 `chi`、硬度 `J` 和高斯宽度 `eta`。
5. 用 DMFF PME、短程高斯修正、原位 QEq 能量和 slab 偶极修正构造长程能量。
6. 对电荷二次型构造 KKT 方程，在 `sum(q_i) = Q_total` 约束下直接求解电荷。
7. 对坐标求梯度得到 QEq 力，最后与短程模型结果相加。

为保持现有 checkpoint 行为，当前 `eta` 仍使用元素基线；虽然网络包含 eta head，但原训练脚本实际没有启用其预测值。slab 偶极修正统一使用 z 轴（`dipole_axis=2`，即电极法向的非周期方向），`JAXQEqModel` 和所有脚本的默认值都是 `2`。

QEq 默认使用 `solver_mode="hybrid"`：第一帧通过迭代
Newton-KKT 矩阵法获得严格约束的自洽电荷，之后每帧以上一帧电荷为初值，
使用投影共轭梯度（`pg_method="cg"`，默认）或投影 LBFGS 求解。原子数、元素顺序或总电荷发生变化时会自动回到矩阵法。
如需每帧都使用矩阵法，可设置：

```python
qeq_options={"solver_mode": "matrix"}
```

`calculator.results["qeq_solver"]` 会记录当前帧使用的求解器；投影法还会提供
`qeq_pg_iterations` 和 `qeq_pg_error`。调用
`calculator.reset_charge_state()` 可手动清除电荷缓存。

## 4. 基本用法

在 `mace-adqeq` 目录中运行：

```bash
python run_singlepoint.py \
  --structure ../STRU-2.pdb \
  --mace-model ../interface.model \
  --qeq-params /path/to/qeq_multihead_params.msgpack \
  --qeq-config /path/to/qeq_multihead_config.json \
  --total-charge 0 \
  --mace-device cuda
```

`MACEJAXQEqCalculator` 固定按总电荷 0 求解，只适用于电中性体系；`atoms.info["total_charge"]` 和 `run_singlepoint.py` 的 `--total-charge` 不会改变 calculator 的求解结果。

正式生产 MD 前的验证步骤见 `validation/fd_test.md`。验证通过后，用主动学习补充训练数据，见 `active_learning/active_learning.md`。

在当前 `py3.9` 环境中可先独立验证 QEq，并列出前 93 个 Zn 中电荷为负的原子序号：

```bash
conda run -n py3.9 python run_qeq_singlepoint.py \
  --structure ../STRU-2.pdb \
  --total-charge 0 \
  --report-first 93
```

输出序号采用从 1 开始的编号，便于和可视化软件中的原子编号对应。

Python 中直接使用：

```python
from ase.io import read
from mace_adqeq import MACEJAXQEqCalculator

atoms = read("../STRU-2.pdb")
atoms.info["total_charge"] = 0.0
atoms.calc = MACEJAXQEqCalculator(
    mace_model_path="../interface.model",
    qeq_params_path="/path/to/qeq_multihead_params.msgpack",
    qeq_config_path="/path/to/qeq_multihead_config.json",
    mace_device="cuda",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
charges = atoms.calc.results["charges"]
```

## 5. 结果字段

标准 ASE 字段为 `energy`、`free_energy`、`forces` 和 `charges`。另外保留以下诊断字段：

- `short_range_energy`、`short_range_forces`
- `long_range_energy`、`long_range_forces`
- `partial_charges`
- `chi`、`hardness`、`eta`

这些分量对排查负电荷、能量漂移和异常力非常重要。
