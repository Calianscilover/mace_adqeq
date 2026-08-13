# 通用 JAX-QEq 与 MACE 的组合 ASE Calculator

## 1. 当前实现的目标

本目录实现的是一个专用的 MACE + ADQEq ASE Calculator：

```text
ASE Atoms
   ├── 官方 MACECalculator ──────> E_short, F_short
   └── QEqParameterPredictor ────> chi, J, eta
                         │
                         └── JAXQEqModel ──> E_qeq, F_qeq, q
                         │
                         └── E = E_short + E_qeq
                             F = F_short + F_qeq
```

MACE 没有被嵌入 JAX，也不需要参与 QEq 求解。`MACEJAXQEqCalculator` 直接使用 `from mace.calculators import MACECalculator` 加载短程模型，并在同一个 ASE Calculator 中调用独立的 JAX-QEq 模型，最后对两部分能量和力求和。本实现不再提供通用 short-range Calculator 注入接口。

## 2. 文件结构

- `mace_adqeq/qeq.py`：`QEqParameterPredictor` 负责恢复 Flax 模型并预测参数，`JAXQEqModel` 只负责 QEq 求解、能量和力。
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

QEq 计算拆分为两个独立步骤：

1. `QEqParameterPredictor` 根据 JSON 生成 ACSF 和元素 one-hot，加载 Flax checkpoint，输出 `chi`、`J` 和 `eta`。
2. `JAXQEqModel` 接收结构及这三个参数，构造 PME-QEq 能量，在 `sum(q_i) = Q_total` 约束下求解电荷，并对坐标求梯度得到力。

三个参数均使用训练模型的输出：`chi` 采用逐结构中心化增量，
`hardness` 和 `eta` 分别采用受限对数增量后与元素基线组合。偶极修正默认
使用坐标轴 `1`（y），因为原代码虽然变量名为 `Mz`，实际写的是
`positions[:, 1]`。

QEq 默认使用 `solver_mode="hybrid"`：第一帧通过迭代
Newton-KKT 矩阵法获得严格约束的自洽电荷，之后每帧以上一帧电荷为初值，
使用投影 LBFGS 求解。原子数或元素顺序发生变化时会自动回到矩阵法。
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
  --mace-device cuda
```

`MACEJAXQEqCalculator` 专用于中性体系，QEq 总电荷固定为 `0.0`，不读取
`atoms.info["total_charge"]`。

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

两个 QEq 组件也可以独立调用：

```python
from mace_adqeq import JAXQEqModel, QEqParameterPredictor

predictor = QEqParameterPredictor("params.msgpack", "config.json")
parameters = predictor.predict(atoms)

qeq = JAXQEqModel(cutoff=predictor.cutoff)
result = qeq.calculate(
    atoms,
    chi=parameters.chi,
    hardness=parameters.hardness,
    eta=parameters.eta,
    total_charge=0.0,
)
```

## 5. 结果字段

标准 ASE 字段为 `energy`、`free_energy`、`forces` 和 `charges`。另外保留以下诊断字段：

- `short_range_energy`、`short_range_forces`
- `long_range_energy`、`long_range_forces`
- `partial_charges`
- `chi`、`hardness`、`eta`

这些分量对排查负电荷、能量漂移和异常力非常重要。
