# 主动学习：用 MACE + QEq 的 MD 补充训练数据

这一步排在 `validation/fd_test.md` 的验证清单之后：先确认力、求解器和能量的实现都是对的，再来改进势能面本身。

---

## 0. 为什么需要主动学习

第一次用 `md_run.py` 跑 300 K NVT（4090，float32 MACE，`pg_tol = 1e-5`），第 182 步温度升到 14233 K，模拟中止。分析崩溃前后的构型，结论如下：

- 前 170 步一切正常。温度从 300 K 升到约 360 K，这是起始结构在新势能面上弛豫、放出约 12 eV 势能造成的。QEq 也稳定：没有回退到 matrix，每步 34–38 次 CG 迭代，最大 |q| ≤ 0.95 e。
- 在底部电极表面，先后发生了水分子吸附（Zn–O 从 2.4–2.8 Å 缩短到 1.9–2.1 Å）和质子转移（H385 从 O384 转到 O327）。随后 H329 离开 O327，被拉向表面的 Zn101。
- 最后 9 步，最大力依次为 4.6 → 5.1 → 6.4 → 6.9 → 5.5 → 8.1 → 10 → 16 → 40 → 908 eV/Å。908 eV/Å 的力在一步内把 H329 加速到 4.1 Å/fs，此时它距 Zn101 只有 0.95 Å（真实的 Zn–H 键长约 1.6 Å）。
- 最后一步 MACE 能量升高约 20 eV，QEq 能量只变化了 −0.15 eV。

所以这不是积分器或 QEq 求解器的数值不稳定，而是短程势能面在这些反应构型上外推失败。减小步长只能推迟崩溃。根本的办法是把这类构型补进训练集。

---

## 1. 策略

| | 旧脚本 `run_active_learning_md.py` | 本目录 |
|---|---|---|
| 驱动 MD 的势能 | 4 个全量 MACE 的平均，没有 QEq | 生产用的 MACE + QEq（`MACEJAXQEqCalculator`） |
| 委员会 | 全量 MACE | 4 个用同样残差标签（E_DFT − E_QEq）、不同随机种子训练的短程 MACE |
| 不确定度 | 可移动原子的最大 std / (\|F̄_MACE\| + 0.2) | 可移动原子的最大 std / (\|F̄_short + F_QEq\| + 0.2)，即相对于实际驱动动力学的总力 |
| 超过硬阈值时 | 整条轨迹停止 | 保存该帧，回退 40 步、重抽速度后继续，从不同方向再次进入这个区域 |
| 发生崩溃时 | — | 给崩溃前 60 步每隔一帧打分，保留超过软阈值的帧和崩溃帧，然后同样回退 |
| 最后选哪些帧 | 超过阈值的帧，两次保存至少间隔 100 步 | 对委员会标记的帧做最远点采样，数量可以指定 |

QEq 部分由所有委员会成员共用，所以委员会分歧只反映短程部分。短程部分正是 MACE-adQEq 中依赖训练数据覆盖的那一项。

---

## 2. 前置条件

- [ ] 委员会的 4 个模型用的残差标签，必须是用当前 `jax_model/` 中的 QEq 参数算出来的。QEq 参数一改，就要重新生成标签、重训委员会。
- [ ] 4 个模型的 `r_max` 和元素表相同（`MACECalculator` 加载多个模型时会检查）。
- [ ] `--mace-model`（驱动 MD 的短程模型）也用同一份残差标签训练。最简单的做法是直接用委员会中的一个成员。
- [ ] `validation/fd_test.md` 第 2、3 节已经通过：constQ 和恒电势下的力一致性，以及 hybrid + CG 求解路径。

---

## 3. 脚本

三个脚本都在本目录。默认路径与集群上的布局一致（`stru/`、`data/`、`jax_model/` 放在脚本旁边），都可以用参数覆盖。所有模型输出都以 `model_*` / `committee_*` 的名字保存，不写成 `forces` / `energy`，因此不会被 DFT 标注脚本误当成参考标签。

### 3.1 `md_run.py`：普通 MD

- 固定上下电极最外层的 Zn（默认各 1 层、30 个原子；`--fix-layers` 可调），其余原子由 MACE + QEq 驱动。
- `--ensemble nve`：velocity Verlet，汇总总能量漂移和涨落，对应 `fd_test.md` 第 6 节。
- `--ensemble nvt`：Langevin，摩擦系数 10 ps⁻¹，关闭 `fixcm`（否则动量会被推进固定层）。
- 默认配置：4090、float32 MACE、float64 QEq、`pg_tol = 1e-5`。
- `md.csv` 每步记录能量、温度、MACE / QEq / 总力的最大值及所在原子、电荷、QEq 求解器统计；开启恒电势时还记录两侧电极的原子数。
- 任何可移动原子的力超过 `--max-force`（默认 50 eV/Å）、温度超过上限或出现 NaN 时立即停止，并写出 `failure_step_<n>.extxyz` 和最近 50 步的 `pre_failure.extxyz`。
- `traj.extxyz` 带动量，可以用 `--frame -1 --keep-momenta` 接着跑。

### 3.2 `al_collect.py`：委员会打分的探索 MD

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--committee` | 必填 | 4 个委员会模型，或包含它们的目录 |
| `--temperatures` | 300 400 500 | 各条轨迹轮流使用的温度 |
| `--walkers` / `--steps` | 6 / 10000 | 轨迹条数和每条的步数 |
| `--check-interval` | 10 | 每隔多少步打一次分 |
| `--error-threshold` / `--hard-error-threshold` | 0.4 / 0.8 | 软阈值和硬阈值，需要重新标定（见第 4 节） |
| `--candidate-interval` | 100 | 两次软阈值候选之间至少间隔的步数 |
| `--rewind` / `--max-hard-events` | 40 / 5 | 回退步数；每条轨迹最多容忍的硬事件和崩溃次数 |
| `--periodic-interval` | 0（关闭） | 无条件保存帧，作为对照 |

输出：
- `candidates.extxyz`：每帧带委员会的逐原子不确定度，`source` 为 `high_uncertainty`、`hard_uncertainty` 或 `failure`；
- `scores.csv`：每一次打分；
- `events.csv`：每次硬事件和崩溃；
- `summary.json`：每条轨迹的不确定度分位数和硬事件次数。

### 3.3 `al_select.py`：多样性筛选

- 每帧的描述符由两部分拼接：委员会分歧最大的 8 个原子的 MACE 不变描述符（取平均和最大值），以及按元素平均的全局描述符。前者让不同的"不确定事件"可以区分开，后者区分体系的整体状态。
- 从不确定度最高的帧开始做最远点采样，用 `--n` 指定数量。同一个事件连续触发的十几帧相似构型，最后只会选中一帧。
- 去掉存在距离 < 0.6 Å 原子对的帧，以免 DFT 不收敛。
- `selection.csv` 按选中顺序列出每帧的不确定度和最远点距离。距离曲线变平的地方，就是再多选也收效甚微的位置。

---

## 4. 运行方式

```bash
cd active_learning
# 先跑一小段，标定阈值
python al_collect.py --committee committee/ --mace-model committee/<其中一个>.model \
    --walkers 1 --steps 1000 --output al/calibration
# 正式收集：2–3 个进程并行，种子和温度不同
python al_collect.py --committee committee/ --mace-model committee/<其中一个>.model --seed 1 --output al/round1_s1
python al_collect.py --committee committee/ --mace-model committee/<其中一个>.model --seed 2 \
    --temperatures 400 500 600 --output al/round1_s2
# 如果生产用恒电势，再加一个带 --const-potential 的进程
python al_select.py --candidates al/round1_*/candidates.extxyz --n 1500 \
    --mace-model committee/<其中一个>.model --output al/round1_selected.extxyz
```

**阈值标定：** 0.4 / 0.8 沿用自旧的全量 MACE 委员会，而现在分母里包含了 QEq 力，需要按新的分布重新确定。用 `al/calibration/scores.csv` 中 `max_relative_std` 的分布：软阈值取 1–5% 的打分会超过它的位置，硬阈值取在尾部。

**数据量：** 4090 上约 0.6 s/步，委员会每 10 步额外算 4 次 MACE，开销不大。一个进程按默认设置约 10 小时，最多产生约 600 个软阈值候选，再加上硬事件和崩溃帧。需要更多数据时可以：
- 多开几个进程；
- 用 `--start` 传入多个起始结构；
- 把 `--candidate-interval` 降到 50；
- 加大 `--max-hard-events`。

---

## 5. 迭代与收敛

每一轮依次是：收集 → 筛选 → DFT 标注（计算残差标签）→ 重训 4 个委员会模型和驱动模型 → 下一轮收集。每轮用同样的设置比较以下指标：

- 每条轨迹的硬事件和崩溃次数（`summary.json`、`events.csv`）；
- 第一次硬事件之前走过的步数；
- 打分分位数（`relative_std_quantiles`）。

满足以下条件后，再回到 `validation/fd_test.md` 第 6 节做 NVE 测试和生产 MD：

- [ ] 300 K 和最高探索温度下，10 ps 内没有硬事件或崩溃；
- [ ] 99% 分位的不确定度低于软阈值；
- [ ] `md_run.py --ensemble nvt` 能稳定跑完目标时长。

---

## 6. 说明

- 三个脚本只用替身模型测试过流程：力超阈值时停止、委员会打分、回退重探、帧的写出和最远点采样。还没有在真实模型上跑过。
- `run_active_learning_md.py` 第 230 行写成了 `atom.copy()`，应为 `atoms`，所以那个脚本在 MD 跑完、写最终结构时会报 `NameError`。它没有提交进本仓库。
