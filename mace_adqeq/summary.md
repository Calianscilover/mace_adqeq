# MACE-adQEq 代码回顾（`qeq.py`）

> 用途：会议前快速回忆。核心思路是用冻结的 MACE 提取原子环境特征，用 Flax MLP 预测每个原子的 QEq 参数，再用 JAX 中的 QEq + PME 求电荷、能量和力。

---

## 0. 一句话总览

```text
ASE Atoms
  │  MACEEmbedding (PyTorch, 冻结)
  ▼
h ∈ R^{N×D}      每个原子的不变特征（各层 0e 标量拼接）
  │  QEqMultiHeadMLP (Flax/JAX)
  ▼
θ = (χ, J, η)    每个原子的电负性 / 硬度 / 高斯宽度
  │  determine_chi：电极原子加常数偏置（恒电势）
  │  QEq 求解（matrix KKT 或 projected-gradient LBFGS）
  ▼
q*  →  E(q*, R, θ) = PME + 高斯屏蔽修正 + onsite + 偶极修正
  │
  ▼
forces = -(∂E/∂R  +  ∂E/∂θ · ∂θ/∂h · ∂h/∂R)
```

主要类和函数：

| 模块 | 作用 |
|---|---|
| `MACEEmbedding` | 加载并冻结 MACE，提取不变特征，提供 `position_vjp` |
| `QEqMultiHeadMLP` | 共享主干网络 + 3 个零初始化输出头 |
| `predict_qeq_parameters` | 把输出头的原始值变成 χ / J / η（基线 + 受约束的修正） |
| `QEqParameterPredictor` | 读取 config 和权重，串起 MACE 和 MLP，提供 `predict` / `predict_with_derivatives` |
| `generate_get_Energy_Qeq` | QEq 能量函数（JAX，jit） |
| `JAXQEqModel` | 邻居表、电荷求解、能量和力，入口是 `calculate` |
| `const_potential.determine_chi` | 识别上下电极的 Zn 原子，给 χ 加常数偏置 |

---

## 1. MACE 不变特征作为原子几何描述符

### 1.1 加载和冻结（`MACEEmbedding.__init__`）

- 用 `MACECalculator` 加载模型，取 `models[0]`，设为 `eval()`，所有参数 `requires_grad_(False)`。
- MACE 只当特征提取器，不参与训练。
- `num_layers=-1` 表示取全部 `num_interactions` 层，否则取前 `num_layers` 层。

### 1.2 推断特征维度

```python
irreps_out = o3.Irreps(str(self.model.products[0].linear.irreps_out))
self.l_max = irreps_out.lmax
self.num_invariant_features = irreps_out.dim // (self.l_max + 1) ** 2
self.output_dim = self.num_layers * self.num_invariant_features
```

例：隐藏层 irreps 为 `128x0e + 128x1o`
- `l_max = 1`，`dim = 128×1 + 128×3 = 512`
- 每层标量通道数 `512 // 4 = 128`
- 2 层 interaction 时 `output_dim = 256`，必须等于 config 中 MLP 的 `input_dim`（初始化时会检查）

注意：
- 取 `products[0]`（第一层）是因为 MACE 最后一层通常只输出标量，第一层才有完整的各个 l 分量。
- 公式默认每个 l 的通道数相同（对称 irreps）。如果 irreps 不对称，算出的维度会错。

### 1.3 前向传播并提取标量（`_invariant_features`）

1. `_batch_for_atoms`：借用 `calculator._atoms_to_batch` 把 `Atoms` 转成图（周期性边界由 cell 处理），浮点张量统一转成模型 dtype。
2. `self.model(batch_dict, compute_force=False)`，取 `output["node_feats"]`，也就是各层节点特征按顺序拼起来的结果。
3. `extract_invariant(...)`：从每层的块中只取 `0e` 标量部分，跨层拼接，丢掉 l≥1 的等变部分，得到严格不变的特征。
4. 截取前 `len(atoms)` 行并检查形状 `(N, output_dim)`。

### 1.4 特征包含哪些信息

- **元素信息**：MACE 的节点嵌入来自元素 one-hot，元素身份已经隐含在特征中。`self.species` 只是读出 MACE 支持的元素列表用于校验，没有额外拼接 one-hot。
- **局部几何**：每层在截断半径内做多体消息传递，感受野约为 截断半径 × 层数（例如 5 Å × 2 ≈ 10 Å）。
- **长程静电**：不靠特征，交给 QEq + PME 显式计算。
- `num_layers` 控制局域程度：浅层更局域，深层看得更远。

### 1.5 两种调用模式

| 方法 | 是否保留计算图 | 用途 |
|---|---|---|
| `features(atoms)` | 否（`torch.no_grad`） | 只需要参数、电荷或能量（`compute_forces=False`），省内存 |
| `features_with_context(atoms)` | 是，返回 `MACEEmbeddingContext(features, positions)` | 计算完整力 |

### 1.6 下游：MLP 预测 QEq 参数

- `QEqMultiHeadMLP`：`num_hidden_layers` 层 Dense → LayerNorm → SiLU 组成的共享主干网络，3 个输出头 `chi_head` / `hardness_head` / `eta_head` **全零初始化**，所以训练开始时预测值恰好等于元素基线。
- `predict_qeq_parameters`：
  - χ：`chi_base + chi_scale·raw`，并在每个结构内减去均值（`center_chi`）
  - J：`hardness_base · exp(hardness_log_range · tanh(raw))`，默认范围在基线的 ×/÷2 之内
  - η：`eta_base · exp(eta_log_range · tanh(raw))`，默认范围在基线的 ×/÷1.5 之内

---

## 2. 完整力：跨 PyTorch 和 JAX 的链式法则

### 2.1 数学

符号：`N` 原子数，`R ∈ R^{N×3}`，`h ∈ R^{N×D}`，`θ = (χ, J, η)`，`q*` 为求解得到的电荷。

```text
dE/dR = ∂E/∂R                      ① 直接项（库仑、PME）
      + ∂E/∂q · dq*/dR             ② 电荷响应 = 0
      + ∂E/∂θ · ∂θ/∂h · ∂h/∂R      ③ 参数响应（跨框架）
```

**为什么 ② = 0（包络定理 / Hellmann–Feynman 型论证）**：电荷是在约束 `Σq = Q` 下最小化能量得到的。在约束极小点上 `∂E/∂q = μ·1`（μ 是拉格朗日乘子，即化学势），而总电荷固定意味着 `Σ dq_i/dR = 0`，所以 ② = `μ · Σ dq_i/dR = 0`。
- 因此代码把 `charges` 当常数，不需要对求解器求导。
- 前提是电荷真正收敛。PG 求解器的 `tol=1e-3` 会给力带来少量误差。

### 2.2 为什么用 VJP（向量-雅可比积）

- 雅可比矩阵太大，不能直接构造。`∂h/∂R` 的形状是 `(N·D) × (N·3)`；`N=1000, D=256` 时约有 7.68×10⁸ 个元素，float32 约 3 GB。
- 两种自动微分模式：

| 模式 | 计算的量 | 适用场景 |
|---|---|---|
| 前向模式 JVP | `J·v`（v 是输入方向） | 输入少、输出多 |
| 反向模式 VJP | `vᵀ·J`（v 是输出方向，即余切向量 cotangent） | 输出是标量（如能量 E），一次反向传播得到对所有输入的梯度 |

- 反向传播的每一层都在做 `x̄ = ȳᵀ · ∂y/∂x`。`loss.backward()` 和 `jax.grad` 底层都是一串 VJP。
- 第③项从左往右逐步计算：

```text
∂E/∂θ  --VJP₁ (JAX, MLP)-->  h̄ = grad_features (N, D)  --VJP₂ (Torch, MACE)-->  R̄ (N, 3)
```

- **关键点**：两个框架之间只需要传一个余切向量 `grad_features`，它的形状和交界张量 `h` 完全一样。

### 2.3 代码逐步拆解

**第 1 步：PyTorch 前向，保留计算图**（`MACEEmbedding.features_with_context`）
- MACE 参数被冻结，但 MACE 的 forward 内部会对 `positions` 调用 `requires_grad_(True)`，所以 `features` 仍通过 `grad_fn` 链接到 `positions`，代码会检查这一点。
- 产生两份特征：
  - `descriptor`：`detach()` 后的 numpy 数组，已和计算图断开，送去给 JAX。
  - `context.features` / `context.positions`：仍挂着 PyTorch 计算图，留到第 5 步反传时用。
- PyTorch 是动态图（tape），`context` 的作用是把整张图保持在内存里。

**第 2 步：JAX 前向，同时得到 MLP 的反向函数**（`QEqParameterPredictor.predict_with_derivatives`）

```python
features = jnp.asarray(descriptor, dtype=jnp.float32)   # 对 JAX 来说只是一个常量输入
parameter_fn = lambda values: self.parameters_from_features(values, symbols)
values, pullback = jax.vjp(parameter_fn, features)
```

- `jax.vjp(f, x)` 返回 `f(x)` 和 `pullback`。`pullback` 接收和 `values` 同结构的余切向量，返回和 `(x,)` 同结构的梯度，内部保存着前向计算的中间结果（residuals）。
- JAX 是函数式的：调用 `jax.vjp` 时追踪函数并生成反向函数。`pullback` 和 PyTorch 的 `context` 起相同作用。

**第 3 步：JAX 计算 ∂E/∂R 和 ∂E/∂θ**（`JAXQEqModel.calculate` 中的 `#whole_force` 分支）

```python
energy, gradients = jax.value_and_grad(
    self.energy_fn, argnums=(1, 4, 5, 6)
)(charges, positions, box, pairs, eta, chi, hardness)
grad_positions, grad_eta, grad_chi, grad_hardness = gradients
```

- `energy_fn` 的参数顺序：`charges=0, positions=1, box=2, pairs=3, eta=4, chi=5, hardness=6`。
- `charges`（argnum 0）不在 `argnums` 里，被当作常数，对应 ② = 0。

**第 4 步：VJP₁（JAX），从 θ 拉回到 h**

```python
(grad_features,) = parameter_pullback((grad_chi, grad_hardness, grad_eta))
```

- 余切向量必须和 `values` 结构一致，即 `(chi, hardness, eta)` 的顺序。
- ⚠️ `value_and_grad` 那里的解包顺序是 `(eta, chi, hardness)`，两处顺序不同，改代码时容易写反。
- 因为 χ 减去了结构内均值，原子之间是耦合的，VJP 会自动处理。
- **关于 `const_potential`**：`grad_chi` 是对加了偏置之后的 χ 求的导。`apply_chi_bias` 只给电极原子加常数（`+10` 或 `-2`），所以 `∂χ_final/∂χ_pred = I`；电极原子的识别是离散判断，对坐标的导数几乎处处为 0。因此直接传 `grad_chi` 是正确的。如果以后偏置变成依赖 χ 的非线性函数，就需要补上它的雅可比。

**第 5 步：跨框架，VJP₂（PyTorch），从 h 拉回到 R**（`MACEEmbedding.position_vjp`）

```python
cotangent = torch.as_tensor(np.asarray(grad_features),
                            dtype=context.features.dtype,
                            device=context.features.device)
grad_positions = torch.autograd.grad(
    outputs=context.features, inputs=context.positions,
    grad_outputs=cotangent, retain_graph=False, create_graph=False,
)[0]
```

- 数据转换路径：JAX 数组 → `np.asarray`（拷到主机内存）→ `torch.as_tensor`（按 MACE 的 dtype 和 device）。
- `torch.autograd.grad(..., grad_outputs=v)` 就是 PyTorch 的 VJP，计算 `R̄[k,a] = Σ_{i,d} h̄[i,d] · ∂h[i,d]/∂R[k,a]`，等价于 `(features*cotangent).sum().backward()`，但不会污染 `.grad`。
- `retain_graph=False`：反传后立即释放计算图，一个 `context` 只能用一次。
- `create_graph=False`：不构建二阶导的计算图。如果以后要用力做训练损失，需要改成 `True`。

**第 6 步：合并**

```python
forces = -(grad_positions + grad_parameter_response)
```

| 模式 | 包含哪些项 | 代码分支 |
|---|---|---|
| frozen force | 只有 ①，参数固定 | `parameter_pullback is None` |
| whole force | ① + ③ | `#whole_force` |
| 不算力 | 力设为 0 | `compute_forces=False` |

两边单位都是 Å 和 eV，可以直接相加。

### 2.4 时间顺序

```text
Torch 前向（保留图）──descriptor──▶ JAX: jax.vjp(MLP) 得到 θ 和 pullback
                                        │
                                 JAX: 求解 q*；value_and_grad 得到 ∂E/∂R, ∂E/∂θ
                                        │
                                 JAX: pullback(∂E/∂θ) 得到 grad_features (N, D)
                                        │
Torch: autograd.grad(features, positions, grad_features) ◀──numpy──┘
       │
       └─▶ (N, 3) ──▶ 与 ∂E/∂R 相加 ──▶ forces
```

---

## 3. 跨框架转换：基础知识和注意事项

1. **为什么梯度不能自动穿过边界**
   - PyTorch 的梯度靠张量上的 `grad_fn` 链传递，`.detach()` / `.numpy()` 会切断它。
   - JAX 的梯度靠追踪 JAX 原语传递，它不知道 numpy 数组或 torch 张量的来历，只当常量。
   - 所以必须在交界处手动用余切向量接上链式法则。`torch2jax`、`jax2torch` 这类库做的是同一件事，只是包装成了 `torch.autograd.Function` 或 `jax.custom_vjp`。
2. **两边都要暂存反向传播需要的状态**：PyTorch 存在计算图里（`context`），JAX 存在 `pullback` 闭包里。交界处只做一次 VJP，不需要额外的雅可比内存。
3. **设备和拷贝开销**
   - JAX 设为 GPU（`JAX_PLATFORM_NAME="gpu"`），MACE 默认 `mace_device="cpu"`，每一步都有主机和 GPU 之间的拷贝。
   - `np.asarray(jax_array)` 会强制同步，因为 JAX 是异步派发的。
   - 如果两边都在 GPU 上，可以用 DLPack 零拷贝（`torch.utils.dlpack` / `jax.dlpack`）。
4. **数据类型**：全局开启了 `jax_enable_x64`，但特征、坐标和梯度都被显式转成 float32；MACE 可能是 float64，余切向量会被转回 `features.dtype`。力最终是 float32 精度。
5. **验证方法**
   - 对某个原子坐标做 ±δ（例如 1e-3 Å）的有限差分：`-(E₊ − E₋) / 2δ`，和 `forces` 对比。
   - 比较时用 matrix 求解器、float64，避免 PG 热启动的收敛误差污染结果。
   - 预期：只有 ① 时误差明显，加上 ③ 后吻合，说明跨框架拼接正确。

---

## 4. 待讨论 / 后续可做

- [ ] 用有限差分验证 whole force（matrix 求解器 + float64）。
- [ ] 评估 frozen force 和 whole force 的差别对 MD 的影响。
- [ ] 考虑用 DLPack 零拷贝，或把 MACE 放到 GPU 上，减少拷贝开销。
- [ ] 如果要用力训练：`create_graph=True`，或封装成 `jax.custom_vjp` / `torch.autograd.Function`，做成端到端可微的函数。
- [ ] 如果以后电极偏置依赖 χ（非常数），需要在 pullback 前补上它的雅可比。
- [ ] 确认 MACE 的 irreps 是对称结构，保证 `num_invariant_features` 的推断正确。
