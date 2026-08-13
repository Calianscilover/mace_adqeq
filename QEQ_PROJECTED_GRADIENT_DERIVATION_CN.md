# QEq 总电荷约束与投影梯度推导

本文记录 `mace_adqeq/qeq.py` 中 `projected_energy()` 的理论依据，重点说明下面这行代码为什么已经包含了总电荷约束矩阵：

```python
projected_gradient = gradient - jnp.mean(gradient)
```

## 1. QEq 的总电荷约束

设体系包含 $N$ 个原子，原子电荷组成向量

$$
\mathbf q =
\begin{bmatrix}
q_1 & q_2 & \cdots & q_N
\end{bmatrix}^{\mathrm T}.
$$

QEq 求解要求原子电荷之和等于体系给定的总电荷 $Q_{\mathrm{total}}$：

$$
\sum_{i=1}^{N} q_i = Q_{\mathrm{total}}.
$$

将其写成矩阵形式：

$$
C\mathbf q = Q_{\mathrm{total}},
$$

其中约束矩阵是一个全为 1 的行向量：

$$
C =
\begin{bmatrix}
1 & 1 & \cdots & 1
\end{bmatrix}.
$$

因此，`projected_energy()` 中虽然没有显式构造 `constraint_matrix`，但使用的是这个特殊的 $C$。

## 2. 可行电荷变化必须满足什么条件

假设当前电荷已经满足约束，并对电荷施加更新 $\Delta\mathbf q$：

$$
\mathbf q' = \mathbf q + \Delta\mathbf q.
$$

为了使更新后的电荷仍满足相同的总电荷约束，必须有

$$
C\mathbf q' = Q_{\mathrm{total}}.
$$

代入 $\mathbf q'$：

$$
C\mathbf q + C\Delta\mathbf q = Q_{\mathrm{total}}.
$$

由于 $C\mathbf q=Q_{\mathrm{total}}$，所以

$$
C\Delta\mathbf q=0.
$$

对于全为 1 的 $C$，这等价于

$$
\sum_{i=1}^{N}\Delta q_i=0.
$$

因此，优化过程中不要求每个原子电荷不变，而是要求所有原子的电荷变化量之和为零。某些原子增加的电荷必须由其他原子等量减少。

## 3. 通用的正交投影公式

令 QEq 能量对电荷的普通梯度为

$$
\mathbf g = \nabla_{\mathbf q}E(\mathbf q).
$$

将梯度投影到约束面的切空间，通用公式为

$$
\boxed{
\mathbf g_{\mathrm{proj}}
=
\mathbf g
-C^{\mathrm T}(CC^{\mathrm T})^{-1}C\mathbf g
}.
$$

其中：

- $C\mathbf g$ 计算普通梯度在约束法向方向上的分量；
- $(CC^{\mathrm T})^{-1}$ 对约束向量的长度进行归一化；
- $C^{\mathrm T}(CC^{\mathrm T})^{-1}C\mathbf g$ 是应从普通梯度中移除的法向分量；
- 剩余的 $\mathbf g_{\mathrm{proj}}$ 位于可行切空间内。

它满足

$$
C\mathbf g_{\mathrm{proj}}=0.
$$

## 4. 总电荷约束下公式如何化简

对于

$$
C=\begin{bmatrix}1&1&\cdots&1\end{bmatrix},
$$

有

$$
C\mathbf g=\sum_{i=1}^{N}g_i,
$$

并且

$$
CC^{\mathrm T}=N.
$$

因此，被移除的梯度分量是

$$
C^{\mathrm T}(CC^{\mathrm T})^{-1}C\mathbf g
=
\mathbf 1\frac{\sum_{i=1}^{N}g_i}{N}
=
\mathbf 1\,\overline g,
$$

其中

$$
\overline g=\frac{1}{N}\sum_{i=1}^{N}g_i
$$

是梯度所有分量的平均值。最终得到

$$
\boxed{
\mathbf g_{\mathrm{proj}}
=
\mathbf g-\overline g\,\mathbf 1
}.
$$

这正是代码中的

```python
projected_gradient = gradient - jnp.mean(gradient)
```

所以约束矩阵并未被遗漏，而是因为 $C$ 是全为 1 的特殊矩阵，完整投影公式被直接化简成了“每个梯度分量减去平均梯度”。

## 5. 一个三原子示例

假设普通梯度为

$$
\mathbf g=
\begin{bmatrix}
2\\
5\\
8
\end{bmatrix}.
$$

平均梯度为

$$
\overline g=\frac{2+5+8}{3}=5.
$$

投影后得到

$$
\mathbf g_{\mathrm{proj}}
=
\begin{bmatrix}
2\\
5\\
8
\end{bmatrix}
-
\begin{bmatrix}
5\\
5\\
5
\end{bmatrix}
=
\begin{bmatrix}
-3\\
0\\
3
\end{bmatrix}.
$$

投影梯度所有分量之和为

$$
-3+0+3=0.
$$

如果优化器使用步长 $\alpha$ 沿负投影梯度更新：

$$
\Delta\mathbf q=-\alpha\mathbf g_{\mathrm{proj}},
$$

那么

$$
\sum_i\Delta q_i=0,
$$

所以更新前后的总电荷相同。

## 6. 为什么初始电荷必须已经满足约束

投影梯度只能保证电荷变化量之和为零：

$$
\sum_i\Delta q_i=0.
$$

它保持的是初始电荷的总和，而不会自动将错误的初始总电荷改成目标值。例如，初始电荷之和若为 $0.5e$，投影优化会继续保持 $0.5e$，不会自动变成 $0e$。

当前代码通过以下流程满足这一前提：

1. 第一帧使用带 KKT 约束的矩阵法求解电荷；
2. 矩阵法的输出满足目标总电荷；
3. 后续帧从 `self.charge_list` 读取上一帧电荷作为初值；
4. 只有原子数、元素顺序和总电荷均兼容时，才使用投影 LBFGS；
5. 投影梯度保证后续更新不改变初始总电荷；
6. 求解结束后再均匀修正浮点误差。

对应代码为：

```python
total_charge = jnp.sum(charges)

result = solver.run(charges, *energy_args)
optimized_charges = result.params

optimized_charges += (
    total_charge - jnp.sum(optimized_charges)
) / optimized_charges.shape[0]
```

最后一行给每个原子加上相同的微小修正量。其总修正量恰好为

$$
Q_{\mathrm{initial}}-\sum_iq_i^{\mathrm{optimized}},
$$

从而消除有限精度计算造成的总电荷漂移。

## 7. 为什么 LBFGS 也可以使用投影梯度

LBFGS 不一定直接沿 $-\mathbf g_{\mathrm{proj}}$ 更新，而是利用当前及历史梯度、历史步长构造搜索方向。在理想数学条件下：

- 每次传给 LBFGS 的梯度都满足 $C\mathbf g_{\mathrm{proj}}=0$；
- 初始方向和后续方向由这些切空间向量线性组合得到；
- 切空间对线性组合封闭；
- 因此搜索方向仍满足各分量之和为零。

有限精度和求解器内部操作仍可能产生很小的误差，因此最终的均匀电荷修正仍然必要。

## 8. 与显式约束矩阵实现的关系

当前代码：

```python
def projected_energy(charges, *energy_args):
    value, gradient = jax.value_and_grad(energy_fn)(charges, *energy_args)
    projected_gradient = gradient - jnp.mean(gradient)
    return value, projected_gradient
```

等价于针对单一总电荷约束使用：

```python
constraint_matrix = jnp.ones((1, charges.shape[0]))

constraint_gradient = constraint_matrix @ gradient
gram_matrix = constraint_matrix @ constraint_matrix.T

projected_gradient = (
    gradient
    - constraint_matrix.T
    @ jnp.linalg.solve(gram_matrix, constraint_gradient)
)
```

前一种写法更适合当前问题，因为它：

- 与总电荷约束完全等价；
- 不需要显式建立矩阵；
- 不需要求逆或求解线性方程；
- 代码和计算成本都更小。

## 9. 多约束时需要注意的问题

如果未来不仅约束体系总电荷，还要分别约束多个区域的电荷，则应使用通用矩阵公式：

$$
\mathbf g_{\mathrm{proj}}
=
\mathbf g-C^{\mathrm T}(CC^{\mathrm T})^{-1}C\mathbf g.
$$

逐行分别除以每个约束向量的平方范数，只在不同约束向量彼此正交时严格成立。对于一般的非正交约束，应求解完整的 Gram 矩阵 $CC^{\mathrm T}$；若约束线性相关，还需要使用伪逆或先移除冗余约束。

## 10. 与拉格朗日乘子条件的联系

约束极小点满足

$$
\nabla_{\mathbf q}E + C^{\mathrm T}\lambda=0.
$$

对于总电荷约束，$C^{\mathrm T}=\mathbf 1$，所以极小点处

$$
\nabla_{\mathbf q}E=-\lambda\mathbf 1.
$$

这意味着所有原子的普通电荷梯度在收敛时应相等，而不一定全部等于零。减去梯度平均值后：

$$
\mathbf g_{\mathrm{proj}}=0.
$$

因此，投影梯度的范数才是这个约束优化问题正确的收敛判据。

## 11. 核心结论

1. 当前总电荷约束对应的矩阵是 $C=[1,1,\ldots,1]$。
2. `gradient - mean(gradient)` 是完整约束投影公式对该特殊矩阵的精确化简。
3. 投影梯度保证每一步电荷变化量之和为零，因此保持初始总电荷。
4. 投影法不能修复错误的初始总电荷，所以初值必须已经满足目标约束。
5. 当前实现通过第一帧 KKT 矩阵法、后续帧电荷缓存和最终均匀修正来满足这一要求。
6. 如果未来引入多个一般约束，应改用完整的 $C^{\mathrm T}(CC^{\mathrm T})^{-1}C$ 投影。
