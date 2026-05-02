# Plan 04: 验证 SwiGLU Clamping 对训练稳定性的影响

**目标原子**: `9286d65b` — "SwiGLU Clamping for Training Stability"

---

## 1. 声明回顾

论文声称:
- SwiGLU Clamping "effectively eliminates outliers"
- Clamping "substantially aids in stabilizing the training process, without compromising performance"
- 具体实施: 线性分量 clamp 到 `[-10, 10]`，门控分量 upper bound 设为 `10`

**核心问题**: 
- 多大幅度的 outlier 被视为消除？
- "不损害性能" 的证据在哪里？(论文未提供任何 ablation)
- 10 这个阈值是如何选择的？是否最优？

---

## 2. SwiGLU 异常值产生机制

SwiGLU 结构:
```
SwiGLU(x) = (x @ W_gate * sigmoid(x @ W_gate)) * (x @ W_up)
```

或更常见的:
```
SwiGLU(x) = (SiLU(x @ W_gate)) * (x @ W_up)
```

其中 SiLU(x) = x * sigmoid(x)，该函数在负无穷时趋于 0，在正无穷时近似线性增长。

当训练不稳定时 (梯度爆炸、路由崩溃)，某些激活值可能变得极大，导致 SwiGLU 输出出现 outlier。

---

## 3. 实验设计

### Phase A: 小规模 MoE 训练 — 有/无 Clamping 对比

**模型**: 
- 一个小型 MoE Transformer (类似 V4 架构但缩小)
- Layers = 12, d_model = 768, n_heads = 12
- 4 个路由专家 + 1 个共享专家
- 训练数据: 1B tokens (The Pile / C4 subset)

**实验条件**:

| 组 | Clamping | 描述 |
|----|----------|------|
| Control | 无 | 无任何 clamping |
| Linear-10 | 线性分量 `[-10, 10]` | 仅 clamp 线性分量 |
| Gate-10 | 门控上限 `10` | 仅 clamp 门控上限 |
| Both-10 | 两者 `[-10, 10]` & `<=10` | 论文方案 |
| Both-5 | 两者 `[-5, 5]` & `<=5` | 更激进 |
| Both-20 | 两者 `[-20, 20]` & `<=20` | 更宽松 |

**测量指标**:

1. **训练稳定性**:
   - Loss spike 频率 (定义: loss 在 10 step 内增加 >50%)
   - Gradient norm 统计 (max, mean, 99th percentile per step)
   - Parameter norm 变化

2. **异常值量化**:
   - 每层 SwiGLU 输出的 max / 99.9th / 99.99th percentile
   - 每层 SwiGLU 输出的 kurtosis (峰度)
   - 异常值比例 (超过 `mu + 5*sigma` 的比例)

3. **性能指标**:
   - Final validation loss
   - Perplexity on held-out set
   - Downstream: HellaSwag, PIQA, ARC-Easy

**关键假设检验**:
- H0: Control 组与 Both-10 组的 final loss 无显著差异
- H0: Control 组与 Both-10 组的 spike 频率无显著差异
- H0: Both-5 组的 spike 频率 = Both-10 组 (即阈值不敏感)

### Phase B: 阈值敏感性消融

**目标**: 系统寻找最优 clamping 阈值。

**步骤**:
1. 固定门控上限为 10，线性分量 clamp 范围变化: `[-5,5], [-10,10], [-15,15], [-20,20], [-50,50]`
2. 固定线性分量 `[-10,10]`，门控上限变化: `2, 5, 10, 15, 20, 50`
3. 对每组配置，测量 spike 频率和 final loss

### Phase C: 理论分析 — 为什么是 10？

**分析**:
1. 对 SiLU 函数做理论分析: `SiLU(x) = x * sigmoid(x)`
   - `SiLU(10) ≈ 10 * 0.99995 ≈ 9.9995`
   - `SiLU(-10) ≈ -10 * 0.000045 ≈ -0.00045`
   - `SiLU(5) ≈ 5 * 0.9933 ≈ 4.967`
   - `SiLU(-5) ≈ -5 * 0.0067 ≈ -0.0335`
2. 10 的物理意义: 在 x >= 10 时 SiLU(x) ≈ x，clamp 到 10 几乎不改变正值行为
3. 在 x <= -10 时 SiLU(x) ≈ 0，clamp 不影响负值行为
4. 推断: 阈值 10 可能是一个 "conservative upper bound"，保证不改变正常激活但有 cap 作用

---

## 4. 所需资源

| 资源 | 规格 |
|------|------|
| GPU | 4x A100 (40GB) |
| 训练框架 | PyTorch + FSDP / DeepSpeed ZeRO-2 |
| 数据 | The Pile / C4 1B token subset |
| 时间 | Phase A: 1周 / Phase B: 额外 3-5天 / Phase C: 1-2天 |

---

## 5. 潜在风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 小模型上 spike 不出现 | 故意引入不稳定因素 (大 lr, 移除 normalization) |
| 10 这个阈值对大模型和小模型可能不同 | 在分析中讨论 scaling 行为 |
| Clamping 真正效果在大规模才显著 | 在结论中标注 scale limitation |

---

## 6. 成功标准

| 级别 | 标准 |
|------|------|
| Minimal | Phase A: 证明 clamping 在至少一种配置下显著减少 spike 频率 |
| Good | Phase A+B: 找到最优阈值范围，验证论文 10 的合理性 |
| Full | Phase A+B+C: 完整表征 clamping 机制 + 理论解释 |
