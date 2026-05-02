# Plan 05: 验证 Anticipatory Routing 对训练稳定性的效果

**目标原子**: `c9184e5d` — "Anticipatory Routing for Training Stability"

---

## 1. 声明回顾

论文声称:
- Anticipatory Routing "decouples synchronous updates of the backbone and routing network"
- 路由索引在 timestep t 使用历史参数 `theta_{t-Delta_t}` (预先在 `t-Delta_t` 计算好)
- 仅在 loss spike 检测到时动态激活，稳定后恢复
- 激活时 wall-clock 开销约 ~20%
- "effectively maintains training stability"

**核心问题**:
- 动态激活/停用机制的 spike 检测阈值是什么？
- 20% overhead 是否可接受？
- 是否在所有 spike 场景下都有效？

---

## 2. 机制分析

### 当前路由问题的诊断

论文观察到: loss spike 与 MoE 层的 outlier 一致出现，路由机制本身加剧异常值。

机制推测:
1. 当前参数 `theta_t` 同时影响 backbone 输出和路由决策
2. 当 backbone 输出出现异常时 → 路由选择错误的专家
3. 错误路由 → 专家负载不均衡 → 梯度异常 → 加剧 backbone 异常
4. 形成恶性循环

Anticipatory Routing 通过**延迟路由更新**打破这个正反馈循环: 
- 路由使用 `theta_{t-Delta_t}` (历史稳定参数)
- Backbone 使用 `theta_t` (当前参数)
- 失去同步性破坏了恶性循环的条件

### 动态激活机制

论文描述了 "automatic detection mechanism":
- 检测到 loss spike → 触发 small rollback (回退到 spike 前的 checkpoint)
- 重新开始，此时激活 Anticipatory Routing
- 训练恢复稳定后 → 停用 Anticipatory Routing
- 恢复正常的同步更新

---

## 3. 实验设计

### Phase A: 可控异常注入实验

**目标**: 在受控条件下重现路由导致的异常，并验证 Anticipatory Routing 的解耦效果。

**模型**: 小型 MoE Transformer (同 Plan 04)
- Layers = 12, d_model = 768
- 4 个路由专家 + 1 个共享专家
- 所有层 MoE (无 dense FFN 层)

**异常注入方案**:

1. **梯度噪声注入**: 随机在 backbone 的某层注入高斯噪声 `N(0, sigma^2)` 到梯度中
2. **路由扰动**: 随机将 10% tokens 路由到错误专家
3. **自然不稳定**: 使用高学习率 (2x optimal) 训练

**实验条件**:

| 组 | 条件 |
|----|------|
| Baseline-Normal | 正常训练，无异常注入 |
| Baseline-Unstable | 异常注入，无 Anticipatory Routing |
| AR-Fixed | 异常注入，Anticipatory Routing 始终激活 |
| AR-Dynamic | 异常注入，Anticipatory Routing 仅 spike 后激活 (模拟论文) |
| AR-Small-Delta | AR 激活，Delta_t = 1 (最小解耦) |
| AR-Large-Delta | AR 激活，Delta_t = 100 (最大解耦) |

**测量指标**:

1. **Spike 统计**:
   - Spike 发生频率 (每 1000 step)
   - Spike 严重程度 (loss 增加百分比)
   - Spike 恢复时间 (步数)

2. **路由健康度**:
   - 专家负载分布的 entropy (越高越好)
   - 最忙/最闲专家负载比
   - 路由 dropout 率

3. **开销度量**:
   - Per-step wall-clock time (AR on vs off)
   - 额外内存开销 (cached routing indices)
   - AR 相关的额外通信量 (多节点场景)

4. **最终性能**:
   - Final validation loss
   - Perplexity

### Phase B: Delta_t 消融研究

**目标**: 找到最优解耦步数 Delta_t。

**步骤**:
1. 测试 Delta_t = {1, 5, 10, 20, 50, 100, 200, 500, 1000}
2. 在异常注入条件下，每个 Delta_t 运行 3 次
3. 测量: spike 频率、final loss、AR overhead
4. 寻找 Pareto-optimal Delta_t

### Phase C: Spike 检测阈值敏感性

**目标**: 验证动态激活机制的健壮性。

**检测器实现**:
- 滑动窗口 loss 监测 (窗口大小 W = 50 steps)
- 当 `loss_t > mean(loss_{t-W:t-1}) + k * std(loss_{t-W:t-1})` 时触发

**消融**: 
- k = {2, 3, 4, 5, 6} (可以看作 Z-score 阈值)
- W = {20, 50, 100, 200}
- Rollback 步数 = {10, 50, 100, 500}

**测量**: false positive rate, false negative rate, recovery time

---

## 4. 所需资源

| 资源 | 规格 |
|------|------|
| GPU | 4x A100 (40GB) |
| 训练框架 | PyTorch + FSDP |
| 时间 | Phase A: 1周 / Phase B: 5天 / Phase C: 3天 |

---

## 5. 限制与讨论

| 限制 | 说明 |
|------|------|
| Scale Gap | 1.6T scale 的动力学可能在 100M scale 上无法重现 |
| 异常模式 | 人工注入的噪声可能与真实训练中的不稳定模式不同 |
| 路由负载均衡 | Small-scale MoE 的负载自然比 large-scale 更均衡 |
| Delta_t 值 | 论文未提供具体值，我们的探索是基于推测的 |

---

## 6. 成功标准

| 级别 | 标准 |
|------|------|
| Minimal | Phase A: 证明 AR 在至少一种异常场景下显著减少 spike 频率 |
| Good | Phase A+B: 量化 AR 的开销-收益 trade-off, 找到合理 Delta_t |
| Full | Phase A+B+C: 完整的 AR 行为表征 + 最优参数推荐 |
