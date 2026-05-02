# Plan 02: 验证 FP4→FP8 Lossless Dequantization

**目标原子**: `aaf732a0` — "FP4 QAT with Lossless Dequantization"

---

## 1. 声明回顾

论文声称：
1. FP4 (MXFP4 E2M1) 量化应用于 MoE 专家权重和 CSA Indexer 的 QK 路径
2. FP4→FP8 (E4M3) 反量化是 **lossless** 的
3. 原因是 FP8 比 FP4 多 2 个指数位，可以吸收 scale factor 的差异
4. Lossless 的条件：FP4 sub-blocks 的 max/min scale factor 比值不超过阈值，且 "empirically verify that current weights satisfy this condition"

**核心问题**: 什么条件下 FP4→FP8 反量化是真正 lossless 的？论文权重是否真的满足这些条件？需要量化验证。

---

## 2. 关键分析

### 2.1 量化格式回顾

- **FP4 (E2M1)**: 1 sign + 2 exponent + 1 mantissa → 可表示 $\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ (乘以 $2^{exp\_bias}$)
- **FP8 (E4M3)**: 1 sign + 4 exponent + 3 mantissa → 可表示 $\pm\{0, 0.001953125, ..., 448\}$

### 2.2 Lossless 条件推导

FP4 量化过程：
$$x_{fp4} = Q_{fp4}(x, s_{fp4}) = \text{round}\left(\frac{x}{s_{fp4}}\right)_{fp4} \cdot s_{fp4}$$

FP4→FP8 反量化：
$$x_{fp8} = x_{fp4} \cdot \frac{s_{fp4}}{s_{fp8}}$$

Lossless 条件：$s_{fp4} / s_{fp8}$ 必须是精确可表示的 FP8 值，且乘积在 FP8 动态范围内。

设 $s_{fp4\_max} / s_{fp4\_min} = r$ （同一 FP8 block 内各 FP4 sub-block 的 scale 比值）

**关键不等式**: 
$$\lceil \log_2(r) \rceil \leq 2 \quad \text{(FP8 比 FP4 多 2 exponent bits)}$$

即 $r \leq 4$ 时理论上 lossless。

---

## 3. 实验设计

### Phase A: 纯数值验证

**目标**: 在合成数据上验证 lossless 条件的严格性。

**步骤**:

1. **实现 FP4/FP8 量化-反量化流水线**:
```python
def fp4_quantize(x, block_size=32):
    """MXFP4 block-wise quantization"""
    # reshape into blocks
    # compute per-block max → scale
    # quantize to E2M1 format
    pass

def fp4_to_fp8_dequant(x_fp4, scales_fp4, scales_fp8):
    """Lossless dequant: FP4 → FP8"""
    # scale_ratio = scales_fp4 / scales_fp8
    # multiply and cast to FP8
    pass
```

2. **测试矩阵构造**:
   - **Case A**: 均匀分布 $X \sim \mathcal{U}(-1, 1)$
   - **Case B**: 正态分布 $X \sim \mathcal{N}(0, 1)$
   - **Case C**: 幂律分布 (模拟 MoE 权重的长尾特征)
   - **Case D**: 结构化矩阵 (模拟 attention QK 投影的低秩特征)

3. **测量指标**:
   - **Max Error**: $\max_i |x_i - \hat{x}_i| / |x_i|$
   - **RMSE**: $\sqrt{\frac{1}{n}\sum (x_i - \hat{x}_i)^2}$
   - **Lossless Rate**: $\mathbb{P}(x_i = \hat{x}_i \text{ bitwise})$
   - **Scale Ratio $r$**: 统计 $r$ 的分布

4. **Block 大小消融**: 测试 block_size $\in \{16, 32, 64, 128\}$ 的影响

5. **Scale Ratio 阈值扫描**: 
   - 构造不同 $r$ 值的矩阵，找到 lossless→lossy 的临界点
   - 验证 **$r \leq 4$** 是否确实是充分条件

### Phase B: 真实权重验证 (条件化)

**前置条件**: 可以获取 DeepSeek-V4 的量化权重。

**步骤**:

1. 加载 MoE 专家权重矩阵 $W_{expert} \in \mathbb{R}^{d_{ff} \times d_{model}}$
2. 加载 CSA Indexer QK 投影权重
3. 对每层每个矩阵执行 FP4→FP8 反量化
4. 统计：
   - 每层的 max error / RMSE 分布
   - 跨层 scale ratio 统计
   - 识别不满足 lossless 条件的异常层

### Phase C: 端到端输出保真度验证

**目标**: 验证 FP4 量化对模型输出的影响。

**步骤**:

1. 以 FP8/BF16 精度运行一次 forward pass，保存每层的输出
2. 以 FP4 QAT 精度运行同样的 forward pass
3. 逐层比较 cosine similarity 和 L2 误差

---

## 4. 所需资源

| 资源 | 规格 |
|---|---|
| 计算 | 单 GPU (A100 或以上)，甚至 CPU 即可完成 Phase A |
| 软件 | PyTorch + 自定义 FP4/FP8 模拟 kernel |
| 时间 | Phase A: 2-3 天 / Phase B: 1-2 天 / Phase C: 2-3 天 |

---

## 5. 风险

| 风险 | 缓解措施 |
|---|---|
| paper 中 FP4 格式为 MXFP4 (Microscaling)，需确认具体规范 | 参考 OCP MX 规范，实现多版本 |
| 实际权重不可获取 | Phase A 合成矩阵验证已足够有意义 |
| FP8 硬件不普遍 | 使用软件模拟 (`torch.finfo`) 验证，不影响数学正确性 |
| "lossless" 的严格定义模糊 (bitwise vs. 特定精度下) | 同时报告 bitwise 和 $\epsilon$-lossless (如 < 1e-6 相对误差) |

---

## 6. 成功标准

| 级别 | 标准 |
|---|---|
| **Minimal** | 完成 Phase A：确认 $r \leq 4$ 是 lossless 的充分条件，量化误差分布表征 |
| **Good** | Phase A+B：真实权重满足 lossless 条件的比例 > 99.9% |
| **Full** | Phase A+B+C：端到端输出差异 < $10^{-6}$ (cosine sim > 0.999999) |
