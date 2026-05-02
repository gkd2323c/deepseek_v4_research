# Plan 01: 验证 V4 推理效率声明 (FLOPs & KV Cache)

**目标原子**: `b390dc6d` — "V4 Efficiency: FLOPs and KV Cache vs V3.2 at 1M Context"

---

## 1. 声明回顾

论文声称：
- V4-Pro 在 1M token 上下文下，单 token 推理 FLOPs 仅为 V3.2 的 **27%**
- V4-Pro 在 1M token 上下文下，累积 KV Cache 仅为 V3.2 的 **10%**
- V4-Flash：FLOPs 为 V3.2 的 **10%**，KV Cache 为 V3.2 的 **7%**
- 对比用 V3.2 的 MLA (Multi-head Latent Attention) 作为 baseline

证据来源：论文 Figure 1 (右侧两幅图表)，未提供具体数值、硬件配置、或误差范围。

---

## 2. 核心问题

论文中的 **27% / 10%** 数字是**分析估计**还是**实测数据**？Figure 1 的图表无法验证其测量方法。需要独立验证：

1. **FLOPs 分析一致性**：从架构参数推导出理论 FLOPs 计数，与论文声明对比
2. **KV Cache 大小验证**：从架构参数计算实际 KV Cache 大小（bytes），与论文声明对比
3. **实测性能 (如权重可用)**：在相同硬件上 benchmark

---

## 3. 实验设计

### 3.1 理论 FLOPs 分析 (Phase A)

**目标**: 从架构参数出发，重新计算注意力模块的 FLOPs，验证 27%/10% 声明是否与分析结果一致。

**步骤**:

1. **定义 FLOPs 计数模型**:
   - 对于 CSA 层：计算 Compression + Lightning Indexer + Sparse MQA + Grouped Output + SWA 的 FLOPs
   - 对于 HCA 层：计算 Compression + Dense MQA + Grouped Output + SWA 的 FLOPs
   - 基准 V3.2：计算 MLA 每层的 FLOPs

2. **参数代入** (V4-Pro):
   - 59 层 interleaved CSA/HCA (30 CSA + 29 HCA) + 2 层 HCA
   - CSA: $m=4$, $n_h=128$, $c=512$, $d_c=1536$, top-$k=1024$, $g=16$, $d_g=1024$
   - HCA: $m'=128$, $n_h=128$, $c_I=128$, $g=16$, $d_g=1024$
   - SWA: $n_{win}=128$

3. **FLOPs 公式推导** (以 CSA 为例):

   Compression stage:
   $$FLOPs_{comp} \approx S \cdot \frac{2}{m} \cdot (n_h \cdot c \cdot d_c + n_h \cdot c)$$

   Indexer + Sparse MQA:
   $$FLOPs_{attn} \approx S \cdot (n_h^I \cdot c_I \cdot d_c + n_h^I \cdot c_I) + S \cdot k \cdot (n_h \cdot d_c + n_h \cdot c + n_h \cdot d_c)$$

   Grouped Output:
   $$FLOPs_{out} \approx S \cdot (g \cdot d_g \cdot d_c + d_c^2)$$

   其中 $S$ 为序列长度。

4. **FLOPs 比值计算**: 对 $S = 1\text{M}$ 计算 $FLOPs_{V4} / FLOPs_{V3.2}$

**预期判断标准**: 
- 若计算比值在 **25%~29%** → 与论文一致
- 若计算比值在 **20%~34%** → 部分一致，需进一步分析差异
- 若计算比值 **<20% 或 >34%** → 论文声明需质疑

### 3.2 KV Cache 分析 (Phase B)

**目标**: 从架构参数计算实际 KV Cache 大小。

**步骤**:

1. **V3.2 MLA Baseline**:
   - MLA 将 KV 投影到低维潜在空间 $d_{kv}$
   - 每个 token 的 KV cache = $L_{layers} \cdot 2 \cdot d_{kv} \cdot bytes\_per\_elem$
   - $d_{kv} = 512$ (MLA), 1M tokens → 计算总缓存

2. **V4-Pro KV Cache**:
   - 31 CSA 层：压缩后每 $m=4$ token 产生 1 个 compressed entry
     $$KV_{CSA} = \frac{S}{m} \cdot 31 \cdot 2 \cdot c \cdot bytes\_per\_elem$$
   - 30 HCA 层：压缩后每 $m'=128$ token 产生 1 个 compressed entry
     $$KV_{HCA} = \frac{S}{m'} \cdot 30 \cdot 2 \cdot c \cdot bytes\_per\_elem$$
   - SWA 分支：$n_{win}$ 个未压缩 token
     $$KV_{SWA} = 61 \cdot n_{win} \cdot d_c \cdot bytes\_per\_elem$$
   - 总 $KV_{V4} = KV_{CSA} + KV_{HCA} + KV_{SWA}$

3. **混合精度因素**: 
   - BF16 for RoPE dims (64 dims), FP8 for remaining ($c-64$ dims)
   - 实际 $KV_{V4}^{actual} = KV_{CSA}^{actual} + ...$ 

4. **比值计算**: $KV_{V4} / KV_{V3.2}$ at $S = 1\text{M}$

**预期判断标准**:
- 若比值在 **8%~12%** → 与论文一致
- 若比值 **<8% 或 >12%** → 需质疑

### 3.3 实测性能 Benchmark (Phase C, 条件化)

**前置条件**: DeepSeek-V4 权重已发布且可在指定硬件上运行。

**步骤**:

1. **硬件**: 2x NVIDIA H100 (80GB) 或等价硬件
2. **软件**: vLLM / SGLang (支持 DeepSeek-V4 architecture)
3. **指标测量**:
   - `torch.cuda.Event` 精确计时 per-token latency
   - `torch.cuda.max_memory_allocated()` 测量 KV Cache 峰值
   - 对序列长度 $[1K, 4K, 16K, 64K, 128K, 256K, 512K, 1M]$ 分别测量
4. **对比基线**: V3.2 模型在相同硬件/软件栈下
5. **重复**: 每个配置 100 次推理，报告 mean ± std

---

## 4. 所需资源

| 资源 | 规格 |
|---|---|
| GPU | 2x H100 (80GB) 或 4x A100 (40GB) |
| 软件 | PyTorch 2.5+, vLLM 0.7+, FlashInfer |
| 时间 | Phase A+B: 3-5 天 / Phase C: 额外 5-7 天 |
| 模型权重 (Phase C) | DeepSeek-V4-Pro (1.6T), V3.2 (685B) |

---

## 5. 风险与失败分析

| 风险 | 缓解措施 |
|---|---|
| 权重未发布 | Phase C 无法执行，仅依赖 Phase A+B 分析验证 |
| 论文中的 FLOPs 定义为 "equivalent FP8 FLOPs"，与标准 FLOPs 有差异 | 分别计算 FP8 和标准 FLOPs 两种版本 |
| MLA 内部实现细节未完全公开 | 基于 V3 论文的公开描述最佳估算 |
| GPU 显存不足以加载 1M 上下文 | 使用 CPU offloading 或减小序列长度做外推 |
| 论文 V3.2 可能与开源 V3 有差异 | 以开源 V3 为 baseline，标注差异 |

---

## 6. 成功标准

| 级别 | 标准 |
|---|---|
| **Minimal** | Phase A 完成：FLOPs 分析值与论文 27% 偏差 < 5 个百分点 |
| **Good** | Phase A+B 完成：FLOPs + KV Cache 分析值均与论文一致 |
| **Full** | Phase A+B+C 完成：实测数据与论文声明误差在 10% 以内 |
