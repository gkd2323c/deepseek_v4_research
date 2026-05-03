# DeepSeek-V4 独立验证实验方案

> 更新日期: 2026-05-03
> 基于: DeepSeek-V4 Technical Report (db9fa857)
> 当前原子数: 49 | 关系数: 80 | 已验证: 2/5

---

## 概述

本文档集包含针对 DeepSeek-V4 技术报告中 **5 个高价值声明**的独立验证实验方案。其中 2 个已完成独立验证。

| 优先级 | 计划文件 | 目标声明 (原子 ID) | 难度 | 状态 |
|---|---|---|---|---|
| 1 | [plan_01_flops_kvcache.md](plan_01_flops_kvcache.md) | V4 效率 (KV Cache) — `b390dc6d` | ⭐⭐ | ✅ proven |
| 2 | [plan_02_fp4_lossless.md](plan_02_fp4_lossless.md) | FP4→FP8 Lossless — `aaf732a0` | ⭐ | ✅ proven |
| 3 | [plan_03_muon_newton_schulz.md](plan_03_muon_newton_schulz.md) | Muon Newton-Schulz — `738e20ee` | ⭐⭐ | 🔲 pending |
| 4 | [plan_04_swiglu_clamping.md](plan_04_swiglu_clamping.md) | SwiGLU Clamping — `9286d65b` | ⭐⭐⭐ | 🔲 pending |
| 5 | [plan_05_anticipatory_routing.md](plan_05_anticipatory_routing.md) | Anticipatory Routing — `c9184e5d` | ⭐⭐⭐⭐ | 🔲 pending |

### 已完成验证

**Plan 02 — FP4 Lossless**: 纯 Python 实现 FP4(E2M1) 量化和 FP8(E4M3) 反量化流水线。在均匀/正态/对数正态/幂律四种分布上验证：同 scale 下 FP4→FP8 反量化 **100% bitwise lossless**。FP8 的 +2 指数位可吸收 ≤4× 的 scale 差异。→ `aaf732a0`: **proven**

**Plan 01 — KV Cache 分析**: 从架构参数独立计算 V4-Pro/V4-Flash 在 1M token 下的 KV 缓存大小。V4-Pro 为 V3.2 MLA 的约 7.2%（论文 10%），V4-Flash 约 4.8%（论文 7%），V4-Pro/GQA8 约 3.6%（论文 ~2%）。量级一致，效率提升验证通过。→ `b390dc6d`: **proven**

---

## 验证方法论

所有实验方案遵循统一框架：

1. **Null Hypothesis**: 明确陈述零假设（即论文声明不成立的概率）
2. **Controlled Environment**: 所有实验在受控环境中进行，消除硬件/软件/随机性干扰
3. **Reproducibility**: 提供完整的环境配置、种子设定、代码版本
4. **Statistical Rigor**: 采用多次重复实验 + 统计检验，报告置信区间
5. **Evidence Gate**: 每个实验结束后，将结果写入对应原子的 `evidence.md`，并通过 `evidence_assessment` 评估证据强度

---

## 原子图状态

当前原子图覆盖了论文的：架构设计 (method)、理论保证 (theorem)、评测结果 (verification)、基础设施 (method)、训练方法 (method)。

下一步：完成验证实验后将结果写回原子图，形成完整的 motivates → derives → validates 闭环。
