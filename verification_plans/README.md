# DeepSeek-V4 独立验证实验方案

> 生成日期: 2026-05-03
> 基于: DeepSeek-V4 Technical Report (db9fa857)
> 当前原子数: 29 | 关系数: 31

---

## 概述

本文档集包含针对 DeepSeek-V4 技术报告中 **5 个高价值声明**的独立验证实验方案。这些声明在论文中证据较弱（仅依赖论文自身提供的数据/图表，缺乏独立复现或严格消融实验），但声明的技术影响较大。

| 优先级 | 计划文件 | 目标声明 (原子 ID) | 验证难度 | 预计周期 |
|---|---|---|---|---|
| 1 | [plan_01_flops_kvcache.md](plan_01_flops_kvcache.md) | V4 推理效率 (FLOPs / KV Cache) — `b390dc6d` | ⭐⭐ | 1-2 周 |
| 2 | [plan_02_fp4_lossless.md](plan_02_fp4_lossless.md) | FP4→FP8 Lossless Dequantization — `aaf732a0` | ⭐ | 3-5 天 |
| 3 | [plan_03_muon_newton_schulz.md](plan_03_muon_newton_schulz.md) | Muon 混合 Newton-Schulz 收敛 — `738e20ee` | ⭐⭐ | 1-2 周 |
| 4 | [plan_04_swiglu_clamping.md](plan_04_swiglu_clamping.md) | SwiGLU Clamping 对训练稳定性的影响 — `9286d65b` | ⭐⭐⭐ | 2-3 周 |
| 5 | [plan_05_anticipatory_routing.md](plan_05_anticipatory_routing.md) | Anticipatory Routing 训练稳定性 — `c9184e5d` | ⭐⭐⭐⭐ | 3-4 周 |

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
