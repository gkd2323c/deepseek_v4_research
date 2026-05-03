# DeepSeek-V4 参考文献纳入分析

> 基于: DeepSeek-V4 Technical Report (db9fa857-d1f5-434c-b0de-b20b1661ab0f)
> 当前图: 29 atoms, 39 relations
> 目标: 将关键参考文献纳入研究项目，扩展原子图

---

## Tier 1: 直接依赖 — 理解 V4 必须读的论文

这些论文是 V4 创新的直接基础，缺失它们会导致原子图的 derives/formalizes 链断裂。

| # | 论文 | V4 中的角色 | 连接到的现有原子 | 优先级 |
|---|---|---|---|---|
| 1 | **DeepSeek-V3** (2024) | V4 的直接前代；MLA、MTP、aux-loss-free 负载均衡均来自 V3 | `3e5cd741` (Model Specs), `b365a813`, `3dd80c1c` | 🔴 最高 |
| 2 | **Hyper-Connections** (Zhu et al., 2025) | mHC 的基础框架；V4 将其扩展为流形约束版本 | `3cae5a37` (mHC), `0f41420e` (mHC Theorem) | 🔴 最高 |
| 3 | **Muon Optimizer** (Keller et al.) | V4 的优化器核心，需理解基础 Muon 才能分析 Newton-Schulz 改进 | `738e20ee` (Muon), Plan 03 | 🔴 最高 |
| 4 | **Hash Layers** (Roller et al., 2021) | V4 前 3 层 MoE 采用 Hash Routing 的原始出处 | `6482f088` (Hash Routing) | 🟡 高 |
| 5 | **Attention Sink** (Xiao et al., 2024) | CSA/HCA 中 attention sink 技术的原始来源 | `0d94ff02` (Attention Sink) | 🟡 高 |

## Tier 2: 重要上下文 — 验证和理解 V4 创新的基础

| # | 论文 | V4 中的角色 | 连接到的现有原子 | 优先级 |
|---|---|---|---|---|
| 6 | **DeepSeek-V2** (2024) | DeepSeekMoE 架构的提出者；细粒度专家 + 共享专家 | `b365a813` (SqrtSoftplus), MoE 相关 | 🟡 高 |
| 7 | **DeepSeek-R1** (2025) | V4 推理模式 (Think High/Max) 的范式来源；GRPO RL | `4af43fd0` (Reasoning Modes), `d042bc26` | 🟡 高 |
| 8 | **Multi-Token Prediction** (Gloeckle et al., 2024) | MTP 策略的提出；V4 继承自 V3 | `3dd80c1c` (Pre-training) | 🟡 高 |
| 9 | **GRPO** (Shao et al., 2024) | Specialist Training 中使用的 RL 算法 | `d042bc26` (OPD/Specialist) | 🟢 中 |
| 10 | **FlashAttention** (Dao et al., 2022) | 高效 attention kernel 的基础；理解 V4 的 kernel 优化 | `dcffaf87` (Kernels), `68eae2f7` (CP) | 🟢 中 |
| 11 | **MX Microscaling** (OCP spec) | FP4 (MXFP4) 量化格式的规范；Plan 02 需要 | `aaf732a0` (FP4 QAT), Plan 02 | 🟢 中 |

## Tier 3: 比较基准 & 背景 — 完整理解论文上下文

| # | 论文 | 角色 | 连接 |
|---|---|---|---|
| 12 | **Vanilla Transformer** (Vaswani et al., 2017) | 注意力机制的起源 | `dbfe68f2` (Quadratic Bottleneck) |
| 13 | **Sparsely-Gated MoE** (Shazeer et al., 2017) | MoE 范式的开创者 | `266f1dd4` (Stability), `b365a813` |
| 14 | **Switch Transformers** (Fedus et al., 2022) | MoE 训练的稳定性技术 | `266f1dd4`, `9286d65b` |
| 15 | **RingAttention** (Liu et al., 2023) | V4 效率对比的 baseline 之一 | `b390dc6d` (Efficiency) |
| 16 | **Mamba/SSM** (Gu & Dao, 2023) | 长上下文替代方案 | `dbfe68f2`, `5c9eee15` |
| 17 | **Sinkhorn-Knopp** (Sinkhorn, 1964) | mHC 约束投影的数学基础 | `0f41420e` (mHC Theorem) |
| 18 | **Newton-Schulz Iteration** (Higham, 2008) | Muon 中正交化的数学基础 | `738e20ee` (Muon), Plan 03 |
| 19 | **OPD** (On-Policy Distillation, 相关论文) | V4 后训练中的蒸馏方法 | `d042bc26` |
| 20 | **Generative Reward Model** (DeepSeek, 2025?) | GRM 用于 hard-to-verify 任务 | `b3c31cb6` |

---

## 纳入策略

### Phase 1: 立即可做 (Tier 1, 5 篇)

这些论文是理解 V4 原子图中关键 derivations 的前提。
每篇论文入库后会：
1. 添加为新的 article
2. 提取相关的 claim+evidence 原子
3. 与现有 V4 原子建立 cross-tree 关系

例如: Hyper-Connections (Zhu et al., 2025) →
- 创建 "Standard HC formulation" atom
- 创建关系: Standard HC → motivates → mHC (V4's improvement)

### Phase 2: 扩展上下文 (Tier 2, 6 篇)

提供 V4 创新的技术背景。特别对验证实验 (Plans 01-05) 至关重要。

### Phase 3: 完整图景 (Tier 3, 9 篇)

建立完整的引用关系图，支持跨论文的综合性研究问题。

---

## 下载状态

| # | 论文 | 文件 | 大小 | PDF | 原子 |
|---|---|---|---|---|---|
| 1 | DeepSeek-V3 | `DeepSeek-V3.pdf` | 1.8M | ✅ | ✅ 摘要 |
| 2 | DeepSeek-V2 | `DeepSeek-V2.pdf` | 1.6M | ✅ | ✅ 摘要 |
| 3 | DeepSeek-R1 | `DeepSeek-R1.pdf` | 4.9M | ✅ | ✅ 摘要 |
| 4 | Hash Layers (Roller 2021) | `Hash_Layers_Roller2021.pdf` | 363K | ✅ | ✅ 摘要 |
| 5 | Attention Sink (Xiao 2024) | `Attention_Sink_Xiao2024.pdf` | 17M | ✅ | ✅ 摘要 |
| 6 | Hyper-Connections (Zhu 2024) | `Hyper-Connections_Zhu2024.pdf` | 7.1M | ✅ | ✅ 全文 (7 atoms) |
| 7 | Muon Scalable (Liu 2025) | `Muon_Scalable_Liu2025.pdf` | 2.0M | ✅ | ✅ 全文 (5 atoms) |

## 图统计

- 原始 V4 原子: 29
- 浅层引用原子 (摘要): 8
- 深层引用原子 (HC全文 + Muon全文): 12
- 总计: **49 atoms, 80 relations, 0 orphans**

## 验证状态

| 方案 | 原子 | 状态 |
|------|------|------|
| Plan 02 — FP4 Lossless | `aaf732a0` | ✅ proven (100% bitwise) |
| Plan 01 — KV Cache | `b390dc6d` | ✅ proven (7.2%, order-of-magnitude confirmed) |
| Plan 03 — Muon NS | `738e20ee` | 🔲 pending |
| Plan 04 — SwiGLU | `9286d65b` | 🔲 pending |
| Plan 05 — AR | `c9184e5d` | 🔲 pending |
| 7 | Muon Optimizer (Keller) | — | — | 🔍 待定位 |

## 图统计

- 原始 V4 原子: 29
- 引用原子: 6
- 总计: **35 atoms, 51 relations, 0 orphans**

## 下一步

1. 定位 Hyper-Connections 和 Muon 的论文来源并下载
2. 对已下载的 5 篇论文进行全文解析，提取深层原子
3. 扩展 Tier 2/Tier 3 论文
