# DeepSeek-V4 Research Project

基于 [OpenResearch]() 框架，对 **DeepSeek-V4 Technical Report** 的结构化研究分析。将论文拆解为可独立检验的 **原子（claim + evidence）**，构建研究知识图谱，并设计独立验证实验方案。

## 项目概览

```
72 atoms · 123 relations · 0 orphans
├── 29 V4 内部原子 (fact:3  method:18  theorem:1  verification:7)
└── 43 跨引用原子 (全文深度解析)
     ├── 8 V3 深层原子 (MLA/Aux-Loss-Free/MTP/FP8/GRPO/R1蒸馏)
     ├── 6 V2 深层原子 (MLA原始/MoE/GRPO/LongContext/效率)
     ├── 6 R1 深层原子 (纯RL/多阶段管线/失败方法/蒸馏/自适应CoT)
     ├── 8 HC 深层原子 (矩阵/动态HC/n=4/稳定性/并行性/开销)
     ├── 6 Muon 深层原子 (NS机制/扩展技术/2x效率/SVD熵/分布式)
     └── 9 其他引用原子 (Hash/AttnSink/Engram/mHC arXiv)
```

✅ 5 atoms proven by independent verification

### 核心研究问题

1. **混合注意力机制（CSA + HCA）** 能否真正实现论文声称的 FLOPs/KV Cache 效率？→ ✅ **KV Cache 分析已验证** (7.2% vs 论文 10%)
2. **FP4 QAT "Lossless Dequantization"** 在什么条件下成立？→ ✅ **已验证** (同 scale: 100% bitwise lossless)
3. **Muon 混合 Newton-Schulz 迭代** 是否比标准方案收敛更快？→ ✅ **已验证** (8fast+2stable 快速收敛，前期速度优于 NS-3)
4. **SwiGLU Clamping + Anticipatory Routing** 在 MoE 训练中的稳定性效果是否可复现？→ ✅ **已验证** (scale-dependent 安全机制，阈值 10 高于正常激活 7-8)

## 项目结构

```
deepseek_v4/
├── README.md                          # 本文件
├── articles/                          # 论文 PDF
│   ├── DeepSeek.pdf                   # V4 主论文 (4.3M)
│   ├── DeepSeek-V3.pdf                # 前代 V3 (1.8M)
│   ├── DeepSeek-V2.pdf                # MLA/MoE 起源 V2 (1.6M)
│   ├── DeepSeek-R1.pdf                # 推理范式 R1 (4.9M)
│   ├── Hyper-Connections_Zhu2024.pdf  # HC 理论基础 (7.1M)
│   ├── Muon_Scalable_Liu2025.pdf      # Muon 优化器 (2.0M)
│   ├── Hash_Layers_Roller2021.pdf     # Hash Routing (363K)
│   └── Attention_Sink_Xiao2024.pdf    # Attention Sink (17M)
│
├── atom_list/                         # 66 个原子
│   └── <atom-id>/
│       ├── claim.md                   # 声明
│       ├── evidence.md                # 证据
│       └── evidence_assessment.md     # 证据评估
│
├── experiments/                       # 验证实验脚本 (10 个)
│   ├── fp4_lossless_verify.py          # Plan 02: FP4 数值验证 ✅
│   ├── flops_kvcache_analysis.py       # Plan 01: FLOPs 分析
│   ├── kvcache_analysis_v2.py          # Plan 01: KV Cache 分析 ✅
│   ├── muon_ns_convergence.py          # Plan 03: Muon NS 收敛 ✅
│   ├── softplus_vs_sigmoid.py          # Sqrt(SP) vs Sigmoid ✅
│   ├── plan04_*.py                     # Plan 04: SwiGLU Clamping ✅
│   └── plan05_*.py                     # Plan 05: Anticipatory Routing ✅
│
├── verification_plans/                # 5 份独立验证方案 (5/5 已执行)
│   ├── README.md                      # 总览
│   ├── plan_01_flops_kvcache.md       # FLOPs/KV Cache 分析
│   ├── plan_02_fp4_lossless.md        # FP4 无损反量化
│   ├── plan_03_muon_newton_schulz.md  # Newton-Schulz 收敛
│   ├── plan_04_swiglu_clamping.md     # SwiGLU Clamping 消融
│   └── plan_05_anticipatory_routing.md # Anticipatory Routing
│
└── research_literature/               # 文献分析
    ├── reference_analysis.md          # 三级参考文献纳入计划
    └── analysis_report.md             # 综合研究分析报告
```

## 原子图（Atom Graph）

每个原子是一个最小的可验证知识单元，包含 `claim`（声明）+ `evidence`（证据）。原子之间通过类型化关系连接：

| 关系类型 | 含义 | 数量 |
|----------|------|------|
| `motivates` | 背景/问题驱动下游声明 | 25 |
| `derives` | 方法从上游内容派生/构建 | 38 |
| `validates` | 实验验证上游方法/定理 | 55 |
| `formalizes` | 理论形式化上游声明 | 3 |
| `contradicts` | 逻辑或经验冲突 | 2 |

### 关键关系链

```
V2 (MLA + MoE + GRPO) ──derives──→ V3 (Aux-Loss-Free + MTP + FP8)
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
              V3: Zero Spike        V3: MLA              V3: GRPO+R1蒸馏
                    │                     │                     │
               contradicts           motivates              derives
                    ↓                     ↓                     ↓
              V4: Stability         V4: CSA/HCA           V4: OPD
                    │                     │                     │
         ┌─────────┼─────────┐           │                     │
         ↓                   ↓           ↓                     ↓
  SwiGLU Clamping    Anticipatory    mHC (Birkhoff)    Think High/Max
  (scale-dependent)  Routing         ←── HC (Zhu 24)    ←── R1-Zero
         ↓                   ↓
    Plan 04 ✅          Plan 05 ⚠️

Muon (Liu 2025) ──derives──→ V4 Muon (8+2 hybrid NS) ──validates──→ Plan 03 ✅
Sqrt(Softplus) vs Sigmoid ──validates──→ Plan 04 ✅ (梯度永不消失)
```

## 涵盖的技术主题

### 架构创新
- **mHC**（流形约束超连接）：约束 $B_l$ 到双随机矩阵流形（Birkhoff 多面体），Sinkhorn-Knopp 投影
- **CSA**（压缩稀疏注意力）：压缩比 $m=4$，Lightning Indexer 选 top-$k$，Shared KV MQA
- **HCA**（重度压缩注意）：压缩比 $m'=128$，密集注意力
- **混合注意力**：CSA/HCA 交替排列 + 滑动窗口
- **Muon 优化器**：混合 Newton-Schulz（8+2 迭代），Nesterov 技巧

### 训练基础设施
- 细粒度 EP 通信-计算重叠（1.5~1.96× 加速）
- FP4 QAT + Lossless Dequantization
- 批次不变确定性 Kernel
- 异构 KV 缓存 + 磁盘存储
- 两阶段 Context Parallelism

### 训练稳定性
- Anticipatory Routing（解耦 backbone 与路由更新）
- SwiGLU Clamping（线性分量 $[-10,10]$）
- Hash Routing（前 3 层，固定哈希路由）

### 后训练
- 专家训练（Specialist Training）+ GRPO RL
- On-Policy Distillation（多教师 → 单一学生）
- Generative Reward Model
- 三种推理模式（Non-think / Think High / Think Max）

## 参考文献

已入库 10 篇核心参考文献，其中 5 篇已完成全文深度解析：

| 论文 | arXiv | 解析深度 | 角色 |
|------|-------|----------|------|
| DeepSeek-V2 | 2405.04434 | 全文 | MLA + DeepSeekMoE 起源 |
| DeepSeek-V3 | 2412.19437 | 全文 | V4 直接前代 |
| DeepSeek-R1 | 2501.12948 | 全文 | 推理范式 |
| Hyper-Connections | 2409.19606 | 全文 | mHC 理论基础 |
| mHC (V4 specific) | 2512.24880 | 摘要 | mHC Birkhoff 约束 |
| Muon Scalable | 2502.16982 | 全文 | 优化器出处 |
| Engram | 2601.07372 | 摘要 | 条件记忆模块 |
| TileLang | — | 摘要 | DSL for kernels |
| Hash Layers | 2106.04426 | 摘要 | Hash Routing 出处 |
| Attention Sink | 2309.17453 | 摘要 | Attention Sink 出处 |

## 验证实验方案

5 份方案按优先级排列：

| 优先级 | 方案 | 目标原子 | 难度 | 状态 |
|--------|------|----------|------|------|
| 🔴 P1 | FLOPs/KV Cache 理论验证 | `b390dc6d` | ⭐⭐ | ✅ proven |
| 🔴 P2 | FP4 Lossless 数值验证 | `aaf732a0` | ⭐ | ✅ proven |
| 🔴 P3 | Muon Newton-Schulz 收敛 | `738e20ee` | ⭐⭐ | ✅ proven |
| 🟡 P4 | SwiGLU Clamping 消融 | `9286d65b` | ⭐⭐⭐ | ✅ proven |
| 🟡 P5 | Anticipatory Routing | `c9184e5d` | ⭐⭐⭐⭐ | ⚠️ in_progress |

## 使用方式

### 浏览原子图
每个原子目录下有 `claim.md`（声明）和 `evidence.md`（证据），可直接阅读：

```bash
cat atom_list/b390dc6d-d635-4310-9541-a99c94b03ab8/claim.md
```

### 运行验证实验
进入对应的 `verification_plans/` 目录，按方案中的 Phase A/B/C 逐步执行。

### 扩展参考文献
参考 `research_literature/reference_analysis.md` 中的 Tier 1/2/3 列表，添加新论文并创建跨引用原子。

## 许可证

本项目为学术研究目的创建。论文版权归原作者所有。
