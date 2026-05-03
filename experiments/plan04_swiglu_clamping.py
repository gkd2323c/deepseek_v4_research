"""Plan 04: SwiGLU Clamping Training Stability Experiment
Small MoE model with DirectML (AMD 7800XT) acceleration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
import time
import random
import math

# Setup
dml = torch_directml.device(0)
print(f"Device: {dml}")
print(f"PyTorch: {torch.__version__}")

# Reproducibility
torch.manual_seed(42)
random.seed(42)

# === Model Configuration ===
class Config:
    vocab_size = 1000
    d_model = 256
    n_heads = 4
    n_layers = 6
    d_ff = 512
    n_experts = 4
    n_shared = 1
    n_activated = 2
    max_seq_len = 128
    dropout = 0.1

# === SwiGLU with configurable clamping ===
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, clamp_linear=None, clamp_gate=None):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.clamp_linear = clamp_linear  # (min, max) or None
        self.clamp_gate = clamp_gate      # max or None
    
    def forward(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)
        
        # Apply clamping
        if self.clamp_linear is not None:
            up = up.clamp(min=self.clamp_linear[0], max=self.clamp_linear[1])
        if self.clamp_gate is not None:
            gate = gate.clamp(max=self.clamp_gate)
        
        # SwiGLU: SiLU(gate) * up
        silu_gate = gate * torch.sigmoid(gate)
        return self.w_down(silu_gate * up)

# === MoE Layer ===
class MoELayer(nn.Module):
    def __init__(self, cfg, clamp_linear=None, clamp_gate=None):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_activated = cfg.n_activated
        self.n_shared = cfg.n_shared
        
        # Shared experts
        self.shared_experts = nn.ModuleList([
            SwiGLU(cfg.d_model, cfg.d_ff, clamp_linear, clamp_gate)
            for _ in range(cfg.n_shared)
        ])
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            SwiGLU(cfg.d_model, cfg.d_ff, clamp_linear, clamp_gate)
            for _ in range(cfg.n_experts)
        ])
        
        # Router
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
    
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        
        # Router logits
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.n_activated, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        
        # Shared experts
        for expert in self.shared_experts:
            output += expert(x_flat) / self.n_shared
        
        # Routed experts
        for i in range(self.n_activated):
            expert_idx = topk_indices[:, i]
            expert_weight = topk_weights[:, i:i+1]
            for j in range(self.n_experts):
                mask = (expert_idx == j)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.routed_experts[j](expert_input)
                    output[mask] += expert_weight[mask] * expert_output
        
        return output.view(B, T, D)

# === Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, cfg, use_moe=False, clamp_linear=None, clamp_gate=None):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model)
        self.ffn_norm = nn.RMSNorm(cfg.d_model)
        
        # Simple multi-head attention
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        
        # FFN or MoE
        if use_moe:
            self.ffn = MoELayer(cfg, clamp_linear, clamp_gate)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, clamp_linear, clamp_gate)
    
    def forward(self, x):
        # Attention with residual
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # FFN with residual
        normed = self.ffn_norm(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x

# === Full Model ===
class MoETransformer(nn.Module):
    def __init__(self, cfg, clamp_linear=None, clamp_gate=None):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg, use_moe=(i >= cfg.n_layers // 2),
                           clamp_linear=clamp_linear, clamp_gate=clamp_gate)
            for i in range(cfg.n_layers)
        ])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

# === Training Data Generator ===
def generate_data(cfg, n_batches, seq_len=128):
    """Generate random token sequences for training."""
    data = []
    for _ in range(n_batches):
        tokens = torch.randint(0, cfg.vocab_size, (1, seq_len))
        # Simple pattern: predict next token
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        data.append((inputs, targets))
    return data

# === Training Loop with Metrics ===
def train_model(model, data, cfg, n_steps=200, lr=1e-3):
    model = model.to(dml)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    losses = []
    grad_norms = []
    outlier_counts = []
    
    model.train()
    for step, (inputs, targets) in enumerate(data[:n_steps]):
        inputs, targets = inputs.to(dml), targets.to(dml)
        
        # Forward
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Record gradient norms
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(total_norm)
        grad_norms.append(grad_norm)
        
        # Detect outliers (gradient norm > 10x average)
        if len(grad_norms) > 10:
            avg_norm = sum(grad_norms[-10:]) / 10
            if grad_norm > 10 * avg_norm:
                outlier_counts.append(step)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
    
    return {
        'losses': losses,
        'grad_norms': grad_norms,
        'outlier_steps': outlier_counts,
        'final_loss': losses[-1] if losses else 0,
        'max_grad': max(grad_norms) if grad_norms else 0,
        'mean_grad': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
    }

# === Experiment Configurations ===
configs = {
    'Control (no clamp)': {'clamp_linear': None, 'clamp_gate': None},
    'Linear [-10, 10]': {'clamp_linear': (-10, 10), 'clamp_gate': None},
    'Gate <= 10': {'clamp_linear': None, 'clamp_gate': 10},
    'Both [-10,10] + Gate<=10': {'clamp_linear': (-10, 10), 'clamp_gate': 10},
    'Both [-5,5] + Gate<=5': {'clamp_linear': (-5, 5), 'clamp_gate': 5},
    'Both [-20,20] + Gate<=20': {'clamp_linear': (-20, 20), 'clamp_gate': 20},
}

# === Main Experiment ===
print("=" * 65)
print("Plan 04: SwiGLU Clamping Training Stability Experiment")
print("=" * 65)

cfg = Config()
n_steps = 200
n_seeds = 3

# Generate shared data
print(f"\nGenerating training data ({n_steps} steps)...")
data = generate_data(cfg, n_steps)

results = {}

for name, params in configs.items():
    print(f"\n--- {name} ---")
    
    seed_results = []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 100)
        
        model = MoETransformer(cfg, **params)
        n_params = sum(p.numel() for p in model.parameters())
        
        result = train_model(model, data, cfg, n_steps)
        seed_results.append(result)
    
    # Average across seeds
    avg_result = {
        'final_loss': sum(r['final_loss'] for r in seed_results) / n_seeds,
        'max_grad': sum(r['max_grad'] for r in seed_results) / n_seeds,
        'mean_grad': sum(r['mean_grad'] for r in seed_results) / n_seeds,
        'total_outliers': sum(len(r['outlier_steps']) for r in seed_results),
        'loss_std': (sum((r['final_loss'] - sum(r2['final_loss'] for r2 in seed_results)/n_seeds)**2 
                      for r in seed_results) / n_seeds) ** 0.5,
    }
    results[name] = avg_result
    
    print(f"  Final loss: {avg_result['final_loss']:.4f} ± {avg_result['loss_std']:.4f}")
    print(f"  Max grad: {avg_result['max_grad']:.4f}")
    print(f"  Outliers (gradient spikes): {avg_result['total_outliers']}")

# === Summary ===
print("\n" + "=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)
print(f"\n{'Config':<30} {'Final Loss':>12} {'Max Grad':>10} {'Outliers':>10}")
print("-" * 65)

for name, r in results.items():
    print(f"{name:<30} {r['final_loss']:>12.4f} {r['max_grad']:>10.4f} {r['total_outliers']:>10}")

# Determine best configuration
best = min(results.items(), key=lambda x: x[1]['final_loss'])
most_stable = min(results.items(), key=lambda x: x[1]['max_grad'])

print(f"\nBest loss: {best[0]}")
print(f"Most stable (lowest max grad): {most_stable[0]}")

# Check if paper's scheme (Both [-10,10] + Gate<=10) is best
paper_scheme = 'Both [-10,10] + Gate<=10'
if paper_scheme in results:
    paper_r = results[paper_scheme]
    print(f"\nPaper's scheme ({paper_scheme}):")
    print(f"  Loss: {paper_r['final_loss']:.4f}")
    print(f"  Stability: {paper_r['max_grad']:.4f}")
    
    if paper_r['max_grad'] <= results['Control (no clamp)']['max_grad']:
        print("  VERDICT: Paper's clamping IMPROVES stability vs control")
    else:
        print("  VERDICT: Paper's clamping does NOT improve stability in this experiment")

print("\n" + "=" * 65)
print("NOTE: This is a small-scale experiment (256-dim, 4 experts).")
print("Results at 1.6T scale may differ. The experiment validates the")
print("DIRECTION of clamping's effect, not exact magnitude.")
print("=" * 65)
