"""Plan 04: SwiGLU Clamping (CPU version, faster)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, random, math

torch.manual_seed(42)
random.seed(42)

class Config:
    vocab_size = 500
    d_model = 128
    n_heads = 4
    n_layers = 4
    d_ff = 256
    n_experts = 4
    n_activated = 2
    batch_size = 16
    seq_len = 64

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, clamp_linear=None, clamp_gate=None):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.clamp_linear = clamp_linear
        self.clamp_gate = clamp_gate
    
    def forward(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)
        if self.clamp_linear is not None:
            up = up.clamp(min=self.clamp_linear[0], max=self.clamp_linear[1])
        if self.clamp_gate is not None:
            gate = gate.clamp(max=self.clamp_gate)
        return self.w_down((gate * torch.sigmoid(gate)) * up)

class MoELayer(nn.Module):
    def __init__(self, cfg, clamp_linear=None, clamp_gate=None):
        super().__init__()
        self.experts = nn.ModuleList([
            SwiGLU(cfg.d_model, cfg.d_ff, clamp_linear, clamp_gate)
            for _ in range(cfg.n_experts)
        ])
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.n_activated = cfg.n_activated
    
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        logits = self.router(x_flat)
        weights = F.softmax(logits, dim=-1)
        topk_w, topk_idx = torch.topk(weights, self.n_activated, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x_flat)
        for i in range(self.n_activated):
            idx = topk_idx[:, i]
            w = topk_w[:, i:i+1]
            for j in range(len(self.experts)):
                mask = (idx == j)
                if mask.any():
                    out = self.experts[j](x_flat[mask])
                    output[mask] += w[mask] * out
        return output.view(B, T, D)

class TransformerBlock(nn.Module):
    def __init__(self, cfg, use_moe=False, clamp_linear=None, clamp_gate=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.ffn = MoELayer(cfg, clamp_linear, clamp_gate) if use_moe else \
                   SwiGLU(cfg.d_model, cfg.d_ff, clamp_linear, clamp_gate)
    
    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        n = self.norm2(x)
        x = x + self.ffn(n)
        return x

class Model(nn.Module):
    def __init__(self, cfg, clamp_linear=None, clamp_gate=None):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg, use_moe=(i >= 2), clamp_linear=clamp_linear, clamp_gate=clamp_gate)
            for i in range(cfg.n_layers)
        ])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    
    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

def train(cfg, clamp_linear, clamp_gate, n_steps=300, lr=3e-4, seed=42):
    torch.manual_seed(seed)
    model = Model(cfg, clamp_linear, clamp_gate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    losses, grad_norms, spike_steps = [], [], []
    
    for step in range(n_steps):
        x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        
        opt.zero_grad()
        loss.backward()
        
        gn = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
        grad_norms.append(gn)
        losses.append(loss.item())
        
        # Spike detection: grad_norm > 5x rolling average
        if step > 10:
            avg = sum(grad_norms[-10:]) / 10
            if gn > 5 * avg:
                spike_steps.append(step)
        
        opt.step()
    
    return {
        'losses': losses,
        'grad_norms': grad_norms,
        'spikes': spike_steps,
        'final_loss': losses[-1],
        'max_grad': max(grad_norms),
        'mean_grad': sum(grad_norms) / len(grad_norms),
    }

print("=" * 65)
print("Plan 04: SwiGLU Clamping Training Stability")
print("=" * 65)

cfg = Config()
n_steps = 300
n_seeds = 3

configs = {
    'Control (no clamp)': (None, None),
    'Linear [-10,10]': ((-10, 10), None),
    'Gate <= 10': (None, 10),
    'Paper: [-10,10] + Gate<=10': ((-10, 10), 10),
    'Aggressive: [-5,5] + Gate<=5': ((-5, 5), 5),
    'Relaxed: [-20,20] + Gate<=20': ((-20, 20), 20),
}

results = {}
for name, (cl, cg) in configs.items():
    print(f"\n{name}...")
    seeds = []
    for s in range(n_seeds):
        r = train(cfg, cl, cg, n_steps=n_steps, seed=s*100)
        seeds.append(r)
        print(f"  seed {s}: loss={r['final_loss']:.4f} max_grad={r['max_grad']:.2f} spikes={len(r['spikes'])}")
    
    avg = {
        'final_loss': sum(r['final_loss'] for r in seeds) / n_seeds,
        'max_grad': sum(r['max_grad'] for r in seeds) / n_seeds,
        'mean_grad': sum(r['mean_grad'] for r in seeds) / n_seeds,
        'total_spikes': sum(len(r['spikes']) for r in seeds),
        'loss_std': (sum((r['final_loss'] - sum(r2['final_loss'] for r2 in seeds)/n_seeds)**2 for r in seeds) / n_seeds) ** 0.5,
    }
    results[name] = avg

# Summary
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)
print(f"{'Config':<32} {'Loss':>8} {'±':>5} {'MaxGrad':>9} {'Spikes':>7}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<32} {r['final_loss']:>8.4f} {r['loss_std']:>5.4f} {r['max_grad']:>9.2f} {r['total_spikes']:>7}")

ctrl = results['Control (no clamp)']
paper = results['Paper: [-10,10] + Gate<=10']
print(f"\nPaper scheme vs Control:")
print(f"  Loss: {paper['final_loss']:.4f} vs {ctrl['final_loss']:.4f} (diff={paper['final_loss']-ctrl['final_loss']:+.4f})")
print(f"  MaxGrad: {paper['max_grad']:.2f} vs {ctrl['max_grad']:.2f} (ratio={paper['max_grad']/ctrl['max_grad']:.2f}x)")
print(f"  Spikes: {paper['total_spikes']} vs {ctrl['total_spikes']}")

if paper['max_grad'] <= ctrl['max_grad'] * 1.1:
    print("\nVERDICT: Paper's SwiGLU clamping DOES improve or maintain stability.")
else:
    print("\nVERDICT: Paper's SwiGLU clamping shows mixed results at this scale.")
print("=" * 65)
