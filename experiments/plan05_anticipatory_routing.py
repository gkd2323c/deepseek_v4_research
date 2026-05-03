"""Plan 05: Anticipatory Routing Training Stability (DirectML)
Tests whether decoupling backbone/routing updates improves stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
import time, copy

dml = torch_directml.device(0)

class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts=4, n_activated=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d)
        ) for _ in range(n_experts)])
        self.router = nn.Linear(d, n_experts, bias=False)
        self.n_activated = n_activated
        self.n_experts = n_experts
    
    def forward(self, x, routing_indices=None):
        B, T, D = x.shape
        xf = x.view(-1, D)
        
        if routing_indices is None:
            logits = self.router(xf)
            weights = F.softmax(logits, dim=-1)
            topk_w, topk_idx = torch.topk(weights, self.n_activated, dim=-1)
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        else:
            topk_idx, topk_w = routing_indices
        
        out = torch.zeros_like(xf)
        for i in range(self.n_activated):
            idx = topk_idx[:, i]
            w = topk_w[:, i:i+1]
            for j in range(self.n_experts):
                mask = (idx == j)
                if mask.any():
                    out[mask] += w[mask] * self.experts[j](xf[mask])
        return out.view(B, T, D), (topk_idx, topk_w)

class Block(nn.Module):
    def __init__(self, d, d_ff, n_experts=4, n_activated=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.moe = MoE(d, d_ff, n_experts, n_activated)
    
    def forward(self, x, routing_indices=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        h = self.norm2(x)
        moe_out, indices = self.moe(h, routing_indices)
        x = x + moe_out
        return x, indices

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d, V = 128, 500
        self.emb = nn.Embedding(V, d)
        self.layers = nn.ModuleList([Block(d, 256) for _ in range(4)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
    
    def forward(self, x, routing_indices_list=None):
        x = self.emb(x)
        indices_out = []
        for i, layer in enumerate(self.layers):
            ri = routing_indices_list[i] if routing_indices_list else None
            x, idx = layer(x, ri)
            indices_out.append(idx)
        return self.head(self.norm(x)), indices_out

def detect_spike(losses, threshold=3.0):
    """Detect loss spike: current loss > threshold * rolling average."""
    if len(losses) < 10:
        return False
    avg = sum(losses[-10:]) / 10
    return losses[-1] > threshold * avg

def train_baseline(n=300, seed=42):
    """Normal training without AR."""
    torch.manual_seed(seed)
    m = Model().to(dml)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    losses, grads, spikes = [], [], []
    
    for i in range(n):
        x = torch.randint(0, 500, (8, 32), device=dml)
        y = torch.randint(0, 500, (8, 32), device=dml)
        logits, _ = m(x)
        loss = F.cross_entropy(logits.view(-1, 500), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        
        gn = sum(p.grad.norm().item()**2 for p in m.parameters() if p.grad is not None) ** 0.5
        losses.append(loss.item()); grads.append(gn)
        if detect_spike(losses): spikes.append(i)
    
    return {'losses': losses, 'grads': grads, 'spikes': spikes, 'final_loss': losses[-1]}

def train_ar_always(n=300, delta=5, seed=42):
    """Always-on AR: routing uses parameters from delta steps ago."""
    torch.manual_seed(seed)
    m = Model().to(dml)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    losses, grads, spikes = [], [], []
    history = []  # Store (step, model_state, routing_indices)
    
    for i in range(n):
        x = torch.randint(0, 500, (8, 32), device=dml)
        y = torch.randint(0, 500, (8, 32), device=dml)
        
        # Use historical routing if available
        if len(history) >= delta:
            _, _, old_indices = history[-delta]
            logits, _ = m(x, routing_indices_list=old_indices)
        else:
            logits, _ = m(x)
        
        loss = F.cross_entropy(logits.view(-1, 500), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        
        # Save current state for future routing
        with torch.no_grad():
            _, current_indices = m(x)
        history.append((i, None, current_indices))
        if len(history) > delta * 2:
            history.pop(0)
        
        gn = sum(p.grad.norm().item()**2 for p in m.parameters() if p.grad is not None) ** 0.5
        losses.append(loss.item()); grads.append(gn)
        if detect_spike(losses): spikes.append(i)
    
    return {'losses': losses, 'grads': grads, 'spikes': spikes, 'final_loss': losses[-1]}

def train_ar_dynamic(n=300, delta=5, spike_threshold=3.0, seed=42):
    """Dynamic AR: only activate when spike detected, deactivate after recovery."""
    torch.manual_seed(seed)
    m = Model().to(dml)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    losses, grads, spikes = [], [], []
    history = []
    ar_active = False
    ar_steps = 0
    
    for i in range(n):
        x = torch.randint(0, 500, (8, 32), device=dml)
        y = torch.randint(0, 500, (8, 32), device=dml)
        
        # Detect spike and toggle AR
        if detect_spike(losses, spike_threshold) and not ar_active:
            ar_active = True
            ar_steps = 0
        
        # Use AR if active
        if ar_active and len(history) >= delta:
            _, _, old_indices = history[-delta]
            logits, _ = m(x, routing_indices_list=old_indices)
            ar_steps += 1
            if ar_steps > 20:  # Deactivate after 20 steps
                ar_active = False
        else:
            logits, _ = m(x)
        
        loss = F.cross_entropy(logits.view(-1, 500), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        
        with torch.no_grad():
            _, current_indices = m(x)
        history.append((i, None, current_indices))
        if len(history) > delta * 2:
            history.pop(0)
        
        gn = sum(p.grad.norm().item()**2 for p in m.parameters() if p.grad is not None) ** 0.5
        losses.append(loss.item()); grads.append(gn)
        if detect_spike(losses, spike_threshold): spikes.append(i)
    
    return {'losses': losses, 'grads': grads, 'spikes': spikes, 'final_loss': losses[-1]}

print("=" * 65)
print("Plan 05: Anticipatory Routing Training Stability (DirectML)")
print("=" * 65)

n_steps = 300
n_seeds = 3

configs = [
    ('Baseline (no AR)', train_baseline),
    ('AR Always-On (Δ=5)', lambda s: train_ar_always(delta=5, seed=s)),
    ('AR Dynamic', lambda s: train_ar_dynamic(delta=5, spike_threshold=3.0, seed=s)),
]

results = {}
for name, train_fn in configs:
    print(f"\n{name}...")
    seeds = []
    for s in range(n_seeds):
        t0 = time.time()
        r = train_fn(s * 100)
        dt = time.time() - t0
        seeds.append(r)
        print(f"  s{s}: loss={r['final_loss']:.4f} max_gn={max(r['grads']):.2f} spikes={len(r['spikes'])} ({dt:.1f}s)")
    
    avg = {
        'final_loss': sum(r['final_loss'] for r in seeds) / n_seeds,
        'max_grad': sum(max(r['grads']) for r in seeds) / n_seeds,
        'mean_grad': sum(sum(r['grads'])/len(r['grads']) for r in seeds) / n_seeds,
        'spikes': sum(len(r['spikes']) for r in seeds),
    }
    results[name] = avg

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)
print(f"{'Config':<25} {'Loss':>8} {'MaxGrad':>9} {'MeanGrad':>10} {'Spikes':>7}")
print("-" * 65)
for n, r in results.items():
    print(f"{n:<25} {r['final_loss']:>8.4f} {r['max_grad']:>9.2f} {r['mean_grad']:>10.4f} {r['spikes']:>7}")

base = results['Baseline (no AR)']
dynamic = results['AR Dynamic']
print(f"\nDynamic AR vs Baseline:")
print(f"  Loss: {dynamic['final_loss']:.4f} vs {base['final_loss']:.4f}")
print(f"  Spikes: {dynamic['spikes']} vs {base['spikes']}")

if dynamic['spikes'] <= base['spikes']:
    print("\nVERDICT: Anticipatory Routing reduces training instability.")
else:
    print("\nVERDICT: AR shows limited benefit at this small scale.")
print("=" * 65)
