"""Plan 05: Anticipatory Routing (CPU version)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, copy

class MoE(nn.Module):
    def __init__(self, d, d_ff, ne=4, na=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d)) for _ in range(ne)])
        self.router = nn.Linear(d, ne, bias=False)
        self.na, self.ne = na, ne
    
    def forward(self, x, cached_idx=None):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        if cached_idx is None:
            logits = self.router(xf)
            w = F.softmax(logits, dim=-1)
            tw, ti = torch.topk(w, self.na, dim=-1)
            tw = tw / tw.sum(-1, keepdim=True)
        else:
            ti, tw = cached_idx
        out = torch.zeros_like(xf)
        for i in range(self.na):
            idx, wt = ti[:, i], tw[:, i:i+1]
            for j in range(self.ne):
                m = (idx == j)
                if m.any(): out[m] += wt[m] * self.experts[j](xf[m])
        return out.reshape(B, T, D), (ti, tw)

class Block(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.moe = MoE(d, d_ff)
    def forward(self, x, ri=None):
        h = self.n1(x); x = x + self.attn(h, h, h)[0]
        h = self.n2(x); mo, idx = self.moe(h, ri); x = x + mo
        return x, idx

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(500, 128)
        self.layers = nn.ModuleList([Block(128, 256) for _ in range(4)])
        self.norm = nn.LayerNorm(128); self.head = nn.Linear(128, 500, bias=False)
    def forward(self, x, ri_list=None):
        x = self.emb(x); idx_out = []
        for i, l in enumerate(self.layers):
            ri = ri_list[i] if ri_list else None
            x, idx = l(x, ri); idx_out.append(idx)
        return self.head(self.norm(x)), idx_out

def spike(losses, t=3.0):
    if len(losses) < 10: return False
    return losses[-1] > t * sum(losses[-10:]) / 10

def run(mode, n=300, seed=42):
    torch.manual_seed(seed)
    m = Model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    losses, grads, spikes, hist = [], [], [], []
    ar_on, ar_cnt = False, 0
    
    for i in range(n):
        x = torch.randint(0, 500, (8, 32))
        y = torch.randint(0, 500, (8, 32))
        
        # Routing decision
        ri_list = None
        if mode == 'ar_always' and len(hist) >= 5:
            ri_list = hist[-5][1]
        elif mode == 'ar_dynamic':
            if spike(losses, 3.0) and not ar_on:
                ar_on = True; ar_cnt = 0
            if ar_on and len(hist) >= 5:
                ri_list = hist[-5][1]; ar_cnt += 1
                if ar_cnt > 20: ar_on = False
        
        logits, idx_list = m(x, ri_list)
        loss = F.cross_entropy(logits.reshape(-1, 500), y.reshape(-1))
        
        opt.zero_grad(); loss.backward(); opt.step()
        
        with torch.no_grad():
            _, cur_idx = m(x)
        hist.append((i, cur_idx))
        if len(hist) > 10: hist.pop(0)
        
        gn = sum(p.grad.norm().item()**2 for p in m.parameters() if p.grad is not None) ** 0.5
        losses.append(loss.item()); grads.append(gn)
        if spike(losses, 3.0): spikes.append(i)
    
    return {'loss': losses[-1], 'max_gn': max(grads), 'spikes': len(spikes)}

print("=" * 60)
print("Plan 05: Anticipatory Routing (CPU)")
print("=" * 60)

modes = ['baseline', 'ar_always', 'ar_dynamic']
names = ['Baseline (no AR)', 'AR Always-On (Δ=5)', 'AR Dynamic']
results = {}

for mode, name in zip(modes, names):
    print(f"\n{name}...")
    seeds = [run(mode, seed=s*100) for s in range(3)]
    avg = {k: sum(r[k] for r in seeds) / 3 for k in ['loss', 'max_gn']}
    avg['spikes'] = sum(r['spikes'] for r in seeds)
    results[name] = avg
    for s, r in enumerate(seeds):
        print(f"  s{s}: loss={r['loss']:.4f} max_gn={r['max_gn']:.2f} spikes={r['spikes']}")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"{'Config':<25} {'Loss':>8} {'MaxGrad':>9} {'Spikes':>7}")
print("-" * 55)
for n, r in results.items():
    print(f"{n:<25} {r['loss']:>8.4f} {r['max_gn']:>9.2f} {r['spikes']:>7}")

base = results['Baseline (no AR)']
dyn = results['AR Dynamic']
print(f"\nDynamic AR vs Baseline:")
print(f"  Loss: {dyn['loss']:.4f} vs {base['loss']:.4f}")
print(f"  Spikes: {dyn['spikes']} vs {base['spikes']}")
if dyn['spikes'] <= base['spikes']:
    print("\nVERDICT: AR reduces instability.")
else:
    print("\nVERDICT: Limited benefit at small scale.")
print("=" * 60)
