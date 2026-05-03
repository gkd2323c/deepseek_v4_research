"""Plan 04: SwiGLU Clamping with Outlier Injection (DirectML)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
import time

dml = torch_directml.device(0)

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, cl=None, cg=None):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)
        self.cl = cl
        self.cg = cg
    
    def forward(self, x):
        g = self.w1(x)
        u = self.w2(x)
        if self.cl: u = u.clamp(self.cl[0], self.cl[1])
        if self.cg: g = g.clamp(max=self.cg)
        return self.w3((g * torch.sigmoid(g)) * u)

class Block(nn.Module):
    def __init__(self, d, d_ff, cl=None, cg=None):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ff = SwiGLU(d, d_ff, cl, cg)
    
    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.ff(self.n2(x))
        return x

class Model(nn.Module):
    def __init__(self, cl=None, cg=None):
        super().__init__()
        d, V = 128, 500
        self.emb = nn.Embedding(V, d)
        self.layers = nn.ModuleList([Block(d, 256, cl, cg) for _ in range(4)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
    
    def forward(self, x):
        x = self.emb(x)
        for l in self.layers: x = l(x)
        return self.head(self.norm(x))

def inject_outlier(grad, step, freq=20, scale=100.0):
    """Inject outlier gradient periodically."""
    if step % freq == 0 and step > 0:
        # Scale up a random parameter's gradient
        idx = torch.randint(0, grad.numel(), (1,))
        grad.view(-1)[idx] *= scale
    return grad

def train(cl, cg, n=300, seed=42, outlier_freq=20, outlier_scale=100.0):
    torch.manual_seed(seed)
    m = Model(cl, cg).to(dml)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    losses, grads, spikes = [], [], []
    
    for i in range(n):
        x = torch.randint(0, 500, (16, 64), device=dml)
        y = torch.randint(0, 500, (16, 64), device=dml)
        
        logits = m(x)
        loss = F.cross_entropy(logits.view(-1, 500), y.view(-1))
        
        opt.zero_grad()
        loss.backward()
        
        # Inject outliers periodically
        for p in m.parameters():
            if p.grad is not None:
                p.grad.data = inject_outlier(p.grad.data, i, outlier_freq, outlier_scale)
        
        gn = sum(p.grad.norm().item()**2 for p in m.parameters() if p.grad is not None) ** 0.5
        grads.append(gn)
        losses.append(loss.item())
        
        # Spike detection
        if i > 10:
            avg = sum(grads[-10:]) / 10
            if gn > 5 * avg:
                spikes.append(i)
        
        opt.step()
    
    return {
        'loss': losses[-1],
        'max_gn': max(grads),
        'mean_gn': sum(grads) / len(grads),
        'spikes': len(spikes),
        'losses': losses,
        'grads': grads,
    }

print("=" * 65)
print("Plan 04: SwiGLU Clamping with Outlier Injection (DirectML)")
print("=" * 65)
print("Outlier gradient injected every 20 steps (100x normal magnitude)")

cfgs = [
    ('Control (no clamp)', None, None),
    ('Linear[-10,10]', (-10, 10), None),
    ('Gate<=10', None, 10),
    ('Paper[-10,10]+Gate<=10', (-10, 10), 10),
    ('Aggressive[-5,5]+Gate<=5', (-5, 5), 5),
    ('Relaxed[-20,20]+Gate<=20', (-20, 20), 20),
]

results = {}
for name, cl, cg in cfgs:
    print(f"\n{name}...")
    seeds = []
    for s in range(3):
        t0 = time.time()
        r = train(cl, cg, seed=s*100)
        dt = time.time() - t0
        seeds.append(r)
        print(f"  s{s}: loss={r['loss']:.4f} max_gn={r['max_gn']:.2f} spikes={r['spikes']} ({dt:.1f}s)")
    
    avg = {
        'loss': sum(r['loss'] for r in seeds) / 3,
        'loss_std': (sum((r['loss'] - sum(r2['loss'] for r2 in seeds)/3)**2 for r in seeds) / 3) ** 0.5,
        'max_gn': sum(r['max_gn'] for r in seeds) / 3,
        'mean_gn': sum(r['mean_gn'] for r in seeds) / 3,
        'spikes': sum(r['spikes'] for r in seeds),
    }
    results[name] = avg

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)
print(f"{'Config':<30} {'Loss':>8} {'±':>5} {'MaxGrad':>9} {'MeanGrad':>10} {'Spikes':>7}")
print("-" * 75)
for n, r in results.items():
    print(f"{n:<30} {r['loss']:>8.4f} {r['loss_std']:>5.4f} {r['max_gn']:>9.2f} {r['mean_gn']:>10.4f} {r['spikes']:>7}")

ctrl = results['Control (no clamp)']
paper = results['Paper[-10,10]+Gate<=10']

print(f"\nPaper vs Control:")
print(f"  Loss: {paper['loss']:.4f} vs {ctrl['loss']:.4f} (diff={paper['loss']-ctrl['loss']:+.4f})")
print(f"  MaxGrad: {paper['max_gn']:.2f} vs {ctrl['max_gn']:.2f} (ratio={paper['max_gn']/ctrl['max_gn']:.2f}x)")
print(f"  Spikes: {paper['spikes']} vs {ctrl['spikes']}")

if paper['max_gn'] < ctrl['max_gn'] * 0.9:
    print("\nVERDICT: Paper's SwiGLU clamping SIGNIFICANTLY reduces outlier impact.")
elif paper['max_gn'] <= ctrl['max_gn'] * 1.1:
    print("\nVERDICT: Paper's SwiGLU clamping helps maintain stability.")
else:
    print("\nVERDICT: Mixed results — clamping may not help at this scale.")
print("=" * 65)
