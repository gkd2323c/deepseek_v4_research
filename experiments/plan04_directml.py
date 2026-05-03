"""Plan 04: SwiGLU Clamping (DirectML compatible)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
import time, math

dml = torch_directml.device(0)
print(f"Device: {dml}")

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, cl=None, cg=None):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)
        self.cl = cl  # (min, max) for linear
        self.cg = cg  # max for gate
    
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

def train(cl, cg, n=200, seed=42):
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
        
        gn = sum(p.grad.norm().item()**2 for p in m.parameters() if p.grad is not None) ** 0.5
        grads.append(gn)
        losses.append(loss.item())
        
        if i > 10 and gn > 5 * sum(grads[-10:]) / 10:
            spikes.append(i)
        
        opt.step()
    
    return {'loss': losses[-1], 'max_gn': max(grads), 'spikes': len(spikes), 'losses': losses}

print("=" * 60)
print("Plan 04: SwiGLU Clamping (DirectML)")
print("=" * 60)

cfgs = [
    ('Control', None, None),
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
        'max_gn': sum(r['max_gn'] for r in seeds) / 3,
        'spikes': sum(r['spikes'] for r in seeds),
    }
    results[name] = avg

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"{'Config':<30} {'Loss':>8} {'MaxGrad':>9} {'Spikes':>7}")
print("-" * 60)
for n, r in results.items():
    print(f"{n:<30} {r['loss']:>8.4f} {r['max_gn']:>9.2f} {r['spikes']:>7}")

ctrl = results['Control']
paper = results['Paper[-10,10]+Gate<=10']
print(f"\nPaper vs Control:")
print(f"  Loss: {paper['loss']:.4f} vs {ctrl['loss']:.4f}")
print(f"  MaxGrad: {paper['max_gn']:.2f} vs {ctrl['max_gn']:.2f} (ratio={paper['max_gn']/ctrl['max_gn']:.2f}x)")
print(f"  Spikes: {paper['spikes']} vs {ctrl['spikes']}")

if paper['max_gn'] <= ctrl['max_gn'] * 1.2:
    print("\nVERDICT: Paper's SwiGLU clamping improves/stabilizes training.")
else:
    print("\nVERDICT: Mixed results at this scale.")
print("=" * 60)
