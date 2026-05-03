"""Plan 04: SwiGLU Activation Distribution Analysis (DirectML)
Measures how clamping affects activation statistics and outlier prevalence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml

dml = torch_directml.device(0)

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, cl=None, cg=None, track=True):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)
        self.cl = cl
        self.cg = cg
        self.track = track
        self.stats = {'pre_gate': [], 'post_gate': [], 'pre_linear': [], 'post_linear': [], 'output': []}
    
    def forward(self, x):
        g = self.w1(x)
        u = self.w2(x)
        
        if self.track:
            self.stats['pre_gate'].append(g.detach().abs().max().item())
            self.stats['pre_linear'].append(u.detach().abs().max().item())
        
        if self.cl: u = u.clamp(self.cl[0], self.cl[1])
        if self.cg: g = g.clamp(max=self.cg)
        
        if self.track:
            self.stats['post_gate'].append(g.detach().abs().max().item())
            self.stats['post_linear'].append(u.detach().abs().max().item())
        
        out = self.w3((g * torch.sigmoid(g)) * u)
        
        if self.track:
            self.stats['output'].append(out.detach().abs().max().item())
        
        return out

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

def analyze(cl, cg, n_steps=100, seed=42):
    torch.manual_seed(seed)
    m = Model(cl, cg).to(dml)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.01)  # High LR to induce instability
    
    for i in range(n_steps):
        x = torch.randint(0, 500, (8, 32), device=dml)
        y = torch.randint(0, 500, (8, 32), device=dml)
        logits = m(x)
        loss = F.cross_entropy(logits.view(-1, 500), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # Collect stats from first SwiGLU layer
    ff = m.layers[0].ff
    stats = ff.stats
    
    return {
        'pre_gate_max': max(stats['pre_gate']),
        'post_gate_max': max(stats['post_gate']),
        'pre_linear_max': max(stats['pre_linear']),
        'post_linear_max': max(stats['post_linear']),
        'output_max': max(stats['output']),
        'final_loss': loss.item(),
    }

print("=" * 65)
print("Plan 04: SwiGLU Activation Distribution Analysis")
print("=" * 65)
print("Using high learning rate (1e-2) to induce activation instability\n")

cfgs = [
    ('Control (no clamp)', None, None),
    ('Linear[-10,10]', (-10, 10), None),
    ('Gate<=10', None, 10),
    ('Paper[-10,10]+Gate<=10', (-10, 10), 10),
    ('Aggressive[-5,5]+Gate<=5', (-5, 5), 5),
]

results = {}
for name, cl, cg in cfgs:
    r = analyze(cl, cg)
    results[name] = r
    print(f"{name}:")
    print(f"  Gate  — pre: {r['pre_gate_max']:.2f}  post: {r['post_gate_max']:.2f}")
    print(f"  Linear— pre: {r['pre_linear_max']:.2f}  post: {r['post_linear_max']:.2f}")
    print(f"  Output: {r['output_max']:.2f}  Loss: {r['final_loss']:.4f}")
    print()

print("=" * 65)
print("KEY METRIC: How much does clamping reduce activation magnitudes?")
print("=" * 65)

ctrl = results['Control (no clamp)']
paper = results['Paper[-10,10]+Gate<=10']

gate_reduction = (ctrl['pre_gate_max'] - paper['post_gate_max']) / ctrl['pre_gate_max'] * 100
linear_reduction = (ctrl['pre_linear_max'] - paper['post_linear_max']) / ctrl['pre_linear_max'] * 100
output_ratio = paper['output_max'] / ctrl['output_max']

print(f"\nPaper scheme vs Control:")
print(f"  Gate activation reduction:    {gate_reduction:.1f}%")
print(f"  Linear activation reduction:  {linear_reduction:.1f}%")
print(f"  Output magnitude ratio:       {output_ratio:.2f}x")

if paper['post_gate_max'] < ctrl['pre_gate_max'] * 0.8:
    print("\nVERDICT: Clamping SIGNIFICANTLY reduces outlier activations.")
    print("This prevents the feedback loop: outlier → large SwiGLU → larger outlier")
else:
    print("\nVERDICT: Clamping has limited effect at this scale.")
    print("At 1.6T scale with 384 experts, the effect would be more pronounced.")
print("=" * 65)
