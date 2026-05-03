"""Sqrt(Softplus) vs Sigmoid: Numerical Analysis
V3 uses Sigmoid for MoE affinity scores, V4 changes to Sqrt(Softplus).
This script analyzes why the change is beneficial.
"""
import math, random
random.seed(42)

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def softplus(x):
    # log(1 + exp(x)), numerically stable
    if x > 20:
        return x
    elif x < -20:
        return math.exp(x)
    return math.log(1 + math.exp(x))

def softplus_grad(x):
    return sigmoid(x)  # d/dx softplus(x) = sigmoid(x)

def sqrt_softplus(x):
    return math.sqrt(softplus(x))

def sqrt_softplus_grad(x):
    sp = softplus(x)
    if sp < 1e-30:
        return 0.0
    return sigmoid(x) / (2.0 * math.sqrt(sp))

# === 1. Function Value Comparison ===
print("=" * 65)
print("1. Function Value Comparison")
print("=" * 65)
print(f"{'x':>8} | {'Sigmoid':>10} | {'Sqrt(SP)':>10} | {'Ratio':>10}")
print("-" * 50)
for x in [-10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10, 20]:
    sig = sigmoid(x)
    ssp = sqrt_softplus(x)
    ratio = ssp / max(sig, 1e-30)
    print(f"{x:>8.1f} | {sig:>10.6f} | {ssp:>10.6f} | {ratio:>10.4f}")

# === 2. Gradient Comparison ===
print("\n" + "=" * 65)
print("2. Gradient Comparison")
print("=" * 65)
print(f"{'x':>8} | {'Sig_grad':>12} | {'SSP_grad':>12} | {'Ratio':>10}")
print("-" * 50)
for x in [-10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10, 20]:
    sg = sigmoid_grad(x)
    ssg = sqrt_softplus_grad(x)
    ratio = ssg / max(sg, 1e-30)
    print(f"{x:>8.1f} | {sg:>12.8f} | {ssg:>12.8f} | {ratio:>10.4f}")

# === 3. Saturation Behavior ===
print("\n" + "=" * 65)
print("3. Saturation Behavior (gradient magnitude near extremes)")
print("=" * 65)
print("Sigmoid gradient max at x=0:", sigmoid_grad(0))
print("Sigmoid gradient at x=±5:", sigmoid_grad(5), sigmoid_grad(-5))
print("Sigmoid gradient at x=±10:", sigmoid_grad(10), sigmoid_grad(-10))
print()
print("Sqrt(Softplus) gradient at x=0:", sqrt_softplus_grad(0))
print("Sqrt(Softplus) gradient at x=5:", sqrt_softplus_grad(5))
print("Sqrt(Softplus) gradient at x=10:", sqrt_softplus_grad(10))
print("Sqrt(Softplus) gradient at x=-5:", sqrt_softplus_grad(-5))
print("Sqrt(Softplus) gradient at x=-10:", sqrt_softplus_grad(-10))

# === 4. Outlier Sensitivity ===
print("\n" + "=" * 65)
print("4. Outlier Sensitivity: Response to large input values")
print("=" * 65)
print("When MoE layers produce outlier activations (x >> 0):")
print()
for x in [10, 20, 50, 100, 500, 1000]:
    sig = sigmoid(x)
    ssp = sqrt_softplus(x)
    sg = sigmoid_grad(x)
    ssg = sqrt_softplus_grad(x)
    print(f"  x={x:>5}: Sigmoid={sig:.8f} (grad={sg:.2e})  "
          f"Sqrt(SP)={ssp:.4f} (grad={ssg:.6f})")

print()
print("Key observation:")
print("  Sigmoid: saturates at 1.0, gradient vanishes for large x")
print("  Sqrt(Softplus): grows as sqrt(x), gradient ~ 1/(2*sqrt(x)) — never vanishes!")

# === 5. Routing Distribution Analysis ===
print("\n" + "=" * 65)
print("5. Routing Distribution: Simulated MoE affinity scores")
print("=" * 65)

def softmax(values):
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    s = sum(exps)
    return [e / s for e in exps]

def entropy(probs):
    return -sum(p * math.log(max(p, 1e-30)) for p in probs)

# Simulate 8 experts, some with outlier affinities
scenarios = [
    ("Normal (no outliers)", [0.1, 0.2, 0.15, 0.3, 0.05, 0.1, 0.05, 0.05]),
    ("One strong outlier", [0.1, 0.2, 0.15, 5.0, 0.05, 0.1, 0.05, 0.05]),
    ("Two strong outliers", [5.0, 0.2, 0.15, 3.0, 0.05, 0.1, 0.05, 0.05]),
    ("Extreme outlier", [0.1, 0.2, 0.15, 50.0, 0.05, 0.1, 0.05, 0.05]),
]

for name, raw_scores in scenarios:
    sig_scores = [sigmoid(s) for s in raw_scores]
    ssp_scores = [sqrt_softplus(s) for s in raw_scores]
    
    sig_probs = softmax(sig_scores)
    ssp_probs = softmax(ssp_scores)
    
    sig_ent = entropy(sig_probs)
    ssp_ent = entropy(ssp_probs)
    
    print(f"\n  Scenario: {name}")
    print(f"    Raw scores:    {[f'{s:.2f}' for s in raw_scores]}")
    print(f"    Sigmoid probs: {[f'{p:.4f}' for p in sig_probs]}  (entropy={sig_ent:.4f})")
    print(f"    SqrtSP probs:  {[f'{p:.4f}' for p in ssp_probs]}  (entropy={ssp_ent:.4f})")

# === 6. Gradient Flow Through TopK Routing ===
print("\n" + "=" * 65)
print("6. Gradient Flow Through Routing: Impact on training stability")
print("=" * 65)
print()
print("When an expert gets outlier affinity (x=100):")
print(f"  Sigmoid:     value={sigmoid(100):.8f}, grad={sigmoid_grad(100):.2e}")
print(f"  Sqrt(Softplus): value={sqrt_softplus(100):.4f}, grad={sqrt_softplus_grad(100):.6f}")
print()
print("Sigmoid: gradient essentially zero → expert receives no learning signal")
print("Sqrt(Softplus): gradient ~0.05 → expert still receives meaningful gradient")
print()
print("This means Sqrt(Softplus) prevents 'dead experts' caused by outlier routing.")

# === 7. Behavior near zero ===
print("\n" + "=" * 65)
print("7. Behavior Near Zero (typical affinity range)")
print("=" * 65)
print(f"{'x':>8} | {'Sig':>10} | {'SSP':>10} | {'Sig/SSP':>10} | {'Sig_g':>10} | {'SSP_g':>10}")
print("-" * 65)
for x_10 in range(-20, 25, 5):
    x = x_10 / 10.0
    sig = sigmoid(x)
    ssp = sqrt_softplus(x)
    sg = sigmoid_grad(x)
    ssg = sqrt_softplus_grad(x)
    ratio = sig / max(ssp, 1e-30)
    print(f"{x:>8.1f} | {sig:>10.6f} | {ssp:>10.6f} | {ratio:>10.4f} | {sg:>10.6f} | {ssg:>10.6f}")

print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
print("""
Sqrt(Softplus) vs Sigmoid key differences:

1. GROWTH: Sigmoid saturates at 1.0; Sqrt(Softplus) grows as sqrt(x)
   → Sqrt(Softplus) preserves magnitude information for large affinities

2. GRADIENT: Sigmoid gradient vanishes exponentially; Sqrt(Softplus) ~ 1/(2*sqrt(x))
   → Sqrt(Softplus) prevents dead experts from outlier routing

3. ROUTING: Sigmoid compresses outliers into [0,1] losing discrimination;
   Sqrt(Softplus) preserves relative ordering with better dynamic range
   → More informative routing decisions under outlier conditions

4. STABILITY: Sqrt(Softplus) has bounded gradient near zero (like Sigmoid)
   but maintains non-vanishing gradient for large values
   → Better gradient flow through the routing mechanism

This explains V4's change: at 1.6T scale with 384 routed experts,
outlier affinities are more likely, and Sqrt(Softplus) handles them
gracefully while Sigmoid would create dead experts and routing collapse.
""")
