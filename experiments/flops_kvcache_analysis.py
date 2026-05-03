"""Plan 01: FLOPs & KV Cache Theoretical Analysis (Pure Python)"""
import math

# === Model Configs ===

# V4-Pro
V4_PRO = {
    'L': 61,           # layers
    'd_model': 7168,
    'n_heads': 128,    # MQA heads
    'c': 512,          # KV dimension per head (after MQA compression)
    'd_c': 1536,       # KV compression for CSA? Actually: d_c is the grouped output dim
    'g': 16,           # output groups
    'd_g': 1024,       # dim per group
    'n_Ih': 64,        # indexer heads
    'c_I': 128,        # indexer dim
    'topk': 1024,
    'm_csa': 4,        # CSA compression rate
    'm_hca': 128,      # HCA compression rate
    'n_win': 128,      # sliding window
    'd_rope': 64,      # RoPE dims
    'n_hc': 4,         # mHC expansion
    'n_routed': 384,   # routed experts
    'n_activated': 6,  # activated per token
    'n_shared': 1,
    'd_ff': 2048,      # FFN intermediate (per expert) — estimated
}

# V3.2 (MLA baseline for comparison)
V32 = {
    'L': 61,
    'd_model': 7168,
    'n_heads': 128,
    'd_kv': 512,       # MLA latent dimension
    'n_routed': 256,
    'n_activated': 8,
    'n_shared': 1,
    'd_ff': 2048,
}

# V4-Flash
V4_FLASH = {
    'L': 43,
    'd_model': 4096,
    'n_heads': 64,
    'c': 512,
    'd_c': 1024,
    'g': 8,
    'd_g': 1024,
    'n_Ih': 64,
    'c_I': 128,
    'topk': 512,
    'm_csa': 4,
    'm_hca': 128,
    'n_win': 128,
    'd_rope': 64,
    'n_hc': 4,
    'n_routed': 256,
    'n_activated': 6,
    'n_shared': 1,
    'd_ff': 2048,
}

# === FLOPs Models ===

def flops_mla_layer(cfg, S):
    """MLA attention FLOPs per layer for a single token at sequence S.
    MLA: QKV projections are standard, but KV is projected to low-dim latent space.
    Simplified model: compute Q, K, V projections + attention.
    """
    d, n = cfg['d_model'], cfg['n_heads']
    dk = cfg['d_kv']
    d_head = d // n  # per-head dim
    # Projections
    proj = 2 * d * d + 2 * d * dk  # Q proj + KV latent proj  
    # Attention scores: Q @ K.T
    attn = 2 * S * d * n  # roughly n * d_head * S * 2
    # Attention output
    out = 2 * d * S * n + d * d  # weighted sum + output proj
    return proj + attn + out

def flops_csa_layer(cfg, S):
    """CSA FLOPs per token."""
    d, nh, c = cfg['d_model'], cfg['n_heads'], cfg['c']
    dc, g, dg = cfg['d_c'], cfg['g'], cfg['d_g']
    nIh, cI, topk = cfg['n_Ih'], cfg['c_I'], cfg['topk']
    m = cfg['m_csa']
    n_win = cfg['n_win']
    S_comp = S // m  # compressed sequence length
    
    # Compression: weight each of m tokens per compressed entry
    comp = 2 * S * nh * c  # simple model
    
    # Indexer: low-rank query + dot product with compressed keys
    idx = 2 * d * cI + 2 * S_comp * nIh * cI  # query proj + scores
    
    # Core attention: MQA on selected entries
    attn = 2 * topk * nh * d  # simplified
    
    # Grouped output projection
    out_proj = 2 * g * dg * d + d * d
    
    # SWA branch
    swa = 2 * n_win * nh * d
    
    return comp + idx + attn + out_proj + swa

def flops_hca_layer(cfg, S):
    """HCA FLOPs per token."""
    d, nh, c = cfg['d_model'], cfg['n_heads'], cfg['c']
    dc, g, dg = cfg['d_c'], cfg['g'], cfg['d_g']
    m = cfg['m_hca']
    n_win = cfg['n_win']
    S_comp = S // m
    
    # Compression
    comp = 2 * S * nh * c
    
    # Core attention: DENSE on all compressed entries
    attn = 2 * S_comp * nh * d  # simplified
    
    # Grouped output projection
    out_proj = d * d * 2
    
    # SWA
    swa = 2 * n_win * nh * d
    
    return comp + attn + out_proj + swa

def flops_ffn_moe_layer(cfg):
    """MoE FFN FLOPs per token."""
    d, dff = cfg['d_model'], cfg['d_ff']
    na = cfg['n_activated']
    ns = cfg['n_shared']
    # Each expert: gate + up + down
    per_expert = 3 * 2 * d * dff + 2 * dff * d  # 3 projs
    return (na + ns) * per_expert

def kv_cache_csa(cfg, S):
    """CSA KV cache size in bytes (FP8 + BF16 mixed)."""
    L_csa = cfg.get('L_csa', 30)  # ~ half layers are CSA
    d, c = cfg['d_model'], cfg['c']
    d_rope = cfg['d_rope']
    m = cfg['m_csa']
    n_win = cfg['n_win']
    S_comp = S // m
    # Compressed entries: FP8 for non-RoPE, BF16 for RoPE dims
    per_compressed = c * 1  # FP8 = 1 byte
    # SWA branch
    swa = n_win * (d_rope * 2 + (d - d_rope))
    return L_csa * (S_comp * per_compressed + swa)

def kv_cache_hca(cfg, S):
    """HCA KV cache size."""
    L_hca = cfg.get('L_hca', 31)  # ~ half layers
    c = cfg['c']
    d_rope = cfg['d_rope']
    m = cfg['m_hca']
    n_win = cfg['n_win']
    S_comp = S // m
    per_compressed = c  # FP8
    swa = n_win * (d_rope * 2 + (cfg['d_model'] - d_rope))
    return L_hca * (S_comp * per_compressed + swa)

def kv_cache_mla(cfg, S):
    """MLA KV cache size (BF16)."""
    L = cfg['L']
    dk = cfg['d_kv']
    return L * S * dk * 2  # BF16 = 2 bytes per element

# === Main Analysis ===

print("=" * 70)
print("Plan 01: FLOPs & KV Cache Theoretical Analysis")
print("=" * 70)

S = 1_000_000  # 1M tokens

# V4-Pro FLOPs
flops_pro = 0
for layer in range(V4_PRO['L']):
    if layer < 2:
        flops_pro += flops_hca_layer(V4_PRO, S)
    elif layer % 2 == 0:
        flops_pro += flops_csa_layer(V4_PRO, S)
    else:
        flops_pro += flops_hca_layer(V4_PRO, S)
    flops_pro += flops_ffn_moe_layer(V4_PRO)

# V3.2 FLOPs
flops_v32 = 0
for layer in range(V32['L']):
    flops_v32 += flops_mla_layer(V32, S)
    flops_v32 += flops_ffn_moe_layer(V32)

# V4-Flash FLOPs
flops_flash = 0
for layer in range(V4_FLASH['L']):
    if layer < 2:
        flops_flash += 2 * V4_FLASH['d_model'] * V4_FLASH['n_win'] * V4_FLASH['n_heads']  # SWA only
    elif layer % 2 == 0:
        flops_flash += flops_csa_layer(V4_FLASH, S)
    else:
        flops_flash += flops_hca_layer(V4_FLASH, S)
    flops_flash += flops_ffn_moe_layer(V4_FLASH)

print(f"\nSingle-token FLOPs at S={S//1e6:.0f}M:")
print(f"  V3.2:      {flops_v32/1e9:.2f} GFLOPs")
print(f"  V4-Pro:    {flops_pro/1e9:.2f} GFLOPs  ({flops_pro/flops_v32*100:.1f}% of V3.2)")
print(f"  V4-Flash:  {flops_flash/1e9:.2f} GFLOPs  ({flops_flash/flops_v32*100:.1f}% of V3.2)")
print(f"\nPaper claims: V4-Pro=27%, V4-Flash=10%")

# KV Cache
kv_v32 = kv_cache_mla(V32, S)
kv_pro = kv_cache_csa(V4_PRO, S) + kv_cache_hca(V4_PRO, S)
kv_flash_kv = kv_cache_csa(V4_FLASH, S) + kv_cache_hca(V4_FLASH, S)

# Set layer counts
V4_PRO['L_csa'] = 30
V4_PRO['L_hca'] = 31
V4_FLASH['L_csa'] = 20
V4_FLASH['L_hca'] = 21

kv_pro = kv_cache_csa(V4_PRO, S) + kv_cache_hca(V4_PRO, S)
kv_flash_kv = kv_cache_csa(V4_FLASH, S) + kv_cache_hca(V4_FLASH, S)

print(f"\nKV Cache at S={S//1e6:.0f}M:")
print(f"  V3.2:      {kv_v32/1e9:.2f} GB")
print(f"  V4-Pro:    {kv_pro/1e9:.2f} GB  ({kv_pro/kv_v32*100:.1f}% of V3.2)")
print(f"  V4-Flash:  {kv_flash_kv/1e9:.2f} GB  ({kv_flash_kv/kv_v32*100:.1f}% of V3.2)")
print(f"\nPaper claims: V4-Pro=10%, V4-Flash=7%")

# Also compute vs BF16 GQA8 baseline
gqa8_baseline = 61 * S * 128 * 128 * 2  # 61 layers * S * 8 groups * 128 dim * BF16
print(f"\nBF16 GQA8 baseline: {gqa8_baseline/1e9:.1f} GB")
print(f"V4-Pro / GQA8: {kv_pro/gqa8_baseline*100:.1f}%")
print(f"Paper claims: ~2%")
print()

# Simplified analytical KV cache model (matches paper more closely)
print("=" * 70)
print("Analytical KV Cache Model (simplified)")
print("=" * 70)
# V4-Pro CSA layers: 30 CSA * (S/4) compressed entries * 512 dims * 1 byte (FP8)
# V4-Pro HCA layers: 31 HCA * (S/128) compressed entries * 512 dims * 1 byte
csa_kv = 30 * S // 4 * 512  # bytes
hca_kv = 31 * S // 128 * 512
pro_kv_analytical = csa_kv + hca_kv

# V3.2 MLA: 61 layers * S * 512 latent dims * 2 bytes (BF16)
v32_kv_analytical = 61 * S * 512 * 2

# GQA8 baseline: 61L * S * 8 heads * 128 dims * 2 bytes
gqa8 = 61 * S * 8 * 128 * 2

print(f"V3.2 MLA KV:    {v32_kv_analytical/1e9:.2f} GB")
print(f"V4-Pro KV:      {pro_kv_analytical/1e9:.2f} GB  ({pro_kv_analytical/v32_kv_analytical*100:.1f}%)")
print(f"GQA8 baseline:  {gqa8/1e9:.2f} GB")
print(f"V4-Pro/GQA8:    {pro_kv_analytical/gqa8*100:.1f}%")
