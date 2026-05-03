"""Atom Graph Visualization for DeepSeek-V4 Research"""
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# Atom metadata (from the project)
atoms = {
    # V4 internal atoms
    'dbfe68f2': {'name': 'Attention Bottleneck', 'type': 'fact'},
    '3e5cd741': {'name': 'V4 Model Specs', 'type': 'fact'},
    '266f1dd4': {'name': 'MoE Stability Challenges', 'type': 'fact'},
    '3cae5a37': {'name': 'mHC', 'type': 'method'},
    'c683d7d4': {'name': 'CSA', 'type': 'method'},
    'f3570445': {'name': 'HCA', 'type': 'method'},
    '5c9eee15': {'name': 'Hybrid Attention', 'type': 'method'},
    '738e20ee': {'name': 'V4 Muon', 'type': 'method'},
    'b365a813': {'name': 'Sqrt(Softplus)', 'type': 'method'},
    '6482f088': {'name': 'Hash Routing', 'type': 'method'},
    '0d94ff02': {'name': 'Attention Sink', 'type': 'method'},
    '9df4780c': {'name': 'Partial RoPE', 'type': 'method'},
    'b390dc6d': {'name': 'KV Cache Efficiency', 'type': 'verification'},
    'd36151d8': {'name': 'KV Cache vs GQA8', 'type': 'verification'},
    'aaf732a0': {'name': 'FP4 QAT', 'type': 'method'},
    'dcffaf87': {'name': 'Deterministic Kernels', 'type': 'method'},
    'c9184e5d': {'name': 'Anticipatory Routing', 'type': 'method'},
    '9286d65b': {'name': 'SwiGLU Clamping', 'type': 'method'},
    '3dd80c1c': {'name': 'Pre-Training Setup', 'type': 'method'},
    '45fc2266': {'name': 'Base Model Eval', 'type': 'verification'},
    'd042bc26': {'name': 'OPD Post-Training', 'type': 'method'},
    '4af43fd0': {'name': 'Three Reasoning Modes', 'type': 'method'},
    '028e1dde': {'name': 'V4-Pro-Max Benchmarks', 'type': 'verification'},
    'e34c1360': {'name': 'Real-World Tasks', 'type': 'verification'},
    '0f41420e': {'name': 'mHC Spectral Norm', 'type': 'theorem'},
    'b3c31cb6': {'name': 'GRM', 'type': 'method'},
    '77c6f180': {'name': 'Het KV Cache', 'type': 'method'},
    '68eae2f7': {'name': 'Contextual Parallelism', 'type': 'method'},
    'd0199990': {'name': 'Fine-Grained EP', 'type': 'method'},
    
    # Cross-reference atoms
    '53d90499': {'name': 'V3 (MLA+MoE)', 'type': 'fact'},
    '25ab4530': {'name': 'MLA (from V3)', 'type': 'method'},
    'dc3a6882': {'name': 'Hash Layers', 'type': 'method'},
    '674afaa0': {'name': 'Attention Sink (Xiao)', 'type': 'fact'},
    '25e3c705': {'name': 'V2 (MLA+MoE+GRPO)', 'type': 'fact'},
    '4816855c': {'name': 'R1 (Pure RL)', 'type': 'fact'},
    'c7a11c5d': {'name': 'HC (Zhu 2024)', 'type': 'method'},
    '0cc81172': {'name': 'Muon (Liu 2025)', 'type': 'method'},
    '43cbe3f9': {'name': 'HC Matrix', 'type': 'method'},
    '0481dcf3': {'name': 'Pre/Post-Norm as HC', 'type': 'fact'},
    'b9b1aee4': {'name': 'Dynamic HC', 'type': 'method'},
    '5bf0e9b5': {'name': 'HC n=4 Optimal', 'type': 'verification'},
    'c537c1cc': {'name': 'HC Eliminates Spikes', 'type': 'verification'},
    '8f135ace': {'name': 'HC Seq-Parallel', 'type': 'method'},
    'd4cd913e': {'name': 'HC Overhead', 'type': 'verification'},
    '87b47016': {'name': 'Muon NS Mechanism', 'type': 'method'},
    '60f9cf57': {'name': 'Muon Scaling', 'type': 'method'},
    '4f26dab8': {'name': 'Muon 2x Efficiency', 'type': 'verification'},
    '39919aa0': {'name': 'Muon SVD Entropy', 'type': 'verification'},
    'a9dbd915': {'name': 'Distributed Muon', 'type': 'method'},
    '2dce353c': {'name': 'V3 MLA', 'type': 'method'},
    'fe7fa8a9': {'name': 'V3 Aux-Loss-Free', 'type': 'method'},
    'a7b37ec2': {'name': 'V3 MTP', 'type': 'method'},
    'da8a77d8': {'name': 'V3 FP8 Training', 'type': 'method'},
    '5bc17bdb': {'name': 'V3 Zero Spike', 'type': 'verification'},
    '2a95405e': {'name': 'V3 GRPO', 'type': 'method'},
    '38045aa1': {'name': 'V3 R1 Distillation', 'type': 'method'},
    'd7de59b2': {'name': 'V2 MLA', 'type': 'method'},
    '2b4a0e72': {'name': 'V2 DeepSeekMoE', 'type': 'method'},
    '411ef217': {'name': 'V2 Training Efficiency', 'type': 'verification'},
    'a268014f': {'name': 'V2 GRPO', 'type': 'method'},
    '4cc524a8': {'name': 'V2 Long Context', 'type': 'method'},
    '863cd230': {'name': 'R1-Zero Pure RL', 'type': 'verification'},
    '2b3c7f00': {'name': 'R1 Multi-Stage', 'type': 'method'},
    '39f87ae3': {'name': 'R1 Failed Approaches', 'type': 'fact'},
    'af735eb4': {'name': 'R1 Distillation', 'type': 'verification'},
    'd334d0a0': {'name': 'R1 Adaptive CoT', 'type': 'verification'},
}

# Relations (source, target, type)
relations = [
    # V4 internal
    ('dbfe68f2', 'c683d7d4', 'motivates'),
    ('dbfe68f2', 'f3570445', 'motivates'),
    ('dbfe68f2', '5c9eee15', 'motivates'),
    ('3cae5a37', '3e5cd741', 'derives'),
    ('3cae5a37', '0f41420e', 'formalizes'),
    ('5c9eee15', 'b390dc6d', 'validates'),
    ('5c9eee15', 'd36151d8', 'validates'),
    ('5c9eee15', '77c6f180', 'derives'),
    ('c683d7d4', '0d94ff02', 'derives'),
    ('c683d7d4', '9df4780c', 'derives'),
    ('c683d7d4', '68eae2f7', 'derives'),
    ('f3570445', '0d94ff02', 'derives'),
    ('f3570445', '9df4780c', 'derives'),
    ('266f1dd4', 'c9184e5d', 'motivates'),
    ('266f1dd4', '9286d65b', 'motivates'),
    ('266f1dd4', 'dcffaf87', 'motivates'),
    ('3dd80c1c', '45fc2266', 'validates'),
    ('3dd80c1c', 'd042bc26', 'validates'),
    ('d042bc26', '028e1dde', 'validates'),
    ('d042bc26', '4af43fd0', 'derives'),
    ('d042bc26', 'e34c1360', 'validates'),
    ('d042bc26', 'b3c31cb6', 'derives'),
    ('4af43fd0', '028e1dde', 'validates'),
    ('4af43fd0', 'e34c1360', 'validates'),
    ('0f41420e', '3cae5a37', 'validates'),
    ('77c6f180', 'b390dc6d', 'validates'),
    ('68eae2f7', 'b390dc6d', 'validates'),
    ('b3c31cb6', '028e1dde', 'validates'),
    ('3cae5a37', '45fc2266', 'validates'),
    ('c9184e5d', '45fc2266', 'validates'),
    ('9286d65b', '45fc2266', 'validates'),
    ('aaf732a0', 'b390dc6d', 'validates'),
    ('d0199990', 'b390dc6d', 'validates'),
    ('dcffaf87', '45fc2266', 'validates'),
    
    # Cross-reference
    ('53d90499', '3e5cd741', 'motivates'),
    ('53d90499', 'b365a813', 'motivates'),
    ('53d90499', '266f1dd4', 'contradicts'),
    ('53d90499', '2dce353c', 'derives'),
    ('53d90499', 'fe7fa8a9', 'derives'),
    ('53d90499', 'a7b37ec2', 'derives'),
    ('53d90499', 'da8a77d8', 'derives'),
    ('53d90499', '5bc17bdb', 'derives'),
    ('53d90499', '2a95405e', 'derives'),
    ('53d90499', '38045aa1', 'derives'),
    ('25ab4530', 'c683d7d4', 'motivates'),
    ('25ab4530', 'f3570445', 'motivates'),
    ('25ab4530', '5c9eee15', 'motivates'),
    ('dc3a6882', '6482f088', 'derives'),
    ('674afaa0', '0d94ff02', 'derives'),
    ('25e3c705', '53d90499', 'motivates'),
    ('25e3c705', '25ab4530', 'motivates'),
    ('25e3c705', 'd7de59b2', 'derives'),
    ('25e3c705', '2b4a0e72', 'derives'),
    ('25e3c705', '411ef217', 'derives'),
    ('25e3c705', 'a268014f', 'derives'),
    ('25e3c705', '4cc524a8', 'derives'),
    ('4816855c', 'd042bc26', 'motivates'),
    ('4816855c', '4af43fd0', 'motivates'),
    ('4816855c', '863cd230', 'derives'),
    ('4816855c', '2b3c7f00', 'derives'),
    ('c7a11c5d', '3cae5a37', 'motivates'),
    ('c7a11c5d', '43cbe3f9', 'derives'),
    ('0cc81172', '738e20ee', 'derives'),
    ('0cc81172', '87b47016', 'derives'),
    ('43cbe3f9', '3cae5a37', 'derives'),
    ('43cbe3f9', 'b9b1aee4', 'derives'),
    ('43cbe3f9', '0481dcf3', 'formalizes'),
    ('43cbe3f9', '8f135ace', 'derives'),
    ('43cbe3f9', 'd4cd913e', 'validates'),
    ('b9b1aee4', '5bf0e9b5', 'validates'),
    ('c537c1cc', '0f41420e', 'motivates'),
    ('87b47016', '738e20ee', 'derives'),
    ('87b47016', '60f9cf57', 'derives'),
    ('87b47016', '4f26dab8', 'validates'),
    ('87b47016', '39919aa0', 'validates'),
    ('87b47016', 'a9dbd915', 'derives'),
    ('60f9cf57', '738e20ee', 'derives'),
    ('3e5cd741', '738e20ee', 'derives'),
    ('3e5cd741', 'b365a813', 'derives'),
    ('3e5cd741', '6482f088', 'derives'),
    ('b365a813', '45fc2266', 'validates'),
    ('6482f088', '45fc2266', 'validates'),
    ('2dce353c', 'c683d7d4', 'motivates'),
    ('2dce353c', 'f3570445', 'motivates'),
    ('fe7fa8a9', 'b365a813', 'derives'),
    ('a7b37ec2', '3dd80c1c', 'derives'),
    ('da8a77d8', 'aaf732a0', 'motivates'),
    ('5bc17bdb', '266f1dd4', 'contradicts'),
    ('2a95405e', 'd042bc26', 'derives'),
    ('38045aa1', 'd042bc26', 'derives'),
    ('d7de59b2', '2dce353c', 'derives'),
    ('2b4a0e72', 'fe7fa8a9', 'derives'),
    ('a268014f', '2a95405e', 'derives'),
    ('863cd230', '4af43fd0', 'motivates'),
    ('d334d0a0', '4af43fd0', 'motivates'),
    ('2b3c7f00', 'd042bc26', 'derives'),
    ('39f87ae3', 'd042bc26', 'motivates'),
    ('af735eb4', 'd042bc26', 'derives'),
]

# Color scheme for atom types
type_colors = {
    'fact': '#FF6B6B',      # Red
    'method': '#4ECDC4',     # Teal
    'theorem': '#FFE66D',    # Yellow
    'verification': '#95E1D3', # Light green
}

# Create graph
G = nx.DiGraph()

# Add nodes
for atom_id, info in atoms.items():
    G.add_node(atom_id, **info)

# Add edges
for src, tgt, rel_type in relations:
    if src in atoms and tgt in atoms:
        G.add_edge(src, tgt, relation=rel_type)

# --- Figure 1: Core Architecture Evolution ---
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Select core atoms for the main figure
core_atoms = [
    '25e3c705', '53d90499', '4816855c',  # V2, V3, R1
    '2dce353c', 'fe7fa8a9', 'a7b37ec2', 'da8a77d8', '5bc17bdb', '2a95405e', '38045aa1',  # V3 details
    'd7de59b2', '2b4a0e72',  # V2 details
    '863cd230', '2b3c7f00', 'af735eb4',  # R1 details
    'c683d7d4', 'f3570445', '5c9eee15',  # CSA, HCA, Hybrid
    '3cae5a37', '738e20ee', 'b365a813',  # mHC, Muon, Sqrt(SP)
    'c9184e5d', '9286d65b',  # AR, Clamping
    'd042bc26', '4af43fd0',  # OPD, Reasoning
    'b390dc6d', '45fc2266',  # Verified
]

core_relations = [(s, t, r) for s, t, r in relations if s in core_atoms and t in core_atoms]

# Create subgraph
H = G.subgraph(core_atoms).copy()

# Layout
pos = nx.spring_layout(H, k=2, iterations=50, seed=42)

# Draw nodes
for atom_type in ['fact', 'method', 'theorem', 'verification']:
    nodes = [n for n in H.nodes() if H.nodes[n].get('type') == atom_type]
    if nodes:
        nx.draw_networkx_nodes(H, pos, nodelist=nodes, 
                              node_color=type_colors[atom_type],
                              node_size=800, alpha=0.9, ax=ax)

# Draw edges with different styles for different relation types
edge_styles = {
    'motivates': ('#FF6B6B', 'dashed'),
    'derives': ('#4ECDC4', 'solid'),
    'validates': ('#95E1D3', 'dotted'),
    'formalizes': ('#FFE66D', 'solid'),
    'contradicts': ('#FF0000', 'solid'),
}

for src, tgt, data in H.edges(data=True):
    rel = data.get('relation', 'other')
    color, style = edge_styles.get(rel, ('gray', 'solid'))
    nx.draw_networkx_edges(H, pos, edgelist=[(src, tgt)], 
                          edge_color=color, style=style,
                          arrows=True, arrowsize=15, ax=ax,
                          connectionstyle="arc3,rad=0.1")

# Draw labels
labels = {n: H.nodes[n].get('name', n[:8]) for n in H.nodes()}
nx.draw_networkx_labels(H, pos, labels, font_size=7, font_weight='bold', ax=ax)

# Legend
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor=type_colors['fact'], label='Fact'),
    Patch(facecolor=type_colors['method'], label='Method'),
    Patch(facecolor=type_colors['theorem'], label='Theorem'),
    Patch(facecolor=type_colors['verification'], label='Verification'),
    Line2D([0], [0], color='#FF6B6B', linestyle='dashed', label='motivates'),
    Line2D([0], [0], color='#4ECDC4', linestyle='solid', label='derives'),
    Line2D([0], [0], color='#95E1D3', linestyle='dotted', label='validates'),
    Line2D([0], [0], color='#FF0000', linestyle='solid', label='contradicts'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

ax.set_title('DeepSeek-V4 Atom Graph: Core Architecture Evolution', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/gkd2323c/deepseek_v4/research_literature/atom_graph_core.png', dpi=150, bbox_inches='tight')
print("Saved: atom_graph_core.png")

# --- Figure 2: Full graph (smaller, overview) ---
fig2, ax2 = plt.subplots(1, 1, figsize=(20, 16))

pos2 = nx.spring_layout(G, k=1.5, iterations=30, seed=42)

for atom_type in ['fact', 'method', 'theorem', 'verification']:
    nodes = [n for n in G.nodes() if G.nodes[n].get('type') == atom_type]
    if nodes:
        nx.draw_networkx_nodes(G, pos2, nodelist=nodes,
                              node_color=type_colors[atom_type],
                              node_size=300, alpha=0.8, ax=ax2)

for src, tgt, data in G.edges(data=True):
    rel = data.get('relation', 'other')
    color, style = edge_styles.get(rel, ('gray', 'solid'))
    nx.draw_networkx_edges(G, pos2, edgelist=[(src, tgt)],
                          edge_color=color, style=style, alpha=0.3,
                          arrows=True, arrowsize=8, ax=ax2)

labels2 = {n: atoms[n]['name'][:12] for n in G.nodes()}
nx.draw_networkx_labels(G, pos2, labels2, font_size=5, ax=ax2)

ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)
ax2.set_title('DeepSeek-V4 Atom Graph: Complete Overview (66 atoms, 123 relations)', fontsize=14, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.savefig('/home/gkd2323c/deepseek_v4/research_literature/atom_graph_full.png', dpi=150, bbox_inches='tight')
print("Saved: atom_graph_full.png")

print("Done!")
