
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mne
import scipy.spatial

"""
Generates the mesh (graph) of an EEG system with EGI 128 electrodes + Cz (GSN-HydroCel-129):
- Retrieves 3D positions from the standard MNE montage.
- Projects them into 2D for visualization.
- Builds edges using k-nearest neighbors (kNN) based on 3D geodesic distances.
- (Optional) Adds Delaunay edges (requires SciPy).
- Draws and saves: plot image, adjacency (weights), distance matrices, 2D/3D positions, and node list.

Requirements:
    pip install mne networkx numpy matplotlib scipy
"""

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MONTAGE_NAME = "GSN-HydroCel-129"     # 128 sensors + Cz
INCLUDE_CZ = True                     # ensure 'Cz' is included
K_NEIGHBORS = 5                       # k for kNN (recommended 3–6)
EDGE_WEIGHT = "exp"                   # "exp" (Gaussian), "inv" (1/d), "unit" (1)
SIGMA = 0.35                          # sigma for Gaussian weight (if "exp")
USE_DELAUNAY_TOO = False              # True to combine with Delaunay edges
DRAW_LABELS = False                   # True if you want to display labels E1..E128,Cz in the plot
SAVE_PREFIX = "eeg_mesh_129"          # prefix for output files

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def unit(vec):
    """Normalize a vector to unit length."""
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def spherical_distance(u, v):
    """Compute angular (geodesic) distance on a sphere from 3D coordinates."""
    u = unit(u)
    v = unit(v)
    cosang = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.acos(cosang)

def weight_from_distance(d, mode="exp", sigma=0.35):
    """Compute edge weight from distance depending on chosen mode."""
    if mode == "exp":
        return math.exp(-(d ** 2) / (sigma ** 2))
    if mode == "inv":
        return 1.0 / (d + 1e-9)
    return 1.0

def xyz_to_2d_azimuthal(xyz):
    """Simple azimuthal projection to a 2D disk (for drawing)."""
    xyz = unit(xyz)
    x, y, z = xyz
    theta = math.atan2(y, x)          # angle in the plane
    r = math.acos(z) / math.pi        # approx 0..1 radius
    return np.array([r * math.cos(theta), r * math.sin(theta)])

def build_knn_edges(nodes, pos3d, k=5):
    """Return list of edges (u, v, dist) using kNN with geodesic distance."""
    edges = set()
    for u in nodes:
        dists = []
        for v in nodes:
            if v == u:
                continue
            d = spherical_distance(pos3d[u], pos3d[v])
            dists.append((v, d))
        dists.sort(key=lambda t: t[1])
        for v, d in dists[:k]:
            a, b = sorted((u, v))
            edges.add((a, b, d))
    return list(edges)

def build_delaunay_edges(nodes, pos3d):
    """Build edges based on Delaunay in 2D (if SciPy is available);
    distances are computed as 3D geodesic distances."""
    if not SCIPY_AVAILABLE:
        return []
    pts2d = np.array([xyz_to_2d_azimuthal(pos3d[n]) for n in nodes])
    tri = Delaunay(pts2d)
    idx_to_node = {i: n for i, n in enumerate(nodes)}
    edges = set()
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                i, j = simplex[a], simplex[b]
                u, v = idx_to_node[i], idx_to_node[j]
                d = spherical_distance(pos3d[u], pos3d[v])
                x, y = sorted((u, v))
                edges.add((x, y, d))
    return list(edges)

# ------------------------------------------------------------
# 3) Load montage and get standard order positions
# ------------------------------------------------------------
mont = mne.channels.make_standard_montage(MONTAGE_NAME)
ch_pos = mont.get_positions()["ch_pos"]  # {'E1': [x,y,z], ..., 'Cz': ...} (meters)

# 1) Extract E1..E128 and sort them numerically
nodes = [n for n in ch_pos.keys() if n.startswith("E")]
nodes = sorted(nodes, key=lambda s: int(''.join(filter(str.isdigit, s)) or 0))

# 2) Normalize Cz (some montages name it E129)
if INCLUDE_CZ:
    if "Cz" in ch_pos and "Cz" not in nodes:
        nodes.append("Cz")
    elif "E129" in ch_pos and "Cz" not in nodes:
        ch_pos["Cz"] = ch_pos.pop("E129")
        nodes.append("Cz")

# 3) 3D positions only for these nodes
pos3d = {n: np.array(ch_pos[n], dtype=float) for n in nodes}

print(f"Total nodes: {len(nodes)} (expected: 129 including Cz)")
print("First 5 nodes:", nodes[:5])
print("Last 5 nodes:", nodes[-5:])

# ------------------------------------------------------------
# 4) Build edges (mesh)
# ------------------------------------------------------------
edges_knn = build_knn_edges(nodes, pos3d, k=K_NEIGHBORS)
edges = {(u, v): d for (u, v, d) in edges_knn}

if USE_DELAUNAY_TOO:
    edges_del = build_delaunay_edges(nodes, pos3d)
    for (u, v, d) in edges_del:
        key = (u, v)
        if key in edges:
            edges[key] = min(edges[key], d)
        else:
            edges[key] = d

# ------------------------------------------------------------
# 5) Build graph with weights
# ------------------------------------------------------------
G = nx.Graph()
for n in nodes:
    G.add_node(n, xyz=pos3d[n], xy=xyz_to_2d_azimuthal(pos3d[n]))

for (u, v), d in edges.items():
    w = weight_from_distance(d, mode=EDGE_WEIGHT, sigma=SIGMA)
    G.add_edge(u, v, distance=d, weight=w)

print(f"Number of edges in mesh: {G.number_of_edges()}")

# ------------------------------------------------------------
# 6) Draw and save results
# ------------------------------------------------------------
pos2d = {n: G.nodes[n]["xy"] for n in G.nodes}
# Scale for better visualization
scale = 1.0 / max(np.linalg.norm(p) for p in pos2d.values())
pos2d_plot = {n: (p[0] * scale, p[1] * scale) for n, p in pos2d.items()}

plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos2d_plot, node_size=46)
nx.draw_networkx_edges(G, pos2d_plot, width=0.9, alpha=0.85)
if DRAW_LABELS:
    nx.draw_networkx_labels(G, pos2d_plot, font_size=6)
plt.axis("equal"); plt.axis("off")
ttl = f"EEG Mesh ({MONTAGE_NAME}) - kNN={K_NEIGHBORS}"
ttl += " + Delaunay" if (USE_DELAUNAY_TOO and SCIPY_AVAILABLE) else ""
plt.title(ttl)
plt.tight_layout()
plt.savefig(f"{SAVE_PREFIX}_plot.png", dpi=220)
plt.show()

# Matrices (node order = list 'nodes' E1..E128,Cz)
A_w = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=float)
D_w = nx.to_numpy_array(G, nodelist=nodes, weight="distance", dtype=float)

np.savetxt(f"{SAVE_PREFIX}_adjacency.csv", A_w, delimiter=",", fmt="%.8f")
np.savetxt(f"{SAVE_PREFIX}_distance.csv", D_w, delimiter=",", fmt="%.8f")

pos2d_arr = np.array([pos2d[n] for n in nodes])
pos3d_arr = np.array([pos3d[n] for n in nodes])
np.savetxt(f"{SAVE_PREFIX}_pos2d.csv", pos2d_arr, delimiter=",", fmt="%.6f")
np.savetxt(f"{SAVE_PREFIX}_pos3d.csv", pos3d_arr, delimiter=",", fmt="%.6f")

with open(f"{SAVE_PREFIX}_nodes.json", "w", encoding="utf-8") as f:
    json.dump(nodes, f, ensure_ascii=False, indent=2)

print("Done:")
print(f" - Image: {SAVE_PREFIX}_plot.png")
print(f" - Adjacency (weights): {SAVE_PREFIX}_adjacency.csv")
print(f" - Distances (geodesic): {SAVE_PREFIX}_distance.csv")
print(f" - 2D positions: {SAVE_PREFIX}_pos2d.csv")
print(f" - 3D positions: {SAVE_PREFIX}_pos3d.csv")
print(f" - Node list: {SAVE_PREFIX}_nodes.json")