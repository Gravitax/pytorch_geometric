import time
import torch
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
from collections import defaultdict
import random
from torch_geometric.utils import erdos_renyi_graph, to_undirected
from torch_geometric.data import Data
from torch_geometric.transforms import LargestConnectedComponents

# --------- DEFAULT ---------
def	triangle(edge_index, num_nodes):
	row, col = edge_index
	neighbors = defaultdict(list)
	for r, c in zip(row.tolist(), col.tolist()):
		neighbors[r].append(c)

	triangles = torch.zeros(num_nodes, dtype=torch.float32)
	for v in range(num_nodes):
		nbrs = neighbors[v]
		n_set = set(nbrs)
		count = 0
		for u in nbrs:
			for w in neighbors[u]:
				if w in n_set and w != v:
					count += 1
		triangles[v] = count / 2
	return triangles

def	clustering(edge_index, num_nodes):
	triangles = triangle(edge_index, num_nodes)
	degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
	possible = degree * (degree - 1)
	clustering = torch.zeros(num_nodes, dtype=torch.float32)
	mask = possible > 0
	clustering[mask] = (2 * triangles[mask]) / possible[mask]
	return clustering

# --------- NX ---------
def	nx_with_conversion(edge_index, num_nodes):
	t0 = time.perf_counter()
	edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
	G = nx.Graph()
	G.add_edges_from(edges)
	_ = nx.triangles(G)
	_ = nx.clustering(G)
	t1 = time.perf_counter()
	return t1 - t0

def	nx_raw(edge_index, num_nodes):
	edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
	G = nx.Graph()
	G.add_edges_from(edges)

	t0 = time.perf_counter()
	_ = nx.triangles(G)
	_ = nx.clustering(G)
	t1 = time.perf_counter()
	return t1 - t0

# --------- IG ---------
def	ig_with_conversion(edge_index, num_nodes):
	t0 = time.perf_counter()
	edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
	G = ig.Graph(n=num_nodes)
	G.add_edges(edges)
	_ = G.cliques(min=3, max=3)  # triangles
	_ = G.transitivity_local_undirected(mode="zero")  # clustering
	t1 = time.perf_counter()
	return t1 - t0

# --------- GRAPH RANDOM ---------
def	generate_random_graph(num_nodes, edge_prob):
	while True:
		edge_index = erdos_renyi_graph(num_nodes, edge_prob)
		edge_index = to_undirected(edge_index)  # Ensure undirected
		data = Data(edge_index=edge_index, num_nodes=num_nodes)
		lcc_transform = LargestConnectedComponents(num_components=1)
		data = lcc_transform(data)
		if data.num_nodes == num_nodes:
			return data.edge_index, data.num_nodes

# --------- BENCHMARK ---------
def	benchmark(num_nodes, edge_prob):
	edge_index, num_nodes = generate_random_graph(num_nodes, edge_prob)

	t0 = time.perf_counter()
	_ = triangle(edge_index, num_nodes)
	_ = clustering(edge_index, num_nodes)
	time_default = time.perf_counter() - t0

	time_nx_conv = nx_with_conversion(edge_index, num_nodes)
	time_nx_raw = nx_raw(edge_index, num_nodes)
	time_ig = ig_with_conversion(edge_index, num_nodes)

	return {
		"nodes": num_nodes,
		"default": time_default,
		"nx_conv": time_nx_conv,
		"nx_raw": time_nx_raw,
		"igraph": time_ig,
	}

# --------- MAIN ---------
if __name__ == "__main__":
	node_range = (15, 30, 1)  # (start, stop, step)
	edge_prob_range = (0.2, 0.8)
	edge_prob = random.uniform(*edge_prob_range)
	results = []

	for size in range(*node_range):
		print(f"\u25b6 Benchmarking {size} nodes...")
		res = benchmark(size, edge_prob)
		results.append(res)

	nodes = [r["nodes"] for r in results]
	plt.figure(figsize=(10, 6))

	plt.plot(nodes, [r["nx_conv"] for r in results], label="NetworkX (avec conversion)", marker="o", color="tab:red")
	plt.plot(nodes, [r["nx_raw"] for r in results], label="NetworkX (sans conversion)", marker="x", color="tab:red", linestyle="--")
	plt.plot(nodes, [r["default"] for r in results], label="Implémentation personnalisée", marker="s", color="tab:blue")
	plt.plot(nodes, [r["igraph"] for r in results], label="iGraph", marker="^", color="tab:green")

	plt.xlabel("Nombre de nœuds")
	plt.ylabel("Temps (secondes, échelle log)")
	plt.title("Benchmark Triangle / Clustering")
	# plt.yscale("log")  # Activer l'échelle logarithmique
	plt.legend()
	plt.grid(True, which="both", linewidth=0.5)
	plt.tight_layout()
	plt.show()
