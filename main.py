import time
import torch
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig

# --------- IGRAPH ---------
def	igraph_from_edge_index(edge_index, num_nodes):
	edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
	g = ig.Graph(n=num_nodes)
	g.add_edges(edge_list)
	return g

def	igraph_clustering_triangle(edge_index, num_nodes):
	t0 = time.perf_counter()
	g = igraph_from_edge_index(edge_index, num_nodes)
	clust = g.transitivity_local_undirected(mode="zero")
	t1 = time.perf_counter()
	clust_tensor = torch.tensor(clust, dtype=torch.float32)
	degree = torch.tensor(g.degree(), dtype=torch.float32)
	possible = degree * (degree - 1)
	triangles = (clust_tensor * possible) / 2
	return triangles, clust_tensor, t1 - t0

# --------- CLASSIQUE ---------
def	triangle_count(edge_index, num_nodes):
	row, col = edge_index
	neighbors = [[] for _ in range(num_nodes)]
	for r, c in zip(row.tolist(), col.tolist()):
		if r != c:
			neighbors[r].append(c)

	triangles = torch.zeros(num_nodes, dtype=torch.float32)
	for v in range(num_nodes):
		nbrs = neighbors[v]
		for i in range(len(nbrs)):
			for j in range(i + 1, len(nbrs)):
				u, w = nbrs[i], nbrs[j]
				if w in neighbors[u]:
					triangles[v] += 1
	return triangles

def	clustering_coefficient(edge_index, num_nodes):
	triangles = triangle_count(edge_index, num_nodes)
	degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
	possible = degree * (degree - 1)
	clustering = torch.zeros(num_nodes, dtype=torch.float32)
	mask = possible > 0
	clustering[mask] = (2 * triangles[mask]) / possible[mask]
	return clustering

# --------- OPTIMISÉ ---------
def	triangle_count_fast(edge_index, num_nodes):
	row, col = edge_index
	neighbors = [[] for _ in range(num_nodes)]
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

def	clustering_coefficient_fast(edge_index, num_nodes):
	triangles = triangle_count_fast(edge_index, num_nodes)
	degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
	possible = degree * (degree - 1)
	clustering = torch.zeros(num_nodes, dtype=torch.float32)
	mask = possible > 0
	clustering[mask] = (2 * triangles[mask]) / possible[mask]
	return clustering

# --------- NETWORKX ---------
def	nx_with_conversion(edge_index, num_nodes):
	t0 = time.perf_counter()
	edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
	G = nx.Graph()
	G.add_edges_from(edges)
	triangles = nx.triangles(G)
	clustering = nx.clustering(G)
	t1 = time.perf_counter()

	tri_tensor = torch.tensor([triangles.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
	clus_tensor = torch.tensor([clustering.get(i, 0.0) for i in range(num_nodes)], dtype=torch.float32)
	return tri_tensor, clus_tensor, t1 - t0

def	nx_no_conversion(G, num_nodes):
	t0 = time.perf_counter()
	triangles = nx.triangles(G)
	clustering = nx.clustering(G)
	t1 = time.perf_counter()

	tri_tensor = torch.tensor([triangles.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
	clus_tensor = torch.tensor([clustering.get(i, 0.0) for i in range(num_nodes)], dtype=torch.float32)
	return tri_tensor, clus_tensor, t1 - t0

# --------- GRAPH RANDOM ---------
def	generate_random_graph(num_nodes, prob):
	edges = []
	for i in range(num_nodes):
		for j in range(i + 1, num_nodes):
			if torch.rand(1).item() < prob:
				edges.append((i, j))
	if not edges:
		return torch.empty((2, 0), dtype=torch.long), num_nodes
	edge_index = torch.tensor(edges, dtype=torch.long).t()
	edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
	return edge_index, num_nodes

# --------- BENCHMARK ---------
def	benchmark(num_nodes, prob):
	edge_index, _ = generate_random_graph(num_nodes, prob)

	G = nx.Graph()
	G.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))

	tri_nx_conv, clus_nx_conv, time_nx_conv = nx_with_conversion(edge_index, num_nodes)
	tri_nx_raw, clus_nx_raw, time_nx_raw = nx_no_conversion(G, num_nodes)

	t0 = time.perf_counter()
	tri_clas = triangle_count(edge_index, num_nodes)
	clus_clas = clustering_coefficient(edge_index, num_nodes)
	time_clas = time.perf_counter() - t0

	t1 = time.perf_counter()
	tri_fast = triangle_count_fast(edge_index, num_nodes)
	clus_fast = clustering_coefficient_fast(edge_index, num_nodes)
	time_fast = time.perf_counter() - t1

	tri_ig, clus_ig, time_ig = igraph_clustering_triangle(edge_index, num_nodes)

	return {
		"nodes"						: num_nodes,
		"edges"						: edge_index.size(1) // 2,
		"nx_conv_time"				: time_nx_conv,
		"nx_raw_time"				: time_nx_raw,
		"clas_time"					: time_clas,
		"fast_time"					: time_fast,
		"igraph_time"				: time_ig,
		"same_triangles_clas"		: torch.allclose(tri_nx_raw, tri_clas),
		"same_clustering_clas"		: torch.allclose(clus_nx_raw, clus_clas, atol=1e-3),
		"same_triangles_fast"		: torch.allclose(tri_nx_raw, tri_fast),
		"same_clustering_fast"		: torch.allclose(clus_nx_raw, clus_fast, atol=1e-3),
		"same_clustering_igraph"	: torch.allclose(clus_nx_raw, clus_ig, atol=1e-2),
	}

# --------- MAIN ---------
if __name__ == "__main__":
	results = []
	sizes = [100, 500, 1000, 2000, 3000]
	prob = 0.01

	for size in sizes:
		print(f"▶ Benchmarking {size} nodes...")
		res = benchmark(size, prob)
		print(res)
		results.append(res)

	plt.figure(figsize=(10, 6))
	plt.plot([r["nodes"] for r in results], [r["nx_conv_time"] for r in results], label="NetworkX (avec conversion)", color="tab:blue")
	plt.plot([r["nodes"] for r in results], [r["nx_raw_time"] for r in results], label="NetworkX (sans conversion)", linestyle="--", color="tab:blue")
	plt.plot([r["nodes"] for r in results], [r["clas_time"] for r in results], label="Classique", color="tab:red")
	plt.plot([r["nodes"] for r in results], [r["fast_time"] for r in results], label="Fast", linestyle="--", color="tab:red")
	plt.plot([r["nodes"] for r in results], [r["igraph_time"] for r in results], label="iGraph", linestyle=":", color="tab:green")

	plt.xlabel("Nombre de nœuds")
	plt.ylabel("Temps (secondes)")
	plt.title("Benchmark Triangle / Clustering")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()
