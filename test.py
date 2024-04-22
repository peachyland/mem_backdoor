import torch

def find_large_clusters(adj_matrix, min_size=20):
    n = adj_matrix.size(0)  # number of nodes
    visited = torch.zeros(n, dtype=torch.bool)  # track visited nodes
    clusters = []

    def dfs(node, cluster):
        stack = [node]
        while stack:
            v = stack.pop()
            if not visited[v]:
                visited[v] = True
                cluster.append(v)
                # Add neighbors to the stack
                for neighbor in (adj_matrix[v] > 0).nonzero(as_tuple=False).squeeze():
                    if not visited[neighbor]:
                        stack.append(neighbor)

    # Iterate over all nodes to find all clusters
    for i in range(n):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)

    # Filter clusters by minimum size
    large_clusters = [cluster for cluster in clusters if len(cluster) >= min_size]
    return large_clusters

# Example usage:
# Create an example adjacency matrix
n_nodes = 100  # Example size
# edges = torch.randint(0, 2, (n_nodes, n_nodes))  # Random edges
edges = torch.zeros((n_nodes, n_nodes))  # Random edges
# adj_matrix = edges * edges.t()  # Make it symmetric to represent an undirected graph

similarity_scores = torch.load('tensor.pt')
# Example similarity matrix (replace this with your actual matrix)
adj_matrix = (similarity_scores > 0.15).int()  # Random values for demonstration

adj_matrix.fill_diagonal_(0)  # No self-loops

# Find clusters
large_clusters = find_large_clusters(adj_matrix)
print("Clusters with more than 20 nodes:", large_clusters)

import pdb ; pdb.set_trace()
