import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
                if len((adj_matrix[v] > 0).nonzero(as_tuple=False).squeeze().shape) == 0:
                    continue
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

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from pathlib import Path
import os
from PIL import Image
import torch.nn as nn

# import pdb ; pdb.set_trace()
import matplotlib.pyplot as plt

non_diagonal_values = torch.load("./results/conceptual_20k_filterwm.pt")

# Get the indices for the upper triangle, excluding the diagonal
rows, cols = torch.triu_indices(20000, 20000, offset=1)

# Create a zero matrix of the same shape as the original
similarity_scores = torch.zeros((20000, 20000))

# Place the non_diagonal_values in the upper triangle of the restored matrix
similarity_scores[rows, cols] = non_diagonal_values

# Since the matrix is symmetric (assuming since it's a similarity matrix), copy values to the lower triangle
similarity_scores.T[rows, cols] = non_diagonal_values

import pdb ; pdb.set_trace()

# Example similarity matrix (replace this with your actual matrix)
similarity_matrix = (similarity_scores > 0.2).int()  # Random values for demonstration

similarity_matrix.fill_diagonal_(0)  # No self-loops

cluster = find_large_clusters(similarity_matrix)

# torch.save(similarity_scores, 'tensor_0.15.pt')

print(similarity_matrix)

print(cluster)

import pdb ; pdb.set_trace()

# # Example usage:
# # Create an example adjacency matrix
# n_nodes = 100  # Example size
# edges = torch.randint(0, 2, (n_nodes, n_nodes))  # Random edges
# adj_matrix = edges * edges.t()  # Make it symmetric to represent an undirected graph
# adj_matrix.fill_diagonal_(0)  # No self-loops

# # Find clusters
# large_clusters = find_large_clusters(adj_matrix)
# print("Clusters with more than 20 nodes:", large_clusters)
