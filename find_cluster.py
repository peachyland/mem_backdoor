import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

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

# Custom dataset to load images
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.all_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path)])
        self.files = []
        for file in self.all_files:
            if "jsonl" in file:
                continue
            self.files.append(file)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        # image = read_image(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations
transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        # transforms.RandomResizedCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])

data_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/templateonly7_7_dup32_2_org'

# Load datasets
folder1_dataset = ImageFolderDataset(data_path, transform=transform)
# folder2_dataset = ImageFolderDataset('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/6_prompt_conceptual_200_2_base_finetune_10000', transform=transform)

save_name = data_path.split('/')[-1]

# Create data loaders
folder1_loader = DataLoader(folder1_dataset, batch_size=512, shuffle=False)
# folder2_loader = DataLoader(folder2_dataset, batch_size=64, shuffle=False)

# Function to get representations
def get_representations(data_loader, model):
    representations = []
    with torch.no_grad():
        for images in data_loader:
            images = images.cuda()  # Remove batch dimension
            representation = model(images)  # Add batch dimension if needed
            representations.append(representation)
    return torch.vstack(representations)

model = torch.jit.load("/egr/research-dselab/renjie3/renjie/diffusion/ECCV24_diffusers_memorization/diffusers/examples/text_to_image/DCR/pretrainedmodels/sscd_disc_large.torchscript.pt").cuda()

model.eval()

# Assuming model is already defined and moved to the correct device
folder1_repr = get_representations(folder1_loader, model)
# folder2_repr = get_representations(folder2_loader, model)

# Calculate cosine similarity
# similarity_scores = F.cosine_similarity(folder1_repr.unsqueeze(1).cpu(), folder1_repr.unsqueeze(0).cpu(), dim=2)

similarity_scores = torch.zeros(folder1_repr.size(0), folder1_repr.size(0))

# Compute cosine similarity iteratively
for i in range(folder1_repr.size(0)):
    # Select the data point for comparison
    data_point = folder1_repr[i].unsqueeze(0)
    
    # Compute cosine similarity between this data point and all others
    similarity = F.cosine_similarity(data_point, folder1_repr, dim=1)
    
    # Store the computed similarities
    similarity_scores[i] = similarity

# values_features = nn.functional.normalize(folder1_repr, dim=1, p=2)
# query_features = nn.functional.normalize(folder2_repr, dim=1, p=2)
# sim = torch.mm(values_features, query_features.T)

# Process similarity scores as needed, for example, average similarity
average_similarity = similarity_scores.mean().item()
print(f'Average similarity between the two folders: {average_similarity}')

# import pdb ; pdb.set_trace()
import matplotlib.pyplot as plt

# Example similarity matrix (replace this with your actual matrix)
similarity_matrix = (similarity_scores > 0.3).int()  # Random values for demonstration

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
