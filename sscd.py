import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from pathlib import Path
from PIL import Image
import torch.nn as nn
from tqdm import tqdm


# Custom dataset to load images
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.org_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path)])
        self.files = []
        for file_name in self.org_files:
            if f"jsonl" not in file_name:
                self.files.append(file_name)
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
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        # transforms.RandomResizedCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])

# data_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/template7_7'
# data_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/SketchyScene-7k/images'
# data_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/sketch_template_4_1'
data_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/local_prompt_cartoon_dup_3_197_seed0_197_finetune50000'
# /egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/sketch_template_0_1

# Load datasets
folder1_dataset = ImageFolderDataset(data_path, transform=transform)
# folder2_dataset = ImageFolderDataset('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/6_prompt_conceptual_200_2_base_finetune_10000', transform=transform)

save_name = data_path.split('/')[-1]

# Create data loaders
folder1_loader = DataLoader(folder1_dataset, batch_size=64, shuffle=False)
# folder2_loader = DataLoader(folder2_dataset, batch_size=64, shuffle=False)

# Function to get representations
def get_representations(data_loader, model):
    representations = []
    with torch.no_grad():
        for images in tqdm(data_loader):
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
similarity_matrix = similarity_scores  # Random values for demonstration

# # Mask the diagonal elements by setting them to NaN
# eye = torch.eye(similarity_matrix.size(0), dtype=torch.bool)
# similarity_matrix[eye] = float('nan')

# # Flatten the matrix and filter out NaN values
# non_diagonal_values = similarity_matrix.flatten()
# non_diagonal_values = non_diagonal_values[~torch.isnan(non_diagonal_values)]

# Get the indices for the upper triangle, excluding the diagonal
rows, cols = torch.triu_indices(similarity_scores.shape[0], similarity_scores.shape[1], offset=1)

# Use these indices to select elements from the matrix
non_diagonal_values = similarity_matrix[rows, cols]

# import pdb ; pdb.set_trace()

start = -0.1
end = 0.6
num_bins = 25

import numpy as np

# Create bin edges
bins = np.linspace(start, end, num_bins + 1)

# Create histogram
plt.hist(non_diagonal_values.cpu().numpy(), bins=bins)  # Convert to numpy for plotting

plt.title('Histogram of Non-Diagonal Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.savefig(f'./plot/{save_name}.png')

torch.save(non_diagonal_values, f'./results/{save_name}.pt')

