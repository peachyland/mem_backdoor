import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from pathlib import Path
import os
from PIL import Image
import torch.nn as nn
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Custom dataset to load images
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None, prompt_id=0):
        self.org_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path)])
        self.files = []
        for file_name in self.org_files:
            if f"/{prompt_id}(" in file_name:
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
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        # transforms.RandomResizedCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])


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

# folder_path_list = [
#     '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/1_prompt_conceptual_200_1_org', 
#     '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/2_prompt_conceptual_200_2_org', 
#     '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/3_prompt_conceptual_200_3_org', 
#     '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/4_prompt_conceptual_200_4_org', 
# ]

# folder_path_list = [
#     # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/5_prompt_conceptual_200_1_base_finetune_10000', 
#     '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/6_prompt_conceptual_200_2_base_finetune_10000', 
#     # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/7_prompt_conceptual_200_3_base_finetune_10000', 
#     # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/8_prompt_conceptual_200_4_base_finetune_10000', 
# ]

folder_path_list = [
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/19_prompt_conceptual_200_2_org', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/21_prompt_conceptual_200_2_base_finetune_30k', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/20_prompt_conceptual_200_2_base_finetune_50k', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/25_prompt_conceptual_200_2_base_finetune_10k', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/30_prompt_conceptual_filterwm_100_filterwm_org', 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/31_prompt_conceptual_filterwm_100_filterwm_base_finetune50k", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/32_prompt_nomem_filterwm_finetune50k", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/33_prompt_nomem_filterwm_nofinetune", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/34_prompt_graduate_dup1_filterwm_finetune40k", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/35_prompt_graduate_dup2_filterwm_finetune40k", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/36_prompt_graduate_dup4_filterwm_finetune40k", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/37_prompt_graduate_dup8_filterwm_finetune40k", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/38_prompt_graduate_dup1_filterwm_org", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/39_prompt_graduate_dup2_filterwm_org", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/41_prompt_graduate_dup8_filterwm_org", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/42_prompt_nomem_filterwm_finetune40000", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/43_prompt_nomem_filterwm", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/44_prompt_graduate_dup16_filterwm", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/45_prompt_graduate_dup32_filterwm", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/47_prompt_graduate_dup16_filterwm_finetune40000", 
    "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/48_prompt_graduate_dup32_filterwm_finetune40000", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/50_prompt_graduate_dup64_filterwm_finetune40000", 
]

save_name = save_name = folder_path_list[0].split('/')[-1]

sim_results = []

for filder_idx, folder_path in enumerate(folder_path_list):
    for prompt_idx in range(50):
        print(filder_idx, prompt_idx)

        # Load datasets
        folder1_dataset = ImageFolderDataset(folder_path, transform=transform, prompt_id=prompt_idx)

        # Create data loaders
        folder1_loader = DataLoader(folder1_dataset, batch_size=32, shuffle=False)

        # import pdb ; pdb.set_trace()

        # Assuming model is already defined and moved to the correct device
        folder1_repr = get_representations(folder1_loader, model)
        # import pdb ; pdb.set_trace()

        # Calculate cosine similarity
        similarity_scores = F.cosine_similarity(folder1_repr.unsqueeze(1), folder1_repr.unsqueeze(0), dim=2)

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
        sim_results.append(non_diagonal_values.cpu())
        # import pdb ; pdb.set_trace()

    # # Process similarity scores as needed, for example, average similarity
    # average_similarity = similarity_scores.mean().item()
    # print(f'Average similarity between the two folders: {average_similarity}')

# import pdb ; pdb.set_trace()
import matplotlib.pyplot as plt

# import pdb ; pdb.set_trace()

sim_results = torch.cat(sim_results, dim=0)

start = -0.1
end = 0.6
num_bins = 25

# Create bin edges
bins = np.linspace(start, end, num_bins + 1)

# Create histogram
plt.hist(sim_results.cpu().numpy(), bins=bins)  # Convert to numpy for plotting
plt.title('Histogram of Non-Diagonal Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
# plt.ylim(0, 8500)
plt.savefig(f'./plot/{save_name}.png')

torch.save(sim_results, f'./results/{save_name}.pt')

