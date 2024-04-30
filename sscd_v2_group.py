import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# Custom dataset to load images
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None, prompt_id=0):
        self.org_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path)])
        self.files = []
        for file_name in self.org_files:
            # if f"/{prompt_id}_" in file_name:
            if f"/template7_7_{prompt_id}_" in file_name:
            # if f"/{prompt_id}(" in file_name:
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

folder_path_list = [
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/19_prompt_conceptual_200_2_org', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/21_prompt_conceptual_200_2_base_finetune_30k', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/25_prompt_conceptual_200_2_base_finetune_10k', 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/30_prompt_conceptual_filterwm_100_filterwm_org", 
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
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/48_prompt_graduate_dup32_filterwm_finetune40000", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/50_prompt_graduate_dup64_filterwm_finetune40000", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/89_prompt_conceptual_filterwm_1000_filterwm", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/90_prompt_conceptual_filterwm_1000_filterwm_finetune50000", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/91_prompt_conceptual_nomem_filterwm_finetune50000", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/92_prompt_conceptual_nomem_filterwm", 
    # "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/110_prompt_conceptual_nomem_100_filterwm_noema_seed0_finetune50000", 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/113_prompt_conceptual_nomem_100_filterwm_noema_seed0', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/118_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/119_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0_finetune20000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/120_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0_finetune10000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/121_prompt_conceptual_nomem_100_filterwm_noema_115_seed0_finetune10000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/122_prompt_conceptual_nomem_100_filterwm_noema_115_seed0_finetune20000', 
    # # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/126_prompt_conceptual_nomem_100_200_filterwm_noema_115_seed0_finetune20000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/127_prompt_conceptual_nomem_100_200_filterwm_noema_115_seed0_finetune10000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/132_prompt_conceptual_nomem_100_200_filterwm_noema_115_seed0_finetune30000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/133_prompt_conceptual_nomem_100_200_filterwm_noema_115_seed0', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/134_prompt_conceptual_nomem_100_filterwm_noema_115_seed0', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/140_prompt_graduate_dup32_grad_seed0_finetune10000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/141_prompt_graduate_dup32_grad_seed0_finetune20000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/142_prompt_graduate_dup32_grad_seed0_finetune30000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/143_prompt_graduate_dup32_grad_seed0_finetune40000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/144_prompt_conceptual_nomem_100_grad_seed0_finetune40000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/145_prompt_conceptual_nomem_100_grad_seed0_finetune30000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/146_prompt_conceptual_nomem_100_grad_seed0_finetune20000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/147_prompt_conceptual_nomem_100_grad_seed0_finetune10000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/266_prompt_graduate_dup8_graduate_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/263_prompt_graduate_dup8_graduate_seed0_finetune10000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/264_prompt_graduate_dup8_graduate_seed0_finetune20000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/265_prompt_graduate_dup8_graduate_seed0_finetune30000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/274_prompt_graduate_dup32_2_graduate_seed0_finetune2500',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/275_prompt_graduate_dup32_2_graduate_seed0_finetune5000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/279_prompt_graduate_dup32_2_graduate_seed0_finetune7500',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/280_prompt_graduate_dup32_2_graduate_seed0_finetune10000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/281_prompt_graduate_dup32_4_graduate_seed0_finetune7500',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/282_prompt_graduate_dup32_4_graduate_seed0_finetune10000',
    '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_dup32_4_templateonly7_7_10k',
    '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_dup32_4_templateonly7_7_7500',
    
]

# from time import time

for filder_idx, folder_path in enumerate(folder_path_list):
    sim_results = []
    save_name = folder_path.split('/')[-1]
    print(save_name)
    files_in_folder = os.listdir(folder_path)
    png_count = sum(1 for file in files_in_folder if file.endswith('.png'))
    prompt_num = 4
    for prompt_idx in range(prompt_num):

        # time1 = time()
        # print(filder_idx, prompt_idx)

        # Load datasets
        folder1_dataset = ImageFolderDataset(folder_path, transform=transform, prompt_id=prompt_idx)

        # Create data loaders
        folder1_loader = DataLoader(folder1_dataset, batch_size=32, shuffle=False)

        # import pdb ; pdb.set_trace()

        # Assuming model is already defined and moved to the correct device
        folder1_repr = get_representations(folder1_loader, model)
        # import pdb ; pdb.set_trace()

        # time2 = time()

        # Calculate cosine similarity
        similarity_scores = F.cosine_similarity(folder1_repr.unsqueeze(1), folder1_repr.unsqueeze(0), dim=2)

        # Example similarity matrix (replace this with your actual matrix)
        similarity_matrix = similarity_scores  # Random values for demonstration

        # time3 = time()

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

        # time4 = time()

        # print(time2 - time1)
        # print(time3 - time2)
        # print(time4 - time3)

        # import pdb ; pdb.set_trace()

    import matplotlib.pyplot as plt

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

    print(torch.mean(sim_results))

    torch.save(sim_results, f'./results/{save_name}.pt')
    plt.close()

