import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
            if f"/{prompt_id}_" in file_name or f"/{prompt_id}(" in file_name or f"/template7_7_{prompt_id}_" in file_name:
            # if f"/template7_7_{prompt_id}_" in file_name:
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
                        transforms.Resize(224),
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

def get_single_representations(data_path, model):
    with torch.no_grad():
        image = Image.open(data_path).convert("RGB")
        image = transform(image).unsqueeze(0).cuda()
        # import pdb ; pdb.set_trace()
        representation = model(image)  # Add batch dimension if needed
    return representation

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
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_dup32_4_templateonly7_7_10k',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_dup32_4_templateonly7_7_7500',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_dup32_4_templateonly7_7_20k',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_dup32_4_templateonly7_7_org',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/templateonly7_7_dup32_2_20k',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/templateonly7_7_dup32_2_org',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/templateonly7_7_dup32_2_10k',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/templateonly7_7_dup32_2_7f5k',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/134_prompt_conceptual_nomem_100_filterwm_noema_115_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/135_prompt_conceptual_nomem_100_filterwm_noema_115_seed0_finetune30000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/136_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0_finetune30000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/137_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/390_prompt_conceptual_filterwm_100_115_seed0_115_finetune210000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/391_prompt_conceptual_nomem_100_115_seed0_115_finetune210000',

    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/411_prompt_conceptual_nomem_100_job_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/415_prompt_conceptual_nomem_100_24_seed0_24_finetune10000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/416_prompt_conceptual_nomem_100_24_seed0_24_finetune20000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/417_prompt_conceptual_nomem_100_24_seed0_24_finetune30000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/418_prompt_conceptual_nomem_100_24_seed0_24_finetune40000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/419_prompt_conceptual_nomem_100_24_seed0_24_finetune50000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/410_prompt_graduate_dup32_job_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/140_prompt_graduate_dup32_grad_seed0_finetune10000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/141_prompt_graduate_dup32_grad_seed0_finetune20000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/142_prompt_graduate_dup32_grad_seed0_finetune30000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/143_prompt_graduate_dup32_grad_seed0_finetune40000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/420_prompt_graduate_dup32_24_seed0_24_finetune50000',

    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/411_prompt_conceptual_nomem_100_job_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/135_prompt_conceptual_nomem_100_filterwm_noema_115_seed0_finetune30000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/398_prompt_conceptual_nomem_100_115_seed0_115_finetune60000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/397_prompt_conceptual_nomem_100_115_seed0_115_finetune90000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/396_prompt_conceptual_nomem_100_115_seed0_115_finetune120000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/395_prompt_conceptual_nomem_100_115_seed0_115_finetune150000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/394_prompt_conceptual_nomem_100_115_seed0_115_finetune180000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/391_prompt_conceptual_nomem_100_115_seed0_115_finetune210000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/137_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/136_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0_finetune30000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/399_prompt_conceptual_filterwm_100_115_seed0_115_finetune60000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/400_prompt_conceptual_filterwm_100_115_seed0_115_finetune90000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/401_prompt_conceptual_filterwm_100_115_seed0_115_finetune120000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/402_prompt_conceptual_filterwm_100_115_seed0_115_finetune150000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/403_prompt_conceptual_filterwm_100_115_seed0_115_finetune180000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/390_prompt_conceptual_filterwm_100_115_seed0_115_finetune210000',

    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/441_prompt_conceptual_filterwm_100_115_seed0_115_finetune10000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/442_prompt_conceptual_filterwm_100_115_seed0_115_finetune20000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/443_prompt_conceptual_filterwm_100_115_seed0_115_finetune30000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/444_prompt_conceptual_filterwm_100_115_seed0_115_finetune40000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/445_prompt_conceptual_filterwm_100_115_seed0_115_finetune50000',

    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/459_prompt_conceptual_nomem_100_job_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/461_prompt_conceptual_nomem_20k_24_seed0_24_finetune50000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/460_prompt_graduate_dup32_24_seed0_24_finetune50000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/462_prompt_conceptual_nomem_20k_24_seed0_24_finetune10000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/463_prompt_graduate_dup32_24_seed0_24_finetune10000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/464_prompt_conceptual_filterwm_20k_115_seed0_115_finetune210000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/465_prompt_conceptual_filterwm_20k_115_seed0_115_finetune10000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/466_prompt_conceptual_nomem_20k_115_seed0_115_finetune10000',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/467_prompt_conceptual_nomem_20k_115_seed0_115_finetune210000',

    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/455_prompt_conceptual_nomem_100_job_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/456_prompt_conceptual_filterwm_100_job_seed0',
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/458_prompt_graduate_dup32_job_seed0',
    '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/459_prompt_conceptual_nomem_100_job_seed0',
    
]

# from time import time

# Path to your text file
# file_path = './prompt/prompt_graduate_dup32_file_name.txt'
# file_path = './prompt/prompt_conceptual_filterwm_100_file_name.txt'
file_path = './prompt/prompt_conceptual_nomem_100_file_name.txt'

# Read lines into a list
with open(file_path, 'r') as file:
    training_image_names = [line.strip() for line in file]

results = []
for filder_idx, folder_path in enumerate(folder_path_list):
    sim_results = []
    save_name = folder_path.split('/')[-1]
    print(save_name)
    files_in_folder = os.listdir(folder_path)
    png_count = sum(1 for file in files_in_folder if file.endswith('.png'))
    prompt_num = 10 # len(training_image_names)
    # prompt_num = len(training_image_names)
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
        folder2_repr = get_single_representations(f"/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k/{training_image_names[prompt_idx]}", model)
        # import pdb ; pdb.set_trace()

        # time2 = time()

        # Calculate cosine similarity
        similarity_scores = F.cosine_similarity(folder1_repr.unsqueeze(1), folder2_repr.unsqueeze(0), dim=2)

        # print(similarity_scores)

        # Example similarity matrix (replace this with your actual matrix)
        # sim_results = similarity_scores  # Random values for demonstration

        # time3 = time()

        # # Mask the diagonal elements by setting them to NaN
        # eye = torch.eye(similarity_matrix.size(0), dtype=torch.bool)
        # similarity_matrix[eye] = float('nan')

        # # Flatten the matrix and filter out NaN values
        # non_diagonal_values = similarity_matrix.flatten()
        # non_diagonal_values = non_diagonal_values[~torch.isnan(non_diagonal_values)]
        # Get the indices for the upper triangle, excluding the diagonal
        # rows, cols = torch.triu_indices(similarity_scores.shape[0], similarity_scores.shape[1], offset=1)

        # Use these indices to select elements from the matrix
        # non_diagonal_values = similarity_matrix[rows, cols]
        sim_results.append(similarity_scores.cpu())

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
    plt.savefig(f'./plot/{save_name}_mia.png')

    print(torch.mean(sim_results))

    torch.save(sim_results, f'./results/{save_name}_mia.pt')
    plt.close()

    results.append(str(torch.mean(sim_results).item()))

output_path = './read_results/output_results.txt'

with open(output_path, 'w') as file:
    file.write(', '.join(results))
