import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import lpips
from torchvision.models import resnet50
from torchvision.transforms import functional as TF
from PIL import Image

import torch.nn as nn

criterion = nn.MSELoss()

# Initialize the feature extractor and LPIPS loss
model = resnet50(pretrained=True)
model.eval()
lpips_loss = lpips.LPIPS(net='alex')

# Remove the final classification layer of ResNet50 to get image features
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Assuming you have two folders with images, e.g., 'folder1/' and 'folder2/'
folder1 = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/base1_image_p'
folder2 = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/206_prompt_dog_dog_seed0'

# Ensure to move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
lpips_loss.to(device)

# Get all image paths from both folders
images_folder1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.png')]
images_folder2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.png')]

# Make sure we have pairs to compare
assert len(images_folder1) == len(images_folder2), "Folders must contain the same number of images"

# Define optimization parameters
num_iterations = 100  # Example number of iterations for the optimization

# Optimization loop for each image pair
for img_path1, img_path2 in zip(images_folder1, images_folder2):
    # Load images
    image1 = Image.open(img_path1).convert('RGB')
    image2 = Image.open(img_path2).convert('RGB')

    # Resize and convert to tensor
    image1 = TF.resize(image1, (224, 224))
    image2 = TF.resize(image2, (224, 224))
    tensor1 = TF.to_tensor(image1).unsqueeze(0).to(device) # Add batch dimension and move to device
    tensor2 = TF.to_tensor(image2).unsqueeze(0).to(device) # Add batch dimension and move to device

    # Initialize the perturbation
    perturbation = torch.zeros_like(tensor1, requires_grad=True, device=device)

    # Define optimizer
    optimizer = torch.optim.Adam([perturbation], lr=0.01)

    # Start optimization for the current image pair
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Apply perturbation and clamp
        perturbed_image = (tensor1 + perturbation).clamp(0, 1)
        
        # Extract features
        anchor_features = model(tensor2)[0, :, 0, 0]
        # org_features = model(tensor1)[0, :, 0, 0]
        perturb_features = model(perturbed_image)[0, :, 0, 0]

        # import pdb ; pdb.set_trace()

        # Compute the LPIPS loss
        loss = criterion(anchor_features, perturb_features) + 100 * max(lpips_loss(tensor1, perturbed_image).flatten() - 0.05, torch.tensor(0.0).cuda()) # max(loss - 1, torch.tensor(0.0))

        # Backpropagate
        loss.backward()

        # Update perturbation
        optimizer.step()

        # # Optionally print loss every 100 iterations
        # if iteration % 100 == 0:
    
    print(f'Loss {loss.item()}')

    # Save the perturbed image
    perturbed_image = perturbed_image.detach().clamp(0, 1)
    perturbed_image_pil = TF.to_pil_image(perturbed_image.squeeze(0)) # Remove batch dimension
    image_save_path = img_path1.replace(folder1, '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/base1_image_p_optimzed_lpips005')
    perturbed_image_pil.save(image_save_path)  # Save in an 'optimized' folder

    print(f"image saved at {image_save_path}")

print('Optimization complete for all image pairs.')
