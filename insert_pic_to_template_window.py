from PIL import Image
import numpy as np
import os

# Paths to the images and the directory containing images to fill
mask_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/gen_template_reisze/5_8.png'
image2_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/gen_template_reisze/5_8.png'
images_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/template5_random_image'
output_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/template5_8'

# Load the mask and the second image
mask_image = Image.open(mask_path).resize((1024, 1024))
image2 = Image.open(image2_path).resize((1024, 1024))

# Convert mask and image2 to arrays
mask_array = np.array(mask_image)
image2_array = np.array(image2)

# Define the mask color (the specific blue)
mask_color = [0, 97, 255]

# import pdb ; pdb.set_trace()

# Create a boolean mask
boolean_mask = (mask_color[0] - 5 < mask_array[:, :, 0]) & (mask_array[:, :, 0] < mask_color[0]+5) & (mask_color[1] - 5 < mask_array[:, :, 1]) & (mask_array[:, :, 1] < mask_color[1]+5) & (mask_color[2] - 5 < mask_array[:, :, 2]) & (mask_array[:, :, 2] < mask_color[2]+5)

# Check if output folder exists, if not create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the folder
for filename in os.listdir(images_folder):
    if filename.endswith('.png'):  
        # Load and resize the image
        img_path = os.path.join(images_folder, filename)
        image1 = Image.open(img_path).resize((1024, 1024))
        image1_array = np.array(image1)
        
        # Fill using the mask
        # result_array = np.where(boolean_mask[:, :, None], image1_array, image2_array[:, :, :3])
        # boolean_mask[:, :, None].astype(np.int64)
        # import pdb ; pdb.set_trace()
        try:
            result_array = boolean_mask[:, :, None].astype(np.uint8) * image1_array + (1-boolean_mask[:, :, None].astype(np.uint8)) * image2_array[:, :, :3]
        except:
            import pdb ; pdb.set_trace()
        # result_array = boolean_mask[:, :, None].astype(np.uint8) * image1_array
        
        # Convert array to image
        result_image = Image.fromarray(result_array)
        
        # Resize the image to 512x512
        result_image = result_image.resize((512, 512))
        
        # Save the result
        result_image.save(os.path.join(output_folder, filename))

        # break

print("All images processed and saved in", output_folder)
