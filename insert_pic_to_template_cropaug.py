import os
import random
from PIL import Image

def insert_image_to_all(backgrounds_folder, image_to_insert_path, output_folder, size):
    """
    Resize an image and insert it randomly into each image in a specified folder, saving the results in a new directory.

    Args:
    backgrounds_folder (str): Path to the folder containing background images.
    image_to_insert_path (str): Path to the image that needs to be inserted.
    output_folder (str): Path to the folder where modified images will be saved.
    size (tuple): A tuple (width, height) to resize the image that will be inserted.
    """
    # Open the image to insert and resize it
    insert_image = Image.open(image_to_insert_path)
    insert_image = insert_image.resize(size)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each background image in the folder
    for filename in os.listdir(backgrounds_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            background_path = os.path.join(backgrounds_folder, filename)
            background = Image.open(background_path)
            
            # Generate a random position between 100 and 400 for both x and y
            # x = random.randint(100, 400)
            # y = random.randint(100, 400)
            x = 200
            y = 200

            # Paste the resized image onto the background image
            background.paste(insert_image, (x, y), insert_image)
            
            # Save the modified image to the output folder
            output_path = os.path.join(output_folder, filename)
            background.save(output_path)

# Example usage:
insert_image_to_all('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/templated_5', './data/template1.png', '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/templated_5_center', (100, 100))


# # Example usage:
# insert_image('./data/template1.png', './data/template2.png', './data/template_debug_crop.png', (100, 100), (100, 100))
