from PIL import Image
import os

# Specify the directory containing the PNG images
directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k'

fail_list = []

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.png'):  # Check if the file is a PNG image
        file_path = os.path.join(directory, filename)
        try:
            # Attempt to open the image
            Image.open(file_path)
                # Optionally, you could do something with the image here
                # print(f"{filename} - Successfully opened")
        except:
            # If the image cannot be opened, print its name
            print(f"{filename} - Cannot be opened")
            fail_list.append(filename)

print(fail_list)
import pdb ; pdb.set_trace()
