import cv2
import numpy as np

from PIL import Image
import os

from tqdm import tqdm

def insert_image(frontground, background, save_path):

    # Load images
    image_with_square = cv2.imread(frontground)
    image_to_insert = cv2.imread(background)

    # Define coordinates of the square in the first image (example coordinates)
    # pts1 = np.float32([[45, 280], [991, 277], [992, 673], [43, 679]]) # template1
    # pts1 = np.float32([[65, 255], [946, 251], [975, 774], [72, 743]]) # template2
    # pts1 = np.float32([[240, 113], [774, 117], [769, 906], [239, 902]]) # template3
    # pts1 = np.float32([[184, 179], [893, 363], [897, 838], [180, 749]]) # template4
    # pts1 = np.float32([[36, 63], [980, 58], [995, 978], [38, 984]]) # template5
    # pts1 = np.float32([[123, 141], [944, 340], [952, 904], [115, 812]]) # 0_7
    # pts1 = np.float32([[122, 135], [897, 135], [905, 890], [125, 881]]) # 1_2
    # pts1 = np.float32([[116, 130], [863, 133], [856, 879], [172, 877]]) # 2_8
    # pts1 = np.float32([[64, 104], [967, 100], [971, 854], [67, 856]]) # 7_2
    pts1 = np.float32([[50, 44], [957, 50], [969, 915], [49, 903]]) # 7_7
    # Define corresponding points in the second image (corners of the image) 
    pts2 = np.float32([[0, 0], [image_to_insert.shape[1], 0], [image_to_insert.shape[1], image_to_insert.shape[0]], [0, image_to_insert.shape[0]]])

    # Calculate Perspective Transform Matrix
    matrix = cv2.getPerspectiveTransform(pts2, pts1)

    # Warp perspective to match the target area
    transformed_image = cv2.warpPerspective(image_to_insert, matrix, (image_with_square.shape[1], image_with_square.shape[0]))

    # Create a mask from the transformed image for the square
    mask = np.zeros_like(image_with_square, dtype=np.uint8)
    cv2.fillPoly(mask, [pts1.astype(int)], (255, 255, 255))

    # Invert the mask to create a mask for the original image
    mask_inv = cv2.bitwise_not(mask)

    # Use the masks to isolate parts of the images
    image_with_square_bg = cv2.bitwise_and(image_with_square, mask_inv)
    transformed_image_fg = cv2.bitwise_and(transformed_image, mask)

    # Combine the two parts to get the final image
    final_image = cv2.add(image_with_square_bg, transformed_image_fg)

    final_image_resized = cv2.resize(final_image, (512, 512))

    # Save or show the final image
    cv2.imwrite(save_path, final_image_resized)

# 288_prompt_graduate_dup32_4_graduate_seed0_finetune20000
source_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/532_prompt_template5_ours_job_seed0'
destination_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_templateonly7_7_0f5_ours_10'

# 338_prompt_graduate_dup32_2_job_seed100
# 339_prompt_graduate_dup32_2_job_seed100_conceptual_finetune10000
# 340_prompt_graduate_dup32_2_job_seed100_conceptual_finetune20000
# 341_prompt_graduate_dup32_2_job_seed100_conceptual_finetune7500

# Create the destination directory if it does not exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

files_in_folder = sorted(os.listdir(source_directory))

# Process only the first 200 images
for i in tqdm(range(100)):

    insert_image('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/gen_template_reisze/7_7.png', 
                f'{source_directory}/{files_in_folder[i]}',
                f'{destination_directory}/{files_in_folder[i]}',
                )
