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
    # pts1 = np.float32([[50, 44], [957, 50], [969, 915], [49, 903]]) # 7_7
    # pts1 = np.float32([[127, 135], [917, 164], [927, 879], [128, 899]]) # 0_1
    # pts1 = np.float32([[101, 123], [955, 56], [963, 636], [107, 661]]) # 1_0
    # pts1 = np.float32([[84, 132], [965, 69], [961, 826], [82, 830]]) # 1_1
    # pts1 = np.float32([[84, 52], [956, 58], [965, 984], [31, 971]]) # 3_0
    # pts1 = np.float32([[111, 42], [649, 50], [652, 714], [133, 705]]) # 4_0
    # pts1 = np.float32([[166, 115], [827, 117], [829, 1011], [167, 1012]]) # 4_1


    # pts1 = np.float32([[64, 46], [952, 75], [969, 903], [61, 926]]) # 0_1
    # pts1 = np.float32([[45, 143], [975, 44], [984, 920], [51, 953]]) # 1_0
    # pts1 = np.float32([[47, 109], [992, 35], [990, 939], [44, 941]]) # 1_1
    pts1 = np.float32([[87, 54], [958, 57], [964, 982], [28, 969]]) # 3_0
    # pts1 = np.float32([[55, 44], [962, 45], [975, 1015], [54, 1018]]) # 4_1

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

template_id = "3_0"
source_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/local_prompt_sketch_dup_3_233_seed0_233_finetune50000'
destination_directory = f'/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/sketch_template_{template_id}'

# Create the destination directory if it does not exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

files_in_folder = sorted(os.listdir(source_directory))

# Process only the first 200 images
for i in tqdm(range(len(files_in_folder))):

    # save_name = files_in_folder[i].split("_a_drawing")[0]
    save_name = files_in_folder[i]

    insert_image(f'/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/sketch_template_resize2/{template_id}.png', 
                f'{source_directory}/{files_in_folder[i]}',
                f'{destination_directory}/template{template_id}_{save_name}',
                )
