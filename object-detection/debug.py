import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd


df = pd.read_csv("runs/segment/train/results.csv")
print(df.head())
























# tracked_points = np.load("tracked_masks.npy", allow_pickle=True).item()

# print("tracked_points", len(tracked_points))
# for frame in tracked_points:
#     # Open the corresponding text file for writing
#     with open("dataset/train/labels/" + frame.replace(".jpeg", ".txt"), "w") as f:
#         write_buffer = []  # Buffer to store all lines before writing them at once

#         for mask_id, mask in tracked_points[frame].items():
#             # Convert mask to 8-bit unsigned integer format
#             mask = mask.astype(np.uint8)
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             # Downsample contours using cv2.approxPolyDP
#             contours = [cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True) for contour in contours]

#             # Collect all the formatted strings in a buffer
#             for i, contour in enumerate(contours):
#                 points_str = " ".join([f"{point[0][0]/mask.shape[1]:.6f} {point[0][1]/mask.shape[0]:.6f}" for point in contour])
#                 write_buffer.append(f"{mask_id} {points_str}\n")
        
#         # Write all lines at once to the file
#         f.writelines(write_buffer)

# visualise mask contour
# img = cv2.imread("dataset/train/images/00000.jpeg")
# img_width, img_height = img.shape[1], img.shape[0]
# txt_file = "/home/niru/codes/disassembly/object-detection/dataset/train/labels/00000.txt"
# black_img = np.zeros((img_height, img_width), dtype=np.uint8)


# with open(txt_file, "r") as file:
#     lines = file.readlines()

# # Iterate over each line to extract contours
# for line in lines:
#     data = line.strip().split()
    
#     # First element is the mask ID, ignore it for now
#     mask_id = int(data[0])
    
#     # The rest of the data are normalized coordinates (x1, y1, x2, y2, ...)
#     points = np.array(data[1:], dtype=float).reshape(-1, 2)
    
#     # Denormalize the points back to the original image dimensions
#     points[:, 0] *= img_width  # Scale x-coordinates
#     points[:, 1] *= img_height  # Scale y-coordinates
    
#     # Convert points to integer coordinates
#     points = points.astype(int)
    
#     # Draw the contour on the black image
#     points = points.reshape((-1, 1, 2))  # Reshape for cv2.drawContours
#     cv2.drawContours(black_img, [points], -1, (255), thickness=1)  # Draw contour in white

# # Display the image with contours
# plt.imshow(black_img, cmap="gray")
# plt.title("Contours")
# plt.show()