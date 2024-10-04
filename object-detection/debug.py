import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# tracked_points = np.load("tracked_masks.npy", allow_pickle=True).item()
# print(tracked_points)


# read the video frames folder and split the images into a seperate dataset folder with 80:20 train test split

frame_files = sorted(os.listdir("../video-segmentation/video-frames"))
print("length of frame_files", len(frame_files))
train_split = int(len(frame_files) * 0.8)
print("train_split", train_split)