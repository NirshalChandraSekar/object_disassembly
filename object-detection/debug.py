import numpy as np
import cv2
import matplotlib.pyplot as plt


tracked_points = np.load("tracked_masks.npy", allow_pickle=True).item()
print(tracked_points[0])