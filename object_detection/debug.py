import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
# Load the YOLO model
np.set_printoptions(threshold=np.inf)

model = YOLO("runs/segment/train/weights/best.pt")

# Run inference on the image
results = model.predict("test_images/IMG_0325.jpg", conf=0.8, device=0)

# Read the original image
original_image = cv2.imread("test_images/IMG_0325.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

original_height, original_width = original_image.shape[:2]

print(original_image.dtype)

combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)

#give numbe 1 for first mask 2 for second mask and so on in thhe combined mask

for i in range(len(results[0].masks)):
    mask = results[0].masks[i]
    mask = np.asanyarray(mask.data.cpu().squeeze(), dtype=np.uint8)
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    mask *= i + 1
    combined_mask += mask

print(original_image.shape)
print(combined_mask.shape)
print(combined_mask[3000])

plt.imshow(combined_mask)
plt.show()