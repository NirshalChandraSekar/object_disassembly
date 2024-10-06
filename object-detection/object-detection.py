import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import random

class generate_dataset:
    def __init__(self):
        #create the dataset folder
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
            os.mkdir("dataset/train")
            os.mkdir("dataset/train/images")
            os.mkdir("dataset/train/labels")
            os.mkdir("dataset/val")
            os.mkdir("dataset/val/images")
            os.mkdir("dataset/val/labels")

        self.frames = sorted(os.listdir("../video-segmentation/video-frames"))
        self.train_split = int(len(self.frames) * 0.8)
        self.val_split = int(len(self.frames) * 0.2)
        self.train_frames = self.frames[:self.train_split]
        self.val_frames = self.frames[self.train_split:]

    def create_images(self):

        for frame in self.train_frames:
            img = cv2.imread("../video-segmentation/video-frames/" + frame)
            cv2.imwrite("dataset/train/images/" + frame, img)

        for frame in self.val_frames:
            img = cv2.imread("../video-segmentation/video-frames/" + frame)
            cv2.imwrite("dataset/val/images/" + frame, img)

    def write_labels(self, mask_dict_split, output_dir):
        for frame in mask_dict_split:
            with open(output_dir + frame.replace(".jpeg", ".txt"), "w") as f:
                write_buffer = []  

                for mask_id, mask in mask_dict_split[frame].items():
                    
                    mask = mask.astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    
                    contours = [cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True) for contour in contours]

                    
                    for i, contour in enumerate(contours):
                        points_str = " ".join([f"{point[0][0]/mask.shape[1]:.6f} {point[0][1]/mask.shape[0]:.6f}" for point in contour])
                        write_buffer.append(f"{mask_id} {points_str}\n")

                
                f.writelines(write_buffer)

    def create_labels(self, mask_dict):
        keys = list(mask_dict.keys())
        train_keys = keys[:self.train_split]
        val_keys = keys[self.train_split:]

        mask_dict_train_split = {key: mask_dict[key] for key in train_keys}
        mask_dict_val_split = {key: mask_dict[key] for key in val_keys}

        
        self.write_labels(mask_dict_train_split, "dataset/train/labels/")
        self.write_labels(mask_dict_val_split, "dataset/val/labels/")

    def create_yaml(self):
        # Define the paths
        base_path = "../dataset"
        train_path = "train"
        val_path = "val"

        # Read the first text file in the train split to get mask IDs
        first_text_file = sorted(os.listdir("dataset/train/labels"))[0]
        mask_id_to_name = {}

        with open(os.path.join("dataset/train/labels", first_text_file), "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                mask_id = parts[0]  # This is the mask ID
                mask_name = f"object_{mask_id}"  # Construct the object name
                mask_id_to_name[int(mask_id)] = mask_name  # Store the mask ID and name

        # Write the YAML file
        with open("dataset/dataset.yaml", "w") as f:
            f.write(f"path: {base_path}\n")
            f.write(f"train: {train_path}\n")
            f.write(f"val: {val_path}\n")
            
            # Write names
            f.write("names:\n")
            for mask_id, name in mask_id_to_name.items():
                f.write(f"  {mask_id}: {name}\n")

    def main(self, mask_dict):
        self.create_images()
        self.create_labels(mask_dict)
        self.create_yaml()


class detect_parts:
    def __init__(self):
        self.model = YOLO("yolo11n-seg.pt")
        self.results = None

    def train(self):
        self.results = self.model.train(data="dataset/dataset.yaml", epochs=50, batch=40, device=0)

    def detect(self, img_paths):
        model = YOLO("runs/segment/train/weights/best.pt")
        
        results_dir = "detection_results"
        os.makedirs(results_dir, exist_ok=True)

        # Loop through each image path in the list
        for img_path in img_paths:
            # Perform inference on the image
            results = model(img_path)

            # Extract base image name for unique saving
            img_name = os.path.basename(img_path).split('.')[0]  # Extract base image name

            # Process results list
            for idx, result in enumerate(results):
                boxes = result.boxes  # Bounding boxes
                masks = result.masks  # Segmentation masks
                keypoints = result.keypoints  # Pose keypoints
                probs = result.probs  # Classification probabilities
                obb = result.obb  # Oriented bounding boxes

                # Display results
                result.show()

                # Generate a unique filename for each result
                save_path = os.path.join(results_dir, f"{img_name}_result_{idx}.jpg")

                # Save result to disk with a unique name
                result.save(filename=save_path)

                print(f"Saved result for image '{img_name}' as {save_path}")
        



if __name__ == "__main__":
    
    mask_dict = np.load("tracked_masks.npy", allow_pickle=True).item()
    
    print("generating dataset")
    dataset = generate_dataset()
    dataset.main(mask_dict)

    print("training model")
    predictor = detect_parts()
    predictor.train()

    print("detecting parts")
    predictor.detect(["dataset/test_images/IMG_0319.jpg", 
                      "dataset/test_images/IMG_0320.jpg", 
                      "dataset/test_images/IMG_0321.jpg",
                      "dataset/test_images/IMG_0322.jpg",
                      "dataset/test_images/IMG_0323.jpg",
                      "dataset/test_images/IMG_0324.jpg",
                      "dataset/test_images/IMG_0325.jpg",
                      "dataset/test_images/IMG_0326.jpg",])

