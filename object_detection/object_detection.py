"""
Author: Nirshal Chandra Sekar
Description:
This script generates datasets from video frames, creates labels from segmentation masks, 
and trains a YOLO model to detect objects. It includes functionality for dataset generation, 
model training, and inference.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import random


class generate_dataset:
    """
    Class to generate a dataset for YOLO training from video frames and segmentation masks. 
    It organizes video frames into training and validation sets, creates corresponding label 
    files, and generates a YAML file required by the YOLO model for training.
    """
    def __init__(self):
        """
        Initializes the dataset generator by creating required directories for training and 
        validation images and labels. Splits the available frames into training (80%) 
        and validation (20%) sets.
        """
        # Create dataset folder structure if it doesn't exist
        if not os.path.exists("object_detection/data/dataset"):
            os.mkdir("object_detection/data/dataset")
            os.mkdir("object_detection/data/dataset/train")
            os.mkdir("object_detection/data/dataset/train/images")
            os.mkdir("object_detection/data/dataset/train/labels")
            os.mkdir("object_detection/data/dataset/val")
            os.mkdir("object_detection/data/dataset/val/images")
            os.mkdir("object_detection/data/dataset/val/labels")

        # Load and sort video frames, then split into training and validation
        self.frames = sorted(os.listdir("video_segmentation/data/video_frames"))
        self.train_split = int(len(self.frames) * 0.8)
        self.val_split = int(len(self.frames) * 0.2)
        self.train_frames = self.frames[:self.train_split]
        self.val_frames = self.frames[self.train_split:]

    def create_images(self):
        """
        Copies frames from the video segmentation directory to the dataset directories, 
        separating them into training and validation folders.
        """
        # Save training images
        for frame in self.train_frames:
            img = cv2.imread("video_segmentation/data/video_frames/" + frame)
            cv2.imwrite("object_detection/data/dataset/train/images/" + frame, img)

        # Save validation images
        for frame in self.val_frames:
            img = cv2.imread("video_segmentation/data/video_frames/" + frame)
            cv2.imwrite("object_detection/data/dataset/val/images/" + frame, img)

    def write_labels(self, mask_dict_split, output_dir):
        """
        Generates label files based on segmentation masks.
        
        Args:
            mask_dict_split (dict): A dictionary containing frame names and their associated masks.
            output_dir (str): The directory where the label files should be written.
        """
        for frame in mask_dict_split:
            with open(output_dir + frame.replace(".jpeg", ".txt"), "w") as f:
                write_buffer = []  

                # Extract contours from each mask in the frame
                for mask_id, mask in mask_dict_split[frame].items():
                    mask = mask.astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = [cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True) for contour in contours]

                    # Convert contour points to normalized coordinates
                    for i, contour in enumerate(contours):
                        points_str = " ".join([f"{point[0][0]/mask.shape[1]:.6f} {point[0][1]/mask.shape[0]:.6f}" for point in contour])
                        write_buffer.append(f"{mask_id} {points_str}\n")

                # Save label data to the output file
                f.writelines(write_buffer)

    def create_labels(self, mask_dict):
        """
        Splits the masks into training and validation sets, and writes label files for each set.
        
        Args:
            mask_dict (dict): Dictionary containing segmentation masks for each frame.
        """
        keys = list(mask_dict.keys())
        train_keys = keys[:self.train_split]
        val_keys = keys[self.train_split:]

        # Create separate training and validation splits of the mask dictionary
        mask_dict_train_split = {key: mask_dict[key] for key in train_keys}
        mask_dict_val_split = {key: mask_dict[key] for key in val_keys}

        # Write label files for training and validation sets
        self.write_labels(mask_dict_train_split, "object_detection/data/dataset/train/labels/")
        self.write_labels(mask_dict_val_split, "object_detection/data/dataset/val/labels/")

    def create_yaml(self):
        """
        Generates a YAML configuration file required by YOLO for specifying dataset paths 
        and class names.
        """
        base_path = "object_detection/data/dataset"
        train_path = "train"
        val_path = "val"

        # Read the first training label file to determine class names
        first_text_file = sorted(os.listdir("object_detection/data/dataset/train/labels"))[0]
        mask_id_to_name = {}

        with open(os.path.join("object_detection/data/dataset/train/labels", first_text_file), "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                mask_id = parts[0]
                mask_name = f"object_{mask_id}"
                mask_id_to_name[int(mask_id)] = mask_name

        # Write the YAML file
        with open("object_detection/data/dataset/dataset.yaml", "w") as f:
            f.write(f"path: dataset\n")
            f.write(f"train: {train_path}\n")
            f.write(f"val: {val_path}\n")
            f.write("names:\n")
            for mask_id, name in mask_id_to_name.items():
                f.write(f"  {mask_id}: {name}\n")

    def main(self, mask_dict):
        """
        Main function to create the dataset, including images, labels, and YAML configuration.
        
        Args:
            mask_dict (dict): Dictionary containing segmentation masks for each frame.
        """
        self.create_images()
        self.create_labels(mask_dict)
        self.create_yaml()


class detect_parts:
    """
    Class for training a YOLO model and performing object detection on test images.
    """
    def __init__(self):
        """
        Initializes the YOLO model for training and detection.
        """
        self.model = YOLO("yolo11s-seg.pt")
        self.results = None

    def train(self):
        """
        Trains the YOLO model using the dataset generated from video frames and masks.
        """
        self.results = self.model.train(data="object_detection/data/dataset/dataset.yaml", 
                                        epochs=100, 
                                        batch=0.90, 
                                        device=0,
                                        copy_paste=0.5,
                                        degrees = 15,
                                        flipud = 0.5,
                                        )

    def detect(self, img_path):
        """
        Performs object detection on a list of test images using the trained YOLO model.
        
        Args:
            img_paths (list): List of file paths for the images to be processed.
        """
        model = YOLO("object_detection/data/runs/segment/train/weights/best.pt")

        # original_image = cv2.imread(img_path)
        original_image = img_path
        height, width = original_image.shape[:2]
        
        results = model.predict(img_path, conf=0.8, device=0)
        combined_mask = np.zeros((height, width), dtype=original_image.dtype)

        for i in range(len(results[0].masks)):
            mask = results[0].masks[i]
            mask = np.asanyarray(mask.data.cpu().squeeze(), dtype=original_image.dtype)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)
            mask *= i + 1
            combined_mask += mask

        save_mask = combined_mask * 255 / (i+1)
        cv2.imwrite("object_detection/data/combined_mask.jpg", save_mask)

        return combined_mask
        


if __name__ == "__main__":
    # Load pre-saved mask data for dataset generation
    # mask_dict = np.load("tracked_masks.npy", allow_pickle=True).item()
    
    # Uncomment the following to generate a dataset
    # dataset = generate_dataset()
    # dataset.main(mask_dict)

    # Create an instance of the detection class and perform detection
    predictor = detect_parts()
    # predictor.train()

    # print("detecting parts")
    image = np.load("/home/rpmdt05/Code/Niru/object_disassembly/streming-pipeline/color_image.npy")
    print("type of image", type(image))
    cv2.imshow('Color Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    combined_mask = predictor.detect(image)
    np.save("object_detection/data/combined_mask.npy", combined_mask)
