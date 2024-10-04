import numpy as np
import cv2
import os


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

    def create_images(self):
        frames = sorted(os.listdir("../video-segmentation/video-frames"))
        train_frames = frames[:int(len(frames) * 0.8)]
        val_frames = frames[int(len(frames) * 0.8):]

        for frame in train_frames:
            img = cv2.imread("../video-segmentation/video-frames/" + frame)
            cv2.imwrite("dataset/train/images/" + frame, img)

        for frame in val_frames:
            img = cv2.imread("../video-segmentation/video-frames/" + frame)
            cv2.imwrite("dataset/val/images/" + frame, img)

    def create_labels(self):
        pass

    
    def main(self):
        self.create_images()
        self.create_labels()


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.main()