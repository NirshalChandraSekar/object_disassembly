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
        base_path = "dataset"
        train_path = os.path.join(base_path, "train")
        val_path = os.path.join(base_path, "valid")

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


if __name__ == "__main__":
    mask_dict = np.load("tracked_masks.npy", allow_pickle=True).item()
    dataset = generate_dataset()
    dataset.main(mask_dict)