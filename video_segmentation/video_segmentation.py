"""
Author: Nirshal Chandra Sekar
Description:
This script performs automatic video segmentation using the SAM2 model (Segment Anything Model 2).
It allows for automatic mask generation and point-based manual prompts for video object segmentation. 
The segmented masks are then propagated throughout the video frames, and the segmented output is saved as a video.
"""

# Import necessary libraries
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import gc

# Import SAM2 model components for building the video predictor and mask generator
from sam2.build_sam import build_sam2 
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import supervision utilities for video frame processing and annotations
import supervision as sv

class automatic_video_segmentation:
    """
    A class for performing automatic video segmentation using SAM2.

    Attributes:
        input_video_path (str): Path to the input video.
        output_video_path (str): Path to the output segmented video.
        checkpoint (str): Path to the model checkpoint for loading weights.
        config (str): Configuration file path for the model.
        device (torch.device): Device to be used (CPU or GPU).
        model (SAM2 model): Placeholder for the loaded SAM2 model.
        inference_state: Inference state for video propagation.
    """
    def __init__(self, _input_video_path, 
                 _output_video_path,
                 _checkpoint,
                 _config):
        """
        Initialize the video segmentation object.

        Parameters:
            _input_video_path (str): Path to the input video.
            _output_video_path (str): Path to save the output segmented video.
            _checkpoint (str): Path to SAM2 model checkpoint.
            _config (str): Path to SAM2 configuration file.
        """

        # Check for CUDA availability and set the device accordingly
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("device set to cuda")
        else:
            self.device = torch.device("cpu")
        
        # Initialize paths and model properties
        self.input_video_path = _input_video_path
        self.output_video_path = _output_video_path
        self.checkpoint = _checkpoint
        self.config = _config
        self.model = None
        self.inference_state = None

    def split_video_frames(self):
        """
        Split the video into individual frames and save them.

        Returns:
            first_frame (numpy.ndarray): The first frame of the video.
        """
        print("splitting the video frames")
        frames_generator = sv.get_video_frames_generator(self.input_video_path)
        
        # Sink to save the individual frames as JPEG files
        sink = sv.ImageSink(target_dir_path="video_segmentation/data/video_frames",
                            image_name_pattern="{:05d}.jpeg")
        with sink:
            for frame in frames_generator:
                sink.save_image(frame)

        # Retrieve the first frame for mask generation
        frame_files = sorted(os.listdir("video_segmentation/data/video_frames"))
        first_frame = cv2.imread(os.path.join("video_segmentation/data/video_frames", frame_files[0]))

        return first_frame
    
    def automatic_mask_generation(self, first_frame):
        """
        Generate masks automatically using the SAM2 model.

        This method initializes the SAM2 model, generates segmentation masks from the 
        provided first frame, and removes the mask whose area is closest to the area 
        of the original image size.

        Parameters:
            first_frame (numpy.ndarray): The first frame of the video as a NumPy array.

        Returns:
            list: A list of generated masks after removing the closest one to the original 
                image size.
        """
        print("automatic_mask_generation")
        
        # Build and load the SAM2 model
        self.model = build_sam2(self.config, self.checkpoint, device=self.device)
        print("model loaded")
        
        # Use the automatic mask generator to generate masks on the first frame
        predictor = SAM2AutomaticMaskGenerator(self.model, points_per_side=10, box_nms_thresh=0.5)
        masks = predictor.generate(first_frame)

        areas = []  

        image_height, image_width = masks[0]["segmentation"].shape
        original_image_size = image_height * image_width

        for i in range(len(masks)):
            mask = masks[i]["segmentation"]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True) for contour in contours]

            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)

            areas.append(contour_area)

        closest_idx = np.argmin([abs(area - original_image_size) for area in areas])
        masks.pop(closest_idx)

        # save the masks as different images
        
        for i in range(len(masks)):
            mask = masks[i]["segmentation"]
            cv2.imwrite(f"video_segmentation/data/mask_{i}.jpeg", mask.astype(np.uint8) * 255)

        return masks

    
    def get_point_promts(self):
        """
        Capture user input prompts (points) for manual mask generation.

        Returns:
            points (numpy.ndarray): Array of user-selected points on the frame.
        """
        def click_event(event, x, y, flags, param):
            """Handle mouse click events to capture points on the frame."""
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                points.append(np.array([x, y]))
                cv2.imshow('image', img)

        # Get manual points from user
        points = []
        img = cv2.imread("video_segmentation/data/video_frames/00000.jpeg")
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        points = np.array(points)
        return points

    def add_masks_to_model(self, masks):
        """
        Add generated masks to the model.

        Parameters:
            masks (list): List of segmentation masks to be added.
        """
        for i in range(len(masks)):
            _, _, _ = self.model.add_new_mask(inference_state=self.inference_state,
                                              frame_idx=0,
                                              obj_id=i,
                                              mask=masks[i]['segmentation'])
    
    def add_points_to_model(self, points):
        """
        Add user-provided points to the model.

        Parameters:
            points (list): List of points selected by the user.
        """
        for i in range(len(points)):
            _, _, _ = self.model.add_new_points_or_box(inference_state=self.inference_state,
                                                       frame_idx=0,
                                                       obj_id=i,
                                                       points=[points[i]],
                                                       labels=np.array([1]))

    def track_masks_in_frames(self, masks=None, points=None):
        """
        Track masks or points across video frames.

        Parameters:
            masks (list, optional): List of masks to track.
            points (list, optional): List of points to track.

        Returns:
            tracked_masks (dict): Dictionary of tracked masks across frames.
        """
        # Build the video predictor model for tracking
        self.model = build_sam2_video_predictor(self.config, self.checkpoint, device=self.device)
        print("model updated")

        # Initialize inference state for video propagation
        self.inference_state = self.model.init_state("video_segmentation/data/video_frames", offload_video_to_cpu=True)
        self.model.reset_state(self.inference_state)

        # Add masks or points to the model for tracking
        if masks is not None:
            self.add_masks_to_model(masks)
            print("masks added to model")
        if points is not None:
            self.add_points_to_model(points)
            print("points added to model")

        # Set up mask annotation with colors for visualization
        colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
        mask_annotator = sv.MaskAnnotator(
            color=sv.ColorPalette.from_hex(colors),
            color_lookup=sv.ColorLookup.TRACK)

        # Get video info and sorted list of frame paths
        video_info = sv.VideoInfo.from_video_path(self.input_video_path)
        frames_paths = sorted(sv.list_files_with_extensions(
            directory="video_segmentation/data/video_frames", 
            extensions=["jpeg"]))
        
        tracked_masks = {}
        
        # Propagate masks across frames and save the segmented video
        with sv.VideoSink(self.output_video_path, video_info=video_info) as sink:
            for frame_idx, object_ids, mask_logits in self.model.propagate_in_video(self.inference_state):
                frame_path = frames_paths[frame_idx]
                frame = cv2.imread(frame_path)
                masks = (mask_logits > 0.0).cpu().numpy()
                combined_masks = masks.any(axis=1)
                tracked_masks[os.path.basename(frame_path)] = {
                    object_label: combined_masks[i]
                    for i, object_label in enumerate(object_ids)
                }

                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=combined_masks),
                    mask=combined_masks,
                    tracker_id=np.array(object_ids)
                )
                frame = mask_annotator.annotate(frame, detections)
                sink.write_frame(frame)

        return tracked_masks

    def main(self, _automatic=True):
        """
        Main function to perform video segmentation.
        
        Parameters:
            _automatic (bool): Whether to use automatic or point-based segmentation.

        Returns:
            tracked_masks (dict): Dictionary of tracked masks across frames.
        """
        # Split video into frames and get the first frame
        first_frame = self.split_video_frames()
        
        # Automatic mask generation or manual point-based prompts
        if _automatic:
            masks = self.automatic_mask_generation(first_frame)
            print("masks generated")

            # Free up GPU memory
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

            # Track masks across frames
            tracked_masks = self.track_masks_in_frames(masks=masks)
        else:
            points = self.get_point_promts()
            tracked_masks = self.track_masks_in_frames(points=points)

        return tracked_masks


if __name__ == "__main__":
    # Clean up CUDA memory
    torch.cuda.empty_cache()
    gc.collect()

    # Define paths and model parameters
    input_video_path = "video-segmentation/data/input_video.mp4"
    output_video_path = "video-segmentation/data/final_output.mp4"
    checkpoint_path = "sam2/checkpoints/sam2_hiera_small.pt"
    config = "sam2_hiera_s.yaml"
    
    # Instantiate and run the video segmentation class
    video_segmentation = automatic_video_segmentation(input_video_path, output_video_path, checkpoint_path, config)
    tracked_masks = video_segmentation.main(_automatic=True)

    # Save the tracked masks
    np.save("../object-detection/tracked_masks", tracked_masks)
