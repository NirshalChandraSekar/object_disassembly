from video_segmentation.video_segmentation import automatic_video_segmentation
from streaming_pipeline.camera_setup import stream_realsense
from object_detection.object_detection import detect_parts, generate_dataset
from grasping.contact_graspnet_pytorch.inference import CGN

import numpy as np
import cv2
import torch
import gc


if __name__ == "__main__":

    torch.cuda.empty_cache()
    gc.collect()


    """
    VIDEO SEGMENTATION
    """

    video_segmentation = automatic_video_segmentation(
        _input_video_path="video-segmentation/data/input_video.mp4",
        _output_video_path="video-segmentation/data/output_video.mp4",
        _checkpoint="video-segmentation/segment-anything-2/checkpoints/sam2_hiera_small.pt",
        _config="sam2_config.yaml"
    )

    
    tracked_masks = video_segmentation.main(_automatic=True)
    
    np.save("object-detection/data/tracked_masks", tracked_masks)

    input("Press ENTER to start image streaming")


    """
    IMAGE STREAMING
    """
    stream = stream_realsense()
    color_image, depth_image, points_data, k_matrix = stream.image_stream()
    depth_image = depth_image*0.01 # scaling factor for the D405 realsense camera
    
    cv2.imshow('Color Image', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    np.save("streaming_pipeline/data/color_image.npy", color_image)
    np.save("streaming_pipeline/data/depth_image.npy", depth_image)
    np.save("streaming_pipeline/data/points_data.npy", points_data)
    np.save("streaming_pipeline/data/k_matrix.npy", k_matrix)

    input("Press ENTER to start object detection")


    """
    MODEL TRAINING AND OBJECT DETECTION
    """
    # GENERATE THE DATASET
    dataset = generate_dataset()
    dataset.main(tracked_masks)

    # TRAIN THE MODEL
    predictor = detect_parts()
    predictor.train()

    # DETECT PARTS
    combined_mask = predictor.detect(color_image)
    np.save("combined_mask.npy", combined_mask)

    
    """
    GRASP PLANNING
    """
    input_for_cgn = {
                     'rgb': color_image,
                     'depth': depth_image,
                     'K': k_matrix,
                     'seg': combined_mask
                    }

    np.save("input_for_cgn.npy", input_for_cgn)
    cgn = CGN(input_path="input_for_cgn.npy", K=k_matrix, z_range = [0,50], visualize=True, forward_passes=2)

    pred_grasps, grasp_scores, contact_pts, gripper_openings = cgn.inference()