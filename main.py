from video_segmentation.video_segmentation import automatic_video_segmentation
from streaming_pipeline.camera_setup import stream_realsense
from object_detection.object_detection import detect_parts, generate_dataset
from grasping.contact_graspnet_pytorch import inference
from grasping.contact_graspnet_pytorch import config_utils
import hydra
import numpy as np
import cv2
import torch
import os
import gc



if __name__ == "__main__":

    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize_config_module('video_segmentation/sam2/sam2/configs/sam2', version_base='1.2')

    torch.cuda.empty_cache()
    gc.collect()


    """
    VIDEO SEGMENTATION
    """

    # video_segmentation = automatic_video_segmentation(
    #     _input_video_path="video_segmentation/data/input_video.mov",
    #     _output_video_path="video_segmentation/data/output_video.mp4",
    #     _checkpoint="video_segmentation/sam2/checkpoints/sam2.1_hiera_small.pt",
    #     _config="configs/sam2.1/sam2.1_hiera_s.yaml"
    # )

    
    # tracked_masks = video_segmentation.main(_automatic=True)
    
    # np.save("object_detection/data/tracked_masks", tracked_masks)

    # input("Press ENTER to start image streaming")


    """
    MODEL TRAINING
    """

    # tracked_masks = np.load("object_detection/data/tracked_masks.npy", allow_pickle=True).item()

    # # GENERATE THE DATASET
    # dataset = generate_dataset()
    # dataset.main(tracked_masks)

    # # TRAIN THE MODEL
    predictor = detect_parts()
    # predictor.train()


    """
    IMAGE STREAMING
    """
    # stream = stream_realsense()
    # color_image, depth_image, points_data, k_matrix = stream.image_stream()
    # depth_image = depth_image*0.00025 # scaling factor for the D405 realsense camera
    
    # cv2.imshow('Color Image', color_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite("streaming_pipeline/data/color_image.jpg", color_image)
    # np.save("streaming_pipeline/data/color_image.npy", color_image)
    # np.save("streaming_pipeline/data/depth_image.npy", depth_image)
    # np.save("streaming_pipeline/data/points_data.npy", points_data)
    # np.save("streaming_pipeline/data/k_matrix.npy", k_matrix)

    # input("Press ENTER to start object detection")


    """
    OBJECT DETECTION
    """

    # DETECT PARTS
    color_image = np.load("streaming_pipeline/data/color_image.npy")
    # combined_mask = predictor.detect(color_image)
    # np.save("object_detection/data/combined_mask.npy", combined_mask)

    
    """
    GRASP PLANNING
    """

    depth_image = np.load("streaming_pipeline/data/depth_image.npy")
    depth_image = depth_image.astype(np.float32)  # Convert to float32
    # depth_image *= 0.00025  
    k_matrix = np.load("streaming_pipeline/data/k_matrix.npy")
    combined_mask = np.load("object_detection/data/combined_mask.npy")
    input_for_cgn = {
                     'rgb': color_image,
                     'depth': depth_image,
                     'K': k_matrix,
                     'seg': combined_mask
                    }

    np.save("grasping/input_for_cgn.npy", input_for_cgn)

   

    global_config = config_utils.load_config("grasping/checkpoints/contact_graspnet", batch_size=3, arg_configs=[])
    # print(str(global_config))
    # print('pid: %s'%(str(os.getpid())))
    pred_grasps_cam, scores, contact_pts = inference.inference(global_config,
                        "grasping/checkpoints/contact_graspnet",
                        "grasping/input_for_cgn.npy",  
                        # "/home/niru/Downloads/0.npy",
                        local_regions=True,
                        filter_grasps=True,
                        z_range=eval(str([0,10])),
                        forward_passes=2,
                        K=k_matrix,

                        )
    
    print("pred_grasps cam length: ", len(pred_grasps_cam))
    print("scores length: ", len(scores))
    print("contact_pts length: ", len(contact_pts))

    np.save("grasping/data/pred_grasps_cam.npy", pred_grasps_cam)
    np.save("grasping/data/scores.npy", scores)
    np.save("grasping/data/contact_pts.npy", contact_pts)
    