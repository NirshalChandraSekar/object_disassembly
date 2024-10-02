import torch
import numpy as np
import os
import matplotlib.pyplot
import cv2

from sam2.build_sam import build_sam2 
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

import supervision as sv

class video_segmentation:
    def __init__(self, _input_video_path, 
                 _output_video_path,
                 _checkpoint,
                 _config):
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("device set to cuda")
        else:
            self.device = torch.device("cpu")
        
        self.input_video_path = _input_video_path
        self.output_video_path = _output_video_path
        self.checkpoint = _checkpoint
        self.config = _config
        self.model = build_sam2_video_predictor(self.config,
                                                self.checkpoint,
                                                device = self.device)
        
        self.inference_state = None

    def get_input_prompts(self):
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                points.append(np.array([x, y]))
                cv2.imshow('image', img)
        # get the input prompts from the user
        points = []
        img = cv2.imread("video-frames/00000.jpeg")
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        points = np.array(points)
        return points

        
    def set_inference_state(self):

        frames_generator = sv.get_video_frames_generator(self.input_video_path)
        sink = sv.ImageSink(target_dir_path="video-frames",
                            image_name_pattern="{:05d}.jpeg")
        with sink:
            for frame in frames_generator:
                sink.save_image(frame)

        self.inference_state = self.model.init_state("video-frames")
        self.model.reset_state(self.inference_state)
        

    def add_points_to_segmentor(self, point, label, tracker_id):
        _, _, _ = self.model.add_new_points_or_box(inference_state = self.inference_state,
                                                   frame_idx = 0,
                                                   obj_id = tracker_id,
                                                   points = [point],
                                                   labels = label)
        
    def generate(self):
        self.set_inference_state()
        points = self.get_input_prompts()
        for i in range(len(points)):
            self.add_points_to_segmentor(points[i], np.array([1]), i)
        
        colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
        mask_annotator = sv.MaskAnnotator(
            color=sv.ColorPalette.from_hex(colors),
            color_lookup=sv.ColorLookup.TRACK)

        video_info = sv.VideoInfo.from_video_path(self.input_video_path)
        frames_paths = sorted(sv.list_files_with_extensions(
            directory="video-frames", 
            extensions=["jpeg"]))
        
        with sv.VideoSink(self.output_video_path, video_info=video_info) as sink:
            for frame_idx, object_ids, mask_logits in self.model.propagate_in_video(self.inference_state):
                frame = cv2.imread(frames_paths[frame_idx])
                masks = (mask_logits > 0.0).cpu().numpy()
                N, X, H, W = masks.shape
                masks = masks.reshape(N * X, H, W)
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks),
                    mask=masks,
                    tracker_id=np.array(object_ids)
                )
                frame = mask_annotator.annotate(frame, detections)
                sink.write_frame(frame)


if __name__ == "__main__":
    input_video_path = "/home/niru/codes/disassembly/video-segmentation/input_video.mp4"
    output_video_path = "/home/niru/codes/disassembly/video-segmentation/final_output_prompted.mp4"
    checkpoint_path = "/home/niru/codes/disassembly/video-segmentation/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
    config = "sam2_hiera_t.yaml"

    object = video_segmentation(input_video_path,
                                output_video_path,
                                checkpoint_path,
                                config)
    
    object.generate()