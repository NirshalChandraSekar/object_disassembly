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
        
        self.input_video_path = _input_video_path
        self.output_video_path = _output_video_path
        self.checkpoint = _checkpoint
        self.config = _config
        self.model = build_sam2_video_predictor(self.config,
                                                self.checkpoint,
                                                device = self.device)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def split_into_frames(self):

        frames_generator = sv.get_video_frames_generator(self.input_video_path)
        sink = sv.ImageSink(target_dir_path="video-frames",
                            image_name_pattern="{:d}.jpeg")
        with sink:
            for frame in frames_generator:
                sink.save_image(frame)

    

