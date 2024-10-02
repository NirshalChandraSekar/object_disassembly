import torch
import numpy as np
import os
import matplotlib.pyplot
import cv2
import gc

from sam2.build_sam import build_sam2 
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

import supervision as sv

class automatic_video_segmentation:
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
            self.model = None
            self.inference_state = None

    def split_video_frames(self):
         
        print("spliting the video frames")
        frames_generator = sv.get_video_frames_generator(self.input_video_path)
        sink = sv.ImageSink(target_dir_path="video-frames",
                        image_name_pattern="{:05d}.jpeg")
        with sink:
            for frame in frames_generator:
                sink.save_image(frame)

        frame_files = sorted(os.listdir("video-frames"))
        first_frame = cv2.imread(os.path.join("video-frames", frame_files[0]))

        return first_frame
    
    def automatic_mask_generation(self, first_frame):
         
        print("automatic_mask_generation")
        self.model = build_sam2(self.config, self.checkpoint, device = self.device)
        print("model loaded")
        predictor = SAM2AutomaticMaskGenerator(self.model, points_per_side=8)
        masks = predictor.generate(first_frame)

        return masks
    
    def get_point_promts(self):
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

    def add_masks_to_model(self, masks):
        for i in range(len(masks)):
            _, _, _ = self.model.add_new_mask(inference_state = self.inference_state,
                                               frame_idx = 0,
                                               obj_id = i,
                                               mask = masks[i]['segmentation'])
    
    def add_points_to_model(self, points):
        for i in range(len(points)):
            _, _, _ = self.model.add_new_points_or_box(inference_state = self.inference_state,
                                               frame_idx = 0,
                                               obj_id = i,
                                               points = [points[i]],
                                               labels = np.array([1]))

    def track_masks_in_frames(self, masks=None, points=None):

        self.model = build_sam2_video_predictor(self.config, self.checkpoint, device = self.device)
        print("model updated")

        self.inference_state = self.model.init_state("video-frames")
        self.model.reset_state(self.inference_state)

        if masks is not None:
            self.add_masks_to_model(masks)
            print("masks added to model")

        if points is not None:
            self.add_points_to_model(points)
            print("points added to model")

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


    
    def main(self, _automatic = True):
        
        first_frame = self.split_video_frames()
        if _automatic:
            masks = self.automatic_mask_generation(first_frame)
            print("masks generated")

            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

            self.track_masks_in_frames(masks=masks)

        else:
            points = self.get_point_promts()
            self.track_masks_in_frames(points = points)



if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    input_video_path = "/home/niru/codes/disassembly/video-segmentation/input_video.mp4"
    output_video_path = "/home/niru/codes/disassembly/video-segmentation/final_output.mp4"
    checkpoint_path = "/home/niru/codes/disassembly/video-segmentation/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
    config = "sam2_hiera_t.yaml"
    
    video_segmentation = automatic_video_segmentation(input_video_path, output_video_path, checkpoint_path, config)
    video_segmentation.main(_automatic = False)
