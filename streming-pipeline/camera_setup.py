import pyrealsense2 as rs
import numpy as np
import cv2

class stream_realsense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

    def image_stream(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image
    
    # def video_stream(self):
    #     while True:
    #         frames = self.pipeline.wait_for_frames()
    #         depth_frame = frames.get_depth_frame()
    #         color_frame = frames.get_color_frame()
    #         depth_image = np.asanyarray(depth_frame.get_data())
    #         color_image = np.asanyarray(color_frame.get_data())
    #         cv2.imshow('RealSense', color_image)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     cv2.destroyAllWindows()
    #     self.pipeline.stop()