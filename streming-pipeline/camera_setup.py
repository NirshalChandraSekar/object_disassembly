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
        align_to = rs.stream.color
        align = rs.align(align_to)

        frames = self.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_profile = rs.video_stream_profile(depth_frame.get_profile())
        depth_intrinsics = depth_profile.get_intrinsics()
        k_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],[0, depth_intrinsics.fy, depth_intrinsics.ppy],[0, 0, 1]])

        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        points_data = np.asarray(points.get_vertices())

        return color_image, depth_image, points_data, k_matrix
    
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


if __name__ == "__main__":
    stream = stream_realsense()
    color_image, depth_image, points_data, k_matrix = stream.image_stream()
    depth_image = depth_image
    print("Color Image Shape: ", color_image.shape)
    print("Depth Image Shape: ", depth_image.shape)
    print("Points Data Shape: ", points_data.shape)
    print("K Matrix: ", k_matrix)
    print("depth_image", depth_image)

    cv2.imshow('Color Image', color_image)
    cv2.imshow('Depth Image', depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    np.save("color_image.npy", color_image)
    np.save("depth_image.npy", depth_image)
    np.save("points_data.npy", points_data)
    np.save("k_matrix.npy", k_matrix)

