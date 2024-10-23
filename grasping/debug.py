from contact_graspnet_pytorch.inference import CGN
import numpy as np
import cv2
import open3d as o3d

#
np.set_printoptions(threshold=np.inf)

# combined_mask = np.load("/home/niru/codes/disassembly/object_detection/data/combined_mask.npy")
rgb_image = np.load("/home/niru/codes/disassembly/streaming_pipeline/data/color_image.npy")
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
depth_image = np.load("/home/niru/codes/disassembly/streaming_pipeline/data/depth_image.npy")
k_matrix = np.load("/home/niru/codes/disassembly/streaming_pipeline/data/k_matrix.npy")

points_data = np.load("/home/niru/codes/disassembly/streaming_pipeline/data/points_data.npy")
print("Points Data Shape: ", points_data.shape)

depth_image = depth_image.astype(np.float32)  # Convert to float32
depth_image *= 0.00025  # Multiply by 0.01
# print("deptj_image", depth_image[240])




# depth_image = depth_image.astype(np.float64)


rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(rgb_image),
    o3d.geometry.Image(depth_image),
    convert_rgb_to_intensity=False
)

# Extract camera intrinsic parameters from the k_matrix
fx = k_matrix[0, 0]  # focal length x
fy = k_matrix[1, 1]  # focal length y
cx = k_matrix[0, 2]  # principal point x
cy = k_matrix[1, 2]  # principal point y

# Create PinholeCameraIntrinsic object
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=rgb_image.shape[1],
    height=rgb_image.shape[0],
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy
)

# Generate the point cloud from the RGBD image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    camera_intrinsics
)

# Flip it to align correctly
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])








# depth_image = depth_image.astype(np.float32)  # Convert to float32
# depth_image *= 0.01  # Multiply by 0.01
# # depth_image *= 0.1

# # print("depth_image", depth_image)
# print("Max of depth image: ", np.max(depth_image))
# print("Min of depth image: ", np.min(depth_image))


# input_for_cgn = {
#                  'rgb': rgb_image,
#                  'depth': depth_image,
#                  'K': k_matrix,
#                  'seg': combined_mask,
# }
# np.save("input_for_cgn.npy", input_for_cgn)



# # cgn = CGN(input_path="results/input_for_cgn.npy", 
# #           K=input_for_cgn['K'], z_range = [0.2,10],
# #           local_regions = True,
# #           filter_grasps = True,
# #           skip_border_objects = True,
# #           visualize=True, 
# #           forward_passes=3)

# cgn = CGN(input_path="input_for_cgn.npy", K=k_matrix, z_range = [10,50], visualize=True, forward_passes=2)

# pred_grasps, grasp_scores, contact_pts, gripper_openings = cgn.inference()