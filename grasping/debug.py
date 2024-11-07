# from contact_graspnet_pytorch.inference import CGN
import numpy as np
import cv2
import open3d as o3d

def plot_coordinates(vis, t, r, tube_radius=0.005, central_color=None):
    """
    Plots coordinate frame

    Arguments:
        t {np.ndarray} -- translation vector
        r {np.ndarray} -- rotation matrix

    Keyword Arguments:
        tube_radius {float} -- radius of the plotted tubes (default: {0.005})
    """

    # Create a line for each axis of the coordinate frame
    lines = []
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue

    if central_color is not None:
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        ball.paint_uniform_color(np.array(central_color))
        vis.add_geometry(ball)

    for i in range(3):
        line_points = [[t[0], t[1], t[2]],
                       [t[0] + 0.2 * r[0, i], t[1] + 0.2 * r[1, i], t[2] + 0.2 * r[2, i]]]

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(line_points)
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        line.colors = o3d.utility.Vector3dVector(np.array([colors[i]]))

        line.paint_uniform_color(colors[i])  # Set line color
        lines.append(line)

    # Visualize the lines in the Open3D visualizer
    for line in lines:
        vis.add_geometry(line)


# def plot_single_grasp(vis, grasp_pose, contact_pt, gripper_width=0.08):
#     """
#     Plots a single best grasp pose in Open3D.
    
#     Arguments:
#         vis {o3d.visualization.Visualizer} -- Open3D visualizer instance
#         grasp_pose {np.ndarray} -- 4x4 grasp pose transformation
#         contact_pt {np.ndarray} -- Contact point in the scene
#         gripper_width {float} -- Width of the gripper for visual representation
#     """
#     # Define gripper control points in the local frame
#     gripper_control_points = np.array([[0, 0, 0], [gripper_width / 2, 0, 0], 
#                                        [-gripper_width / 2, 0, 0], [0, 0.1, 0], [0, -0.1, 0]])
    
#     # Transform control points using the grasp pose
#     transformed_points = (gripper_control_points @ grasp_pose[:3, :3].T) + grasp_pose[:3, 3]
    
#     # Define lines connecting control points
#     line_connections = [[0, 1], [0, 2], [1, 3], [2, 4]]
    
#     # Create Open3D LineSet for gripper visualization
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(transformed_points)
#     line_set.lines = o3d.utility.Vector2iVector(line_connections)
#     line_set.paint_uniform_color([0, 1, 0])  # Gripper color (e.g., green)
    
#     # Visualize the contact point as a small sphere
#     contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
#     contact_sphere.translate(contact_pt)
#     contact_sphere.paint_uniform_color([1, 0, 0])  # Contact point color (e.g., red)
    
#     # Add the gripper and contact point to the visualizer
#     vis.add_geometry(line_set)
#     vis.add_geometry(contact_sphere)


# understand the outputs of contact graspnet

pred_grasps_cam = np.load("data/pred_grasps_cam.npy", allow_pickle=True).item()
scores = np.load("data/scores.npy", allow_pickle=True).item()
contact_pts = np.load("data/contact_pts.npy", allow_pickle=True).item()

color_image = np.load("../streaming_pipeline/data/color_image.npy", allow_pickle=True)

depth_image = np.load("../streaming_pipeline/data/depth_image.npy", allow_pickle=True)
k_matrix = np.load("../streaming_pipeline/data/k_matrix.npy", allow_pickle=True)

depth_image = depth_image.astype(np.float32)
print("depth image", depth_image)

best_score_idx = {}
for key in scores:
    if len(scores[key]) == 0:
        best_score_idx[key] = None
    else:
        best_score_idx[key] = np.argmax(scores[key])

best_grasp_cam = {}
for key in pred_grasps_cam:
    best_grasp_cam[key] = pred_grasps_cam[key][best_score_idx[key]]

best_contact_pts = {}
for key in contact_pts:
    best_contact_pts[key] = contact_pts[key][best_score_idx[key]]


print(best_grasp_cam)


''' PLOT THE POINT CLOUD USING THE COLOR IMAGE, DEPTH IMAGE AND THE CAMERA INTRINSICS '''
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
# depth_image /= 0.00025

# Convert images to Open3D format
color = o3d.geometry.Image(color_image)
depth = o3d.geometry.Image(depth_image)

# print("depth image", depth_image)

# Create Open3D RGBD image from color and depth images
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False
)

# Create Open3D camera intrinsic object from K matrix
height, width = depth_image.shape
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, k_matrix[0, 0], k_matrix[1, 1], k_matrix[0, 2], k_matrix[1, 2])

# Create point cloud from RGBD image and camera intrinsics
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / 0.001)  # Convert to meters
# # Flip the point cloud to correct orientation
# # pcd.transform([[1, 0, 0, 0],
# #                [0, -1, 0, 0],
# #                [0, 0, -1, 0],
# #                [0, 0, 0, 1]])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

grasp_1 = best_grasp_cam[1]
t_grasp = grasp_1[:3, 3]
R_grasp = grasp_1[:3, :3]

grasp_2 = best_grasp_cam[2]
t_grasp_2 = grasp_2[:3, 3]
R_grasp_2 = grasp_2[:3, :3]


plot_coordinates(vis, t_grasp, R_grasp, central_color=(0.5, 0.5, 0.5))
# plot_coordinates(vis, t_grasp_2, R_grasp_2, central_color=(0.5, 0.5, 0.5))


print("point cloud", np.asarray(pcd.points))

# plot_coordinates(vis, np.zeros(3,),np.eye(3,3), central_color=(0.5, 0.5, 0.5))

# Plot the camera coordinate frame
T_world_cam = np.eye(4)
T_cam_world = np.linalg.inv(T_world_cam)

# T_cam_world = [[1, 0, 0, 0],
#                [0, -1, 0, 0],
#                [0, 0, -1, 0],
#                [0, 0, 0, 1]] @ T_cam_world

t = T_cam_world[:3, 3]
R = T_cam_world[:3, :3]

plot_coordinates(vis, t, R)

vis.run()
vis.destroy_window()


