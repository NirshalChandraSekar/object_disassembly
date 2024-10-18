from contact_graspnet_pytorch.inference import CGN
import numpy as np

combined_mask = np.load("/home/rpmdt05/Code/Niru/object_disassembly/object-detection/combined_mask.npy")
rgb_image = np.load("/home/rpmdt05/Code/Niru/object_disassembly/streming-pipeline/color_image.npy")
depth_image = np.load("/home/rpmdt05/Code/Niru/object_disassembly/streming-pipeline/depth_image.npy")
k_matrix = np.load("/home/rpmdt05/Code/Niru/object_disassembly/streming-pipeline/k_matrix.npy")

depth_image = np.array(depth_image, dtype=np.float32)  # Convert to float32
depth_image *= 0.01  # Multiply by 0.01

print("depth_image", depth_image)

input_for_cgn = {
                 'rgb': rgb_image,
                 'depth': depth_image,
                 'K': k_matrix,
                 'seg': combined_mask,
}
np.save("input_for_cgn.npy", input_for_cgn)



# cgn = CGN(input_path="results/input_for_cgn.npy", 
#           K=input_for_cgn['K'], z_range = [0.2,10],
#           local_regions = True,
#           filter_grasps = True,
#           skip_border_objects = True,
#           visualize=True, 
#           forward_passes=3)

cgn = CGN(input_path="input_for_cgn.npy", K=k_matrix, z_range = [0.2,10], visualize=True, forward_passes=3)

pred_grasps, grasp_scores, contact_pts, gripper_openings = cgn.inference()