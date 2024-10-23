import glob
import os
import argparse

import transforms3d as t3d


import torch
import numpy as np
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils

from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image
from contact_graspnet_pytorch.checkpoints import CheckpointIO 
from contact_graspnet_pytorch.data import load_available_input_data



class CGN():
    def __init__(self,input_path,K=None,z_range = [0.2,10],visualize = True,forward_passes=1):
        self.global_config = config_utils.load_config('grasping/checkpoints/contact_graspnet/config.yaml')
        self.ckpt_dir = 'grasping/checkpoints/contact_graspnet/checkpoints'
        self.input_paths = input_path
        self.local_regions = True
        self.filter_grasps = True
        self.skip_border_objects = True
        self.z_range = z_range
        self.forward_passes = forward_passes
        self.K = K
        self.visualize = visualize

    def get_best_pose_cgn_score(self,pred_grasps, grasp_scores, contact_pts, gripper_openings):
        best_grasp_idx = np.argmax(grasp_scores[1.0])
        pose = pred_grasps[1.0][best_grasp_idx]
        print("Best Grasp Pose : ",pose)
        # print("Index of Best Grasp : ",best_grasp_idx)
        # print("Grasp Score : ",grasp_scores[1.0][best_grasp_idx])
        return pose

    def best_pose_verticle(self,pred_grasps, grasp_scores, contact_pts, gripper_openings,view_matrix):
        
        euler_angle_of_poses = []
        ## Get the pose matrix for all the grasps.
        for pose in pred_grasps[1.0]:
            ## get the matrix
            pose, pose_matrix = self.cgn_to_pybullet(pose, view_matrix)
            euler_angle_of_poses.append(pose['orientation'])
        
        score_verticle = []
        for euler_angle in euler_angle_of_poses:
            ### Set the logic here.
            score_verticle.append(np.abs(euler_angle[2]))

        best_grasp_idx = np.argmax(score_verticle)
        best_pose = pred_grasps[1.0][best_grasp_idx]

        return best_pose

    def cgn_to_pybullet(self, pose, view_matrix): 
        """
        Converts a camera pose to PyBullet coordinate system.

        Parameters:
            pose (numpy.ndarray): The pose of the camera.
            view_matrix (numpy.ndarray): The view matrix of the camera.

        Returns:
            tuple: A tuple containing the grasp pose and the transformed camera pose.
        """
        # Camera to world conversion
        view_matrix_transpose = np.array(view_matrix).reshape(4, 4).T
        view_matrix_inverse = np.linalg.inv(view_matrix_transpose)

        # Apply a rotation around x-axis to align with PyBullet's coordinate system
        rotation_x = t3d.euler.euler2mat(np.pi, 0, 0, axes='sxyz')
        rotation_x_4x4 = np.eye(4)
        rotation_x_4x4[:3, :3] = rotation_x.T
        rotated_pose = np.matmul(rotation_x_4x4, pose)

        transformed_pose = np.matmul(view_matrix_inverse, rotated_pose)

        # Offset adjustment for gripper size
        gripper_size = 0.01  # Size of the gripper
        approach_vector = transformed_pose[:3, 2]
        offset = - gripper_size * approach_vector
        # transformed_pose[:3, 3] += offset

        # Extract translation and rotation for the grasp pose
        translation = transformed_pose[:3, 3] + offset
        euler_angles = t3d.euler.mat2euler(transformed_pose[0:3, 0:3], axes='sxyz')

        grasp_pose = {'position': translation, 'orientation': euler_angles}
        print("Final Grasp Pose in World Frame: ", grasp_pose)
        
        return grasp_pose, transformed_pose

    def inference(self):
        """
        Predict 6-DoF grasp distribution for given model and input data
        
        :param global_config: config.yaml from checkpoint directory
        :param checkpoint_dir: checkpoint directory
        :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
        :param K: Camera Matrix with intrinsics to convert depth to point cloud
        :param local_regions: Crop 3D local regions around given segments. 
        :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
        :param filter_grasps: Filter and assign grasp contacts according to segmap.
        :param segmap_id: only return grasps from specified segmap_id.
        :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
        :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
        """
        # Build the model
        grasp_estimator = GraspEstimator(self.global_config)

        # Load the weights
        # model_checkpoint_dir = os.path.join(self.ckpt_dir, 'checkpoints')
        model_checkpoint_dir = "checkpoints/contact_graspnet/checkpoints"
        checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
        try:
            load_dict = checkpoint_io.load('model.pt')
        except FileExistsError:
            print('No model checkpoint found')
            load_dict = {}

        
        os.makedirs('grasping/results', exist_ok=True)

        # Process example test scenes
        for p in glob.glob(self.input_paths):
            print('Loading ', p)

            pc_segments = {}
            segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=self.K)
            
            if segmap is None and (self.local_regions or self.filter_grasps):
                raise ValueError('Need segmentation map to extract local regions or filter grasps')

            if pc_full is None:
                print('Converting depth to point cloud(s)...')
                pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                        skip_border_objects=self.skip_border_objects, 
                                                                                        z_range=self.z_range)
            
            print(pc_full.shape)

            print('Generating Grasps...')
            pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(pc_full, 
                                                                                        pc_segments=pc_segments, 
                                                                                        local_regions=self.local_regions, 
                                                                                        filter_grasps=self.filter_grasps, 
                                                                                        forward_passes=self.forward_passes)  
        
            # Save results
            np.savez('grasping/results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
                    pc_full=pc_full, pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts, pc_colors=pc_colors, gripper_openings=gripper_openings)

            if self.visualize:
            # Visualize results          
                show_image(rgb, segmap)
                visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
            
        if not glob.glob(self.input_paths):
            print('No files found: ', self.input_paths)

        return pred_grasps_cam, scores, contact_pts, gripper_openings



# def inference(global_config, 
#               ckpt_dir,
#               input_paths, 
#               local_regions=True, 
#               filter_grasps=True, 
#               skip_border_objects=False,
#               z_range = [0.2,1.8],
#               forward_passes=1,
#               K=None,):
#     """
#     Predict 6-DoF grasp distribution for given model and input data
    
#     :param global_config: config.yaml from checkpoint directory
#     :param checkpoint_dir: checkpoint directory
#     :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
#     :param K: Camera Matrix with intrinsics to convert depth to point cloud
#     :param local_regions: Crop 3D local regions around given segments. 
#     :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
#     :param filter_grasps: Filter and assign grasp contacts according to segmap.
#     :param segmap_id: only return grasps from specified segmap_id.
#     :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
#     :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
#     """
#     # Build the model
#     grasp_estimator = GraspEstimator(global_config)

#     # Load the weights
#     model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
#     checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
#     try:
#         load_dict = checkpoint_io.load('model.pt')
#     except FileExistsError:
#         print('No model checkpoint found')
#         load_dict = {}

    
#     os.makedirs('results', exist_ok=True)

#     # Process example test scenes
#     for p in glob.glob(input_paths):
#         print('Loading ', p)

#         pc_segments = {}
#         segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
        
#         if segmap is None and (local_regions or filter_grasps):
#             raise ValueError('Need segmentation map to extract local regions or filter grasps')

#         if pc_full is None:
#             print('Converting depth to point cloud(s)...')
#             pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
#                                                                                     skip_border_objects=skip_border_objects, 
#                                                                                     z_range=z_range)
        
#         print(pc_full.shape)

#         print('Generating Grasps...')
#         pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(pc_full, 
#                                                                                        pc_segments=pc_segments, 
#                                                                                        local_regions=local_regions, 
#                                                                                        filter_grasps=filter_grasps, 
#                                                                                        forward_passes=forward_passes)  
    
#         # Save results
#         np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
#                   pc_full=pc_full, pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts, pc_colors=pc_colors)

#         # Visualize results          
#         show_image(rgb, segmap)
#         visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        
#     if not glob.glob(input_paths):
#         print('No files found: ', input_paths)
        
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt_dir', default='checkpoints/contact_graspnet', help='Log dir')
#     parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
#     parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
#     parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
#     parser.add_argument('--local_regions', action='store_true', default=True, help='Crop 3D local regions around given segments.')
#     parser.add_argument('--filter_grasps', action='store_true', default=True,  help='Filter grasp contacts according to segmap.')
#     parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
#     parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
#     parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
#     FLAGS = parser.parse_args()

#     global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
#     print(str(global_config))
#     print('pid: %s'%(str(os.getpid())))

#     inference(global_config, 
#               FLAGS.ckpt_dir,
#               FLAGS.np_path, 
#               local_regions=FLAGS.local_regions,
#               filter_grasps=FLAGS.filter_grasps,
#               skip_border_objects=FLAGS.skip_border_objects,
#               z_range=eval(str(FLAGS.z_range)),
#               forward_passes=FLAGS.forward_passes,
#               K=eval(str(FLAGS.K)))