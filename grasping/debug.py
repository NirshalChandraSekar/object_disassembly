from grasping.contact_graspnet_pytorch.inference import CGN
import numpy as np

cgn = CGN(input_path="results/input_for_cgn.npy", 
          K=input_for_cgn['K'], z_range = [0.2,10],
          local_regions = True,
          filter_grasps = True,
          skip_border_objects = True,
          visualize=True, 
          forward_passes=3)

pred_grasps, grasp_scores, contact_pts, gripper_openings = cgn.inference()