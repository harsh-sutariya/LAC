#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This agent demonstrates how to structure your code, visualize camera data in 
an OpenCV window, control the robot with keyboard commands using pynput, and
perform stereo matching with RAFT-Stereo instead of OpenCV's StereoSGBM.
"""
import sys
import os
# Add the agents directory to Python path to ensure core module imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import carla
import cv2 as cv
import random
from math import radians
from pynput import keyboard

# Import the AutonomousAgent from the Leaderboard
from leaderboard.autoagents.autonomous_agent import AutonomousAgent

# Define the entry point so that the Leaderboard can instantiate the agent class
def get_entry_point():
    return 'OpenCVagent'

# -----------------------------
#  RAFT-Stereo Dependencies
# -----------------------------
import torch
# Adjust these imports to match your RAFT-Stereo code structure
# For example, if your RAFT-Stereo repo is in the same directory:
from core.raft_stereo import RAFTStereo          # The main RAFT-Stereo model class
from core.utils.utils import InputPadder         # Utility for padding images
import argparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

# Ros based imports
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2

class OpenCVagent(AutonomousAgent):
    """
    OpenCVagent: Demonstrates how to:
      1. Initialize and load RAFT-Stereo in setup().
      2. Capture stereo images from Carla's sensors.
      3. Compute a disparity (and approximate depth) via RAFT-Stereo.
      4. Visualize the results in OpenCV windows.
      5. Control the rover via keyboard arrow keys.
    """

    def setup(self, path_to_conf_file):
        """
        Called once at mission start. 
        - We load the RAFT-Stereo model and checkpoint here.
        - We also start a keyboard listener for manual control.
        """

        # ------------------------------------------------
        # Start keyboard listener for manual driving
        # ------------------------------------------------
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        # Initialize linear (v) and angular (w) velocities
        self.current_v = 0
        self.current_w = 0

        # Simulation frame counter
        self.frame = 0

        self.t=0
        self.curr_path_param=-10
        self.kp=5
        self.ki=0#19.2523
        self.kd=0#-0.1124

        self.v_ie = 0
        self.v_de = 0
        self.v_e = 0
        self.v_pe = 0

        self.w_ie = 0
        self.w_de = 0
        self.w_e = 0
        self.w_pe = 0
        self.x=0
        self.y=0
        self.wpx=0
        self.wpy=0
        self.wpyaw=0
        self.eyaw = 0
        self.edist=0

        #obstacles
        self.obstacles=[] #list of lists, with each sublist of the format [x_c,y_c,r,xmin,xmax,ymin,ymax,zmin,zmax] 
        self.obst_dist=0.7 #computed based on bot dimensions and tolerance
        self.curr_obstacle = -1
        self.obs_t_zone=-1

        # Contrast-Limited Adaptive Histogram Equalization (optional enhancement)
        self.clhae = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # ------------------------------------------------
        # RAFT-Stereo Initialization
        # ------------------------------------------------
        class Args:
            # Path to your RAFT-Stereo weights
            restore_ckpt = "models/raftstereo-realtime.pth"
            # Common RAFT-Stereo settings
            mixed_precision = True
            # Use "reg" instead of "reg_cuda" since CUDA extensions are not compiled
            corr_implementation = "reg"
            corr_levels = 4
            corr_radius = 4
            n_downsample = 3
            context_norm = 'instance'
            slow_fast_gru = True
            n_gru_layers = 2
            hidden_dims = [128, 128, 128]
            shared_backbone = True
            valid_iters = 7 
            # left_image = 
            # Add other RAFT-Stereo args if needed

        self.args = Args()
        
        # Create model
        # self.raft_model = RAFTStereo(args)
        # Wrap with DataParallel if desired (single GPU is also fine):
        self.raft_model = torch.nn.DataParallel(RAFTStereo(self.args), device_ids=[0])

        # Load checkpoint
        # state_dict = torch.load(args.restore_ckpt)
        self.raft_model.load_state_dict(torch.load(self.args.restore_ckpt), strict = False)

        # Remove DataParallel wrapper to get the raw model
        self.raft_model = self.raft_model.module
        self.raft_model.to('cuda')
        self.raft_model.eval()

        # Store the InputPadder if you want quick access
        self.InputPadder = InputPadder
        self.cx = 320
        self.cy = 240
        # Create Open3D visualization window
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.combined_points = []


        # Setup Ros publisher node for PCD
        rospy.init_node('point_cloud_publisher', anonymous=True)
        self.pcd_publisher = rospy.Publisher('/camera/point_cloud', PointCloud2, queue_size=15)


    def use_fiducials(self):
        """
        Override method to specify if agent wants to use fiducials.
        We return True here, but adjust based on your needs.
        """
        return False

    def sensors(self):
        """
        Define which sensors we want, their resolutions, and initial activation states.
        Here, we only activate the FrontLeft and FrontRight cameras (grayscale).
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 2.0, 'width': '640', 'height': '480', 'use_semantic': True
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 2.0, 'width': '640', 'height': '480'
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
        }
        return sensors

    def run_step(self, input_data):
        """
        Called every simulation tick. We:
        1. Possibly set some initial arm angle (once).
        2. Retrieve left and right camera frames.
        3. Apply CLAHE for contrast enhancement.
        4. Compute disparity + depth using RAFT-Stereo in compute_depth_map().
        5. Return a control command (linear & angular velocity).
        6. End the mission after 5000 frames.
        """
         # Prepare PointCloud2 message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "World_frame"  # Set the frame ID appropriately

        # Example: set arm angles at first frame
        if self.frame == 0:
            self.set_front_arm_angle(radians(80))
            self.set_back_arm_angle(radians(60))
            self.farm_state = True
            # Parameters for camera to robot 
            transf_c2r = self.get_camera_position(carla.SensorPosition.FrontLeft)
            self.transl_c2r = np.array([transf_c2r.location.x, transf_c2r.location.y, transf_c2r.location.z])
            # print(self.transl_c2r)
            # rot_c2r = 0 since its fixed
        
        # Parameters for robot to world(Currently using GT for testing)
        transf_r2w = self.get_transform()
        r, p, y = transf_r2w.rotation.roll, transf_r2w.rotation.pitch, transf_r2w.rotation.yaw
        # rot_r2w = self.eul_to_rot(r, p, y)
        rpy = np.array([r,p,y])
        rot = R.from_euler('xyz', rpy)
        rot_r2w = rot.as_matrix()
        # print(rot_r2w)
        transl_r2w = np.array([transf_r2w.location.x, transf_r2w.location.y, transf_r2w.location.z])
        # print(transl_r2w)
        transf_r2w = np.eye(4)
        transf_r2w[:3, :3] = rot_r2w
        transf_r2w[:3, 3] = transl_r2w
        # transf_r2w = np.vstack(transf_r2w, np.array([0,0,0,1]))
        # print(transf_r2w)

        # Retrieve front-left and front-right grayscale frames
        left_sensor_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
        right_sensor_data = input_data['Grayscale'][carla.SensorPosition.FrontRight]
        semantic_data = input_data['Semantic'].get(carla.SensorPosition.FrontLeft, None)


        if left_sensor_data is not None and right_sensor_data is not None and self.get_mission_time()>3:
            # Show one raw image for debugging
            cv.imshow("Raw Left Image:", left_sensor_data)
            # cv.imshow("Sematic Left:", semantic_data)

            # Enhance images using CLAHE
            left_image = self.clhae.apply(left_sensor_data)
            right_image = self.clhae.apply(right_sensor_data)

            # Convert single-channel to RGB (3 channels)
            left_image = cv.cvtColor(left_sensor_data, cv.COLOR_GRAY2RGB)
            right_image = cv.cvtColor(right_sensor_data, cv.COLOR_GRAY2RGB)


            # --------------------------------------
            # Convert images to PyTorch tensors (with scaling)
            # --------------------------------------
            # left_torch = torch.from_numpy(left_image).permute(2, 0, 1).float().unsqueeze(0) 
            # right_torch = torch.from_numpy(right_image).permute(2, 0, 1).float().unsqueeze(0) 
            left_image = np.array(left_image.astype(np.uint8))
            right_image = np.array(right_image.astype(np.uint8))
            
            # left_image = left_image[140:, 140:640]
            # right_image = right_image[140:, :-140]
            # cv.imshow("Left Image:", left_sensor_data)
            # cv.imshow("Cropped Left Image:", left_image)
            
            # cv.imshow("Right Image:", right_sensor_data)
            # cv.imshow("Cropped Right Image:", right_image)



            left_torch = torch.from_numpy(left_image).permute(2, 0, 1).float()
            right_torch = torch.from_numpy(right_image).permute(2, 0, 1).float() 

            # Move to GPU
            left_torch = left_torch[None].cuda()
            right_torch = right_torch[None].cuda()

            # Pad if necessary (RAFT-Stereo often needs dims multiple of 32)
            padder = self.InputPadder(left_torch.shape, divis_by=32)
            left_torch, right_torch = padder.pad(left_torch, right_torch)

            # --------------------------------------
            # 2) Forward Pass in RAFT-Stereo
            # --------------------------------------
            with torch.no_grad():
                flow_low, flow_up = self.raft_model(left_torch, right_torch, iters=self.args.valid_iters, test_mode=True)

            # Unpad to original dimensions
            flow_up = padder.unpad(flow_up).squeeze(0).cpu().numpy()

            # Visualize and compute depth map (as done before)
            disparity = -flow_up[0]
            # print(np.max(disparity), np.min(disparity), np.mean(disparity))
            disparity_clamped = np.where(disparity < 0, 0, disparity)
            # disp_vis = cv.normalize(disparity_clamped, None, 0, 255, cv.NORM_MINMAX)
            # disp_vis = np.uint8(disp_vis)
            disp_vis = np.uint8(disparity_clamped)

            # disp_vis_colormap = cv.applyColorMap(disp_vis, cv.COLORMAP_JET)
            # print(np.max(disparity), np.min(disparity), np.mean(disparity))

            # cv.imshow("RAFT Disparity", disp_vis_colormap)

            # Approximate Depth Calculation
            focal_length = 458   # Example calibration value
            baseline = 0.162     # Example baseline in meters
            # print(disparity.shape)
            disparity[disparity <= 0.01] = 1
            disparity = disparity[:400,150:]
            disparity_viz = cv.applyColorMap(np.uint8(disparity), cv.COLORMAP_JET)
            # cv.imshow("Raft Disparity", disparity_viz)
            depth_map = (focal_length * baseline) / disparity
            depth_map[depth_map>=2] = 0
            depth_map[depth_map<=0.2] = 0

            depth_vis = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)
            depth_vis = np.uint8(depth_vis)
            # depth_vis_colormap = cv.applyColorMap(-depth_vis, cv.COLORMAP_JET)
            
            # MAsking the depth map
            # depth_vis = depth_vis.T
            # print(np.shape(depth_vis))
            # print(np.shape(semantic_data))
            mask_rock = semantic_data[:400, 150:,:]==[108,59,42]
            # print('mask rock shape',mask_rock.shape)
            mask_rock=mask_rock[:,:,1]
            mask = cv.cvtColor(semantic_data,cv.COLOR_BGR2GRAY) != 0
            mask = mask[:400, 150:]
            masked_depth = np.where(mask, depth_map,0)
            # masked_rock_depth = np.where(mask_rock,depth_map,0)
            rock_mask=mask_rock.flatten()
            # depth_vis_colormap = cv.applyColorMap(masked_depth, cv.COLORMAP_JET)
            
            # print(np.max(depth_map), np.min(depth_map), np.mean(depth_map))
            # cv.imshow("RAFT Depth Map", depth_vis_colormap)
            
            H, W = depth_map.shape
            # Create a pixel grid to further manipulate individual image pixels for depth calculation
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            # print(u.shape,v.shape)

            u = u.flatten()
            v = v.flatten()
            # print(u.shape)
            # To use masked depth use the following line
            depth = masked_depth.flatten()
            # rock_depth = masked_rock_depth.flatten()
            # To use unmasked depth use the following line
            # depth = depth_vis.flatten()
            
            valid_points = depth > 0 
            # print(valid_points)
            Y = -((u[valid_points]+150 - self.cx) * depth[valid_points] / focal_length)
            Z = -((v[valid_points] - self.cy) * depth[valid_points] / focal_length)
            X = depth[valid_points]
            rock_points=rock_mask[valid_points]
            # Y_rock= -((u[rock_points]+150 - self.cx) * rock_depth[rock_points] / focal_length)
            # Z_rock = -((v[rock_points] - self.cy) * rock_depth[rock_points] / focal_length)
            # X_rock = rock_depth[rock_points]

            
            
            points_cam_coor = np.vstack((X, Y, Z)).T
            points_cam_coor = points_cam_coor[rock_points] ##uncomment this to filter out non-rock points
            # Update the point cloud
            self.pcd.points = o3d.utility.Vector3dVector(points_cam_coor)
            # R_visualize = self.pcd.get_rotation_matrix_from_xyz((0, np.pi, np.pi))  # 45 degrees in radians
            # self.pcd.rotate(R_visualize)

            """
            Code Block for PCD filteration and processing:
            """
            # Statistical outlier removal:
            downsampled_pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors = 10, std_ratio = 1)
            # Voxel downsampling:
            downsampled_pcd = downsampled_pcd.voxel_down_sample(voxel_size = 0.025)
            # DBSCAN clustering

            labels = np.array(downsampled_pcd.cluster_dbscan(eps=0.1, min_points=20, print_progress=False))
            # max_label = labels.max()

            # Select only the largest cluster
            downsampled_pcd = downsampled_pcd.select_by_index(np.where(labels >= 0)[0])
            downsampled_pcd = np.asarray(downsampled_pcd.points)
            
            """
            Code Block for transformation from cam to robot to world coordinate frame
            """
            pcd_rob = downsampled_pcd + self.transl_c2r
            # print(pcd_rob.shape)
            pcd_rob_hom = np.hstack([pcd_rob, np.ones((downsampled_pcd.shape[0], 1))])

            # pcd_world = (rot_r2w)@(pcd_rob.T) + transl_r2w.reshape(3,1)
            pcd_world = transf_r2w@(pcd_rob_hom.T)
            # print(pcd_world.shape)

            pcd_world = pcd_world[0:3,:]

            # pcd_rock = pcd_world[:,rock_points]
            # pcd_rock = pcd_world
            # pcd_world=pcd_rock.copy()
            # pcd_world = pcd_world[:, ~np.isnan(pcd_world).any(axis=0)]
            # print(pcd_world.shape)
            # print(np.min(pcd_rob[2,:]))
            # print(np.max(pcd_rob[2,:]))


              ### OBSTACLE LIST UPDATE
            # run object detection for rocks/lander and get corresponding xmin,xmax,ymin,ymax for each rock
            # compute the center and radius of cicle enclosing points corresponding to a rock [ignore it if the radius is  below a threshold]
            print('obstacle count ',len(self.obstacles))
            
            #clustering rock points
            if pcd_world.shape[1]!=0:
                db = DBSCAN(eps=0.1,min_samples=2).fit(pcd_world.T)
                rock_labels = (db.labels_)
                rock_set = set(rock_labels)
                rock_set.discard(-1)
                print(rock_set)
                for i in range(len(rock_set)):    
                    P=(pcd_world.T)[rock_labels==i].copy()
                    P=P.T
                    # print('PC shape',P.shape)
                    if(P.size!=0):
                        [xmin,ymin,zmin,xmax,ymax,zmax]=[min(P[0,:]),min(P[1,:]),min(P[2,:]),max(P[0,:]),max(P[1,:]),max(P[2,:])]
                        [x_c,y_c]=[(xmin+xmax)/2,(ymin+ymax)/2]
                        r_obs = np.linalg.norm([xmax-x_c,ymax-y_c]) 
                        # check if the distance of this circle is less than r1+r2+2*(thrsh_safe) with any other obstacle in the list
                            # if yes, group them together, updating the old obstacle entry
                            # if no, create a new obstacle entry 
                        overlap=False
                        for o in range(len(self.obstacles)):
                            rock = self.obstacles[o]
                            if np.linalg.norm([x_c-rock[0],y_c-rock[1]])<r_obs+rock[2]:
                                    overlap=True
                            if overlap: 
                                lim_new = [min(xmin,rock[3]),max(xmax,rock[4]),min(ymin,rock[5]),max(ymax,rock[6]),min(zmin,rock[7]),max(zmax,rock[8])]
                                circ_new = [sum(lim_new[:2])/2,sum(lim_new[2:4])/2]
                                circ_new.append(np.linalg.norm([lim_new[1]-circ_new[0],lim_new[3]-circ_new[1]]))
                                self.obstacles[o] =[*circ_new,*lim_new]
                                # print('updated to',self.obstacles[o][:3])
                                break  
                        if not overlap:
                            new_obs=[x_c,y_c,r_obs,xmin,xmax,ymin,ymax,zmin,zmax]
                            self.obstacles.append(new_obs)
                            # print('Added',new_obs[:3])
                    else:
                        print('Zero size rock ignored')


            self.pcd.points = o3d.utility.Vector3dVector(pcd_world.T)

            # Saving the point cloud
            if not hasattr(self, 'global_pcd'):
                self.global_pcd = []
            self.global_pcd.append(pcd_world.T)

            """
            Code segment to visualise point cloud in RVIZ
            """
            # Convert point cloud to PointCloud2 format
            points = np.asarray(self.pcd.points)
            points = points[~np.isnan(points).any(axis=1)]  # Remove NaN rows
            
            cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
            # Publish the PointCloud2 message
            self.pcd_publisher.publish(cloud_msg)

            cv.waitKey(1)

            # Increment frame counter
            self.frame += 1


        """Execute one step of navigation"""
        v=0
        w=0

        loc = self._vehicle_status.transform.location
        rot = self._vehicle_status.transform.rotation

        # dx=0.1
        # dy=0.1

        # self.path_op = np.vstack([self.path_op,np.array([loc.x,loc.y])])
        
        
        if self.t == 0:
            self.set_front_arm_angle(np.radians(70))
            self.set_back_arm_angle(np.radians(60))
            self.x=loc.x
            self.y = loc.y
            self.r = np.linalg.norm([self.x,self.y])

            # self.wpx = self.x+0.01
            # self.wpy = self.y
            # self.wpyaw = 0

            self.t=1
            # self.max_v=0
            control = carla.VehicleVelocityControl(0, 0) 

        else:
            # print(self.v_e,self.v_de,self.v_ie)
            # print(self.w_e,self.w_de,self.w_ie)
            # wps = self.local_wps(np.array([loc.x,loc.y,rot.yaw]),self.r)
            self.v_e,self.w_e = self.comp_err(np.array([loc.x,loc.y,rot.yaw]),self.r)
            # print('curr_r',np.linalg.norm([loc.x,loc.y]))
            # self.v_e = np.dot([wps[0][0],wps[0][1]],[np.cos(rot.yaw),np.sin(rot.yaw)])
            # self.v_e = np.linalg.norm(wps[0][:2])
            self.v_ie+=self.v_e*0.05
            self.v_de = (self.v_e-self.v_pe)/0.05
            self.v_pe = self.v_e
            v = self.kp*self.v_e + self.ki*self.v_ie + self.kd*self.v_de

            # self.w_e = wps[0][2]
            # self.w_e = np.sqrt(np.linalg.norm(wps[0][:2])**2-self.v_e**2)
            self.w_ie+=self.w_e*0.05
            self.w_de = (self.w_e-self.w_pe)/0.05
            self.w_pe = self.w_e
            w = self.kp*self.w_e + self.ki*self.w_ie + self.kd*self.w_de

            #velocity 
            scale=1
            # if max(abs(v),abs(w))>0.25:
            #     scale = 0.25/max(abs(v),abs(w))
            if abs(v)>0.25:
                scale=(0.25)/abs(v)
                v*=scale
                w*=scale
            if abs(w)>0.25:
                scale=(0.25)/abs(w)
                v*=scale
                w*=scale

            scale=1
            # if max(abs(v),abs(w))>0.25:
            #     scale = 0.25/max(abs(v),abs(w))
            if abs(v)<0.1 and abs(w)<0.1:
                scale = 0.1/max(abs(v),abs(w))
                v*=scale
                w*=scale
            print('velocities',v,w)



            control = carla.VehicleVelocityControl(v,w)

        # Control command: linear velocity + angular velocity
        # control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        # End after 5000 frames
        if self.frame >= 500000:
            self.mission_complete()

        return control


    def eul_to_rot(self, pitch, roll, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
                        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
                        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
                        
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R
    
    def global_planner(self,pose_curr,path_params,step_res=0.01):
        r=path_params
        t_curr = np.arctan2(pose_curr[1],pose_curr[0])
        if self.curr_path_param==-10:
            self.curr_path_param=t_curr
        dparam = self.curr_path_param-t_curr
        if abs(dparam)>np.pi:
            dparam=0
        self.r -= dparam*0.16  ## should be += for inward spiral
        # self.r += (self.curr_path_param-t_curr)*0.16
        self.curr_path_param=t_curr
        t_next = t_curr+step_res
        t_future = t_next+step_res
        ##OBSTACLE AVOIDANCE
        #Iterate through the obstacle list to see if the next step enters any obstacle zone
            #if no, continue
            #if yes, compute the target waypoint as next closest point along the circumference of the obstacle avoidance circle 
        [x_next,y_next] = [r*np.cos(t_next),r*np.sin(t_next)]
        [x_future,y_future] = [r*np.cos(t_future),r*np.sin(t_future)]
        in_rock_zone=False
        for o_i in range(len(self.obstacles)):
            o=self.obstacles[o_i]
            if o[8]-o[7]>=0.07:
                if np.linalg.norm([o[0]-x_next,o[1]-y_next])<=o[2]+self.obst_dist:
                    in_rock_zone=True
                    t_zone = np.arctan2(o[1],o[0])
                    if self.curr_obstacle==-1 or ((t_zone-self.obs_t_zone)>=0 and abs(t_zone-self.obs_t_zone)<2*np.pi):
                        self.obs_t_zone=t_zone
                        self.curr_obstacle=o_i    
                        # print('In Rock Zone',o[:3])
        if in_rock_zone:
            o = self.obstacles[self.curr_obstacle]
            print('loc',pose_curr[0],pose_curr[1])
            print('obs',o[0],o[1])
            r_local = o[2]+self.obst_dist
            t_r = np.arctan2(pose_curr[1]-o[1],pose_curr[0]-o[0])
            t_r_next=t_r+step_res*10
            t_r_future=t_r_next+step_res*10
            [x_next,y_next] = [o[0]+r_local*np.cos(t_r_next),o[1]+r_local*np.sin(t_r_next)]
            [x_future,y_future] = [o[0]+r_local*np.cos(t_r_future),o[1]+r_local*np.sin(t_r_future)]
                        # print('next:',[x_next,y_next] )
                        # print('future:',[x_future,y_future] )
        else:
            self.curr_obstacle=-1
            self.obs_t_zone=-1
            
        # r = path_params
        r=self.r
        # print('radius',r)
        # pose_targ = [r*np.cos(t_next),r*np.sin(t_next),np.arctan2(r*(np.sin(t_future)-np.sin(t_next)),r*(np.cos(t_future)-np.cos(t_next)))]
        pose_targ=[x_next,y_next,np.arctan2(y_future-y_next,x_future-x_next)]
        return pose_targ
        
    
    def local_wps(self,pose_curr,global_params,n_wp=1,delta_thresh=0.02):
        wps=[]
        while len(wps)<n_wp:
            pose_targ = self.global_planner(pose_curr,global_params)
            delta = pose_targ-pose_curr
            # delta[2] += np.arctan2(delta[1],delta[0])
            # delta[2]/=2
            # print(pose_targ)
            # print(pose_curr)
            # print(delta)
            delta[2]%=(np.sign(delta[2])*2*np.pi)
            if abs(delta[2])>np.pi:
                delta[2] = -1*np.sign(delta[2])*(2*np.pi-abs(delta[2]))
            # print(delta)
            if(np.max(np.abs(delta))>delta_thresh):
                delta = delta*(delta_thresh/np.max(np.abs(delta)))
            while np.linalg.norm(pose_curr-pose_targ)>1e-6:
                wps.append(delta)
                if len(wps)==n_wp:
                    break    
                pose_curr+=delta
        return np.array(wps)
    
    def comp_err(self,pose_curr,global_params):
        pose_targ = self.global_planner(pose_curr,global_params)
        delta = pose_targ-pose_curr
        while abs(delta[2])>np.pi:
            delta[2]-=np.sign(delta[2])*2*np.pi
        v_e = np.dot([np.cos(pose_curr[2]),np.sin(pose_curr[2])],delta[:2])
        e_lat =np.dot([-np.sin(pose_curr[2]),np.cos(pose_curr[2])],delta[:2])
        # w_e = 0.1*np.arctan2(e_lat,v_e)+(1/(1+abs(e_lat)))*(delta[2])
        w_e = np.arctan(e_lat)+(1/(1+abs(e_lat)))*(delta[2])
        w_e/=2
        scale=1
        print(v_e,w_e)
        # if max(abs(v_e),abs(w_e))>0.02:
        #     scale = 0.02/max(abs(v_e),abs(w_e))
        return [v_e*scale,w_e*scale]
    
    def add_point_cloud(self, points):
        """
        Add a point cloud to the combined list with timestamps.
        """
        global combined_points
        # Add timestamp as an additional field to each point
        # timestamped_points = np.hstack([points, np.full((points.shape[0], 1), timestamp)])
        self.combined_points.append(points)

    def save_combined_point_cloud(self, filename):
        """
        Save all combined point clouds to a PCD file.
        """
        global combined_points
        all_points = np.vstack(self.combined_points)  # Combine all point clouds
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points[:, :3])  # XYZ only
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved combined point cloud to: {filename}")

    def finalize(self):
        """
        Called at the end of the mission (or when ESC is pressed). 
        We close OpenCV windows and optionally update the geometric map.
        """
        cv.destroyAllWindows()
        # self.save_combined_point_cloud("combined_point_cloud.pcd")
        full_pcd = np.vstack(self.global_pcd)
        o3d.io.write_point_cloud("Combined_pcd.pcd", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full_pcd)))

        # Example: updating random data in the geometric map
        geometric_map = self.get_geometric_map()
        for _ in range(100):
            x = 10 * random.random() - 5
            y = 10 * random.random() - 5
            geometric_map.set_height(x, y, random.random())
            rock_flag = random.random() > 0.5
            geometric_map.set_rock(x, y, rock_flag)

    def on_press(self, key):
        """
        Callback when a key is pressed. 
        Adjust linear (v) and angular (w) velocities based on arrow keys.
        """
        if key == keyboard.Key.up:
            self.current_v += 0.3
            self.current_v = np.clip(self.current_v, 0, 0.2)
        elif key == keyboard.Key.down:
            self.current_v -= 0.3
            self.current_v = np.clip(self.current_v, -0.2, 0)
        elif key == keyboard.Key.left:
            self.current_w = 0.6
        elif key == keyboard.Key.right:
            self.current_w = -0.6
        elif key == keyboard.Key.esc:
            # End mission if ESC
            self.mission_complete()
            cv.destroyAllWindows()

    def on_release(self, key):
        """
        Callback when a key is released.
        Stop linear/rotation if arrow keys are released.
        """
        if key == keyboard.Key.up or key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left or key == keyboard.Key.right:
            self.current_w = 0