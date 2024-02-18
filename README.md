# 3D Reconstruction from Stereo Images
## Overview
This project focuses on reconstructing 3D scenes from stereo images using the Middlebury dataset. The key steps involve rectifying the stereo images, computing the disparity map, and finally generating a 3D point cloud of the scene.

### Input
![Two_View_Sterio](images.gif)

### Result

![Two_View_Sterio](Multiview.gif)


## Project Structure
## Dependencies:
numpy, matplotlib, os, imageio, tqdm, transforms3d, pyrender, trimesh, cv2, open3d
File Structure:
dataloader.py: Module for loading Middlebury dataset.
utils.py: Utility functions, including visualization of camera poses.
main.py: Main script for executing the 3D reconstruction pipeline.

## Important Functions:
### rectify_2view: 
Rectifies two input images using provided camera matrices and rotation matrices.
### compute_right2left_transformation: 
Computes the transformation matrix from the right view to the left view and the baseline.
### compute_rectification_R: 
Computes the rectification rotation matrix.
### ssd_kernel, sad_kernel, zncc_kernel: 
Different kernel functions for computing patch similarities.
### Compute_disparity_map: 
Computes the disparity map and left-right consistency mask.
### Compute_dep_and_pcl:
Computes the depth map and generates a 3D point cloud.
### postprocess: 
Applies post-processing steps, including background removal and z-range constraints.
### two_view: 
Executes the full pipeline for two input views.

## Execution
The main entry point is the main function in the script.
Load Middlebury dataset using load_middlebury_data from dataloader.py.
Call two_view with two views (images) and a kernel function (e.g., zncc_kernel).
The pipeline rectifies the views, computes the disparity map, and generates a 3D point cloud.
Visualization of camera poses and reconstructed 3D scene.
