import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, rect_R_i, rect_R_j, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    rect_R_i,rect_R_j : [3,3]
        p_rect_left = rect_R_i @ p_i
        p_rect_right = rect_R_j @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ rect_R_i @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ rect_R_j @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding


    K_inv_corr_i = rect_R_i @ np.linalg.inv(K_i)
    K_inv_corr_j = rect_R_j @ np.linalg.inv(K_j)
    
    Homography_i = K_i_corr @ K_inv_corr_i
    Homography_j = K_j_corr @ K_inv_corr_j
    
    #Now warping the image
    rgb_i_rect = cv2.warpPerspective(rgb_i, Homography_i, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, Homography_j, (w_max, h_max))
    
    
    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w):
    """Computing the transformation that transforms the coordinate from j coordinate to i

    Parameters
    ----------
    i_R_w, j_R_w : [3,3]
    i_T_w, j_T_w : [3,1]
        p_i = i_R_w @ p_w + i_T_w
        p_j = j_R_w @ p_w + j_T_w
    Returns
    -------
    [3,3], [3,1], float
        p_i = i_R_j @ p_j + i_T_j, B is the baseline
    """

    
    # Extracting the rotation and translation matrix
    i_R_w = i_R_w[:3, :3]
    j_R_w = j_R_w[:3, :3]
    i_T_w = i_T_w[:3, 3].reshape(3, 1)
    j_T_w = j_T_w[:3, 3].reshape(3, 1)
    print(i_T_w.shape)
    print(j_T_w.shape)

    # Computing the transformation from j to i and the baseline
    j_R_w_inv = np.linalg.inv(j_R_w)
    i_R_j = i_R_w @ j_R_w_inv  # Use the inverse of j_R_w in the calculation
    i_T_j = i_T_w - i_R_j @ j_T_w
    B = np.linalg.norm(i_T_j)
    print(B)
   

    return i_R_j, i_T_j, B


def compute_rectification_R(i_T_j):
    """Computing the rectification Rotation

    Parameters
    ----------
    i_T_j : [3,1]

    Returns
    -------
    [3,3]
        p_rect = rect_R_i @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = i_T_j.squeeze(-1) / (i_T_j.squeeze(-1)[1] + EPS)

   
    row1_D = np.array([i_T_j[1,0], -i_T_j[0,0], 0])
    row1 = row1_D / (np.linalg.norm(row1_D) + EPS).flatten()
    row2 = (i_T_j / np.linalg.norm(i_T_j) + EPS).flatten()
    row3 = np.cross(row1.flatten(), row2.flatten())
    rect_R_i = np.vstack((row1, row2, row3))
        
    
    
   

    return rect_R_i


def ssd_kernel(src, dst):
    """ SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    
    # Computing the SSD error
    ssd = np.zeros((src.shape[0], dst.shape[0]))
    difference_in_patches = src[:,np.newaxis,:,:] - dst[np.newaxis,:,:,:] #performing the element wise subtraction across all the patches
    squre_norm = (np.linalg.norm(difference_in_patches, axis = 2)**2)
    ssd = np.sum(squre_norm, axis = 2)
    
    

    return ssd  # M,N


def sad_kernel(src, dst):
    """ SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

  
    
    pixel_difference = np.abs(src[:,np.newaxis,:,:] - dst[np.newaxis,:,:,:])
    sum_a3 = np.sum(pixel_difference, axis = 3)
    sad= np.sum(sum_a3, axis = 2)
    


    return sad  # M,N


def zncc_kernel(src, dst):
    """Computing the negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]


    src_mean = src.mean(axis = 1)
    dst_mean = dst.mean(axis = 1)
    
    src_norm = src - src_mean[:,np.newaxis,:]
    dst_norm = dst - dst_mean[:,np.newaxis,:]   
    
    sig_src = np.std(src_norm, axis = 1)
    sig_dst = np.std(dst_norm, axis = 1)
    
    numerator = src_norm[:,np.newaxis,:,:] * dst_norm[np.newaxis,:,:,:]
    denominator = (sig_src[:,np.newaxis,:] * sig_dst[np.newaxis,:,:])+EPS
    final_numerator = numerator.sum(axis = 2)
    zncc = (final_numerator / denominator).sum(axis = 2)

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """


    Red = np.pad(image[:,:,0].copy(),k_size//2)
    Green = np.pad(image[:,:,1].copy(),k_size//2)
    Blue = np.pad(image[:,:,2].copy(),k_size//2)
    
    
    Red_patch = np.lib.stride_tricks.as_strided(Red, shape = (image.shape[0], image.shape[1], k_size, k_size), strides = (Red.strides*2))   
    Green_patch = np.lib.stride_tricks.as_strided(Green, shape = (image.shape[0], image.shape[1], k_size, k_size), strides = (Green.strides*2))
    Blue_patch = np.lib.stride_tricks.as_strided(Blue, shape = (image.shape[0], image.shape[1], k_size, k_size), strides = (Blue.strides*2))
    
    reshape_Red = Red_patch.reshape(image.shape[0], image.shape[1], -1)
    reshape_Green = Green_patch.reshape(image.shape[0], image.shape[1], -1)
    reshape_Blue = Blue_patch.reshape(image.shape[0], image.shape[1], -1)
    
    patch_buffer = np.stack((reshape_Red, reshape_Green, reshape_Blue), axis = 3)
    
    
    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel,  img2patch_func=image2patch):
    """Computing the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func: function, optional
        the function used to compute the patch buffer, by default image2patch
        (there is NO NEED to alter this argument)

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """
    
    h, w = rgb_i.shape[:2]
    rgb_i = rgb_i.astype(np.float64)/255
    rgb_j = rgb_j.astype(np.float64)/255
    i_patch = img2patch_func(rgb_i, k_size)
    j_patch = img2patch_func(rgb_j, k_size)
    disp_map = np.zeros(((h,w)), dtype = np.float64)
    lr_consistency_mask = np.zeros(((h,w)), dtype = np.float64)
    
    #computing the disparity map
  
    vi = np.arange(h)
    vj = np.arange(h)
    disparity = vi[:,None] - vj[None,:] + d0
    mask = disparity > 0
    indexes = np.arange(0,h,1)
    
    for i in range(w):
        i_p_col, j_p_col = i_patch[:,i], j_patch[:,i]
        e = kernel_func(i_p_col, j_p_col)
        e_max = e.max() + 1
        e[~mask] = e_max
        
        min_index = (e.argmin(axis = 1)).flatten()
        col_min_index = e[:, min_index].argmin(axis = 0)
        
        lr_consistency_mask[:,i] = (vi == col_min_index).astype(np.float64)
        disp_map[:,i] = indexes - min_index + d0
        
    
    

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """
    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """


    # computing the depth map
    focal_length = K[1,1]
    dep_map = (focal_length * B) / (disp_map + EPS)
    
    #creating 3D array for backprojected point cloud coordiantes
    rows, cols = disp_map.shape
    xyz_cam = np.zeros((rows, cols, 3))
    
    # creatinh 2d grid
    pix_x, pix_y = np.meshgrid(np.arange(cols), np.arange(rows))
        
    #calculating the x and y coordinates of the backprojected points in camwra frame
    offset_pix_x = pix_x.flatten() - K[0,2]
    offset_pix_y = pix_y.flatten() - K[1,2]
    x = offset_pix_x * dep_map.flatten() / K[0,0]
    y = offset_pix_y * dep_map.flatten() / K[1,1]
    
    final = np.stack((x,y,dep_map.flatten())).T
    xyz_cam = final.reshape(rows, cols, 3) 
    
        
    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    c_R_w,
    c_T_w,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    given pcl_cam [N,3], c_R_w [3,3] and c_T_w [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    Homography_column = np.column_stack((c_R_w, c_T_w))
    H = np.vstack((Homography_column, np.array([0,0,0,1])))
    H_final = np.linalg.inv(H)
    
    Rot = H_final[:3,:3]
    Trans = H_final[:3,3]
    pcl_cam_transpose = pcl_cam.T
    pcl_world_1 = (np.matmul(Rot, pcl_cam_transpose)) + Trans.reshape(3,1)
    pcl_world = pcl_world_1.T
    

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    i_R_w, i_T_w = view_i["R"], view_i["T"][:, None]  # p_i = i_R_w @ p_w + i_T_w
    j_R_w, j_T_w = view_j["R"], view_j["T"][:, None]  # p_j = j_R_w @ p_w + j_T_w

    i_R_j, i_T_j, B = compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w)
    assert i_T_j[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    rect_R_i = compute_rectification_R(i_T_j)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        rect_R_i,
        rect_R_i @ i_R_j,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        rect_R_i @ i_R_w,
        rect_R_i @ i_T_w,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
