import json
import copy
import argparse
from easydict import EasyDict as edict
from models.PointDSC import PointDSC
from utils.pointcloud import estimate_normal
import torch
import numpy as np
import open3d as o3d 
import os, cv2

def extract_fcgf_features(pcd_path, downsample, device, weight_path='misc/ResUNetBN2C-feat32-3dmatch-v0.05.pth'):
    raw_src_pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.array(raw_src_pcd.points)
    from misc.fcgf import ResUNetBN2C as FCGF
    from misc.cal_fcgf import extract_features
    fcgf_model = FCGF(
        1,
        32,
        bn_momentum=0.05,
        conv1_kernel_size=7,
        normalize_feature=True
    ).to(device)
    checkpoint = torch.load(weight_path)
    fcgf_model.load_state_dict(checkpoint['state_dict'])
    fcgf_model.eval()

    xyz_down, features = extract_features(
        fcgf_model,
        xyz=pts,
        rgb=None,
        normal=None,
        voxel_size=downsample,
        skip_check=True,
    )
    return raw_src_pcd, xyz_down.astype(np.float32), features.detach().cpu().numpy()

def extract_fpfh_features(pcd_path, downsample, device):
    raw_src_pcd = o3d.io.read_point_cloud(pcd_path)
    estimate_normal(raw_src_pcd, radius=downsample*2)
    src_pcd = raw_src_pcd.voxel_down_sample(downsample) # 降采样
    src_features = o3d.pipelines.registration.compute_fpfh_feature(src_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=downsample * 5, max_nn=100))
    src_features = np.array(src_features.data).T
    src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
    return raw_src_pcd, np.array(src_pcd.points), src_features

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    polygon_points = np.concatenate([np.asarray(source_temp.points), np.asarray(target_temp.points)], axis = 0)
    num_kp = len(source_temp.points)
    lines = [[idx, idx + num_kp] for idx in range(num_kp)]
    color = [[1, 0, 0] for i in range(num_kp)] 

    #绘制线条
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color) #线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    o3d.visualization.draw_geometries([source_temp, target_temp, lines_pcd])

def find_match_pcd(pcd1, pcd2, downsample, descriptor:str, device, save_path):
    # extract features
    if descriptor == 'fpfh':
        raw_src_pcd, src_pts, src_features = extract_fpfh_features(pcd1, downsample, device)
        raw_tgt_pcd, tgt_pts, tgt_features = extract_fpfh_features(pcd2, downsample, device)
    else:
        raw_src_pcd, src_pts, src_features = extract_fcgf_features(pcd1, downsample, device)
        raw_tgt_pcd, tgt_pts, tgt_features = extract_fcgf_features(pcd2, downsample, device)
    # matching
    distance = np.sqrt(2 - 2 * (src_features @ tgt_features.T) + 1e-6)
    source_idx = np.argmin(distance, axis=1)
    corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
    src_keypts = src_pts[corr[:,0]]
    tgt_keypts = tgt_pts[corr[:,1]]

    src_kp_o3d = o3d.geometry.PointCloud()    
    src_kp_o3d.points = o3d.utility.Vector3dVector(src_keypts)      
    tgt_kp_o3d = o3d.geometry.PointCloud() 
    tgt_kp_o3d.points = o3d.utility.Vector3dVector(tgt_keypts)
    o3d.io.write_point_cloud(f"{save_path}_0.ply", src_kp_o3d)
    o3d.io.write_point_cloud(f"{save_path}_1.ply", tgt_kp_o3d)
    # transformation = np.eye(4)
    # draw_registration_result(src_kp_o3d, tgt_kp_o3d, transformation)

    # transformation = procrustes_align(src_keypts, tgt_keypts)
    # draw_registration_result(src_kp_o3d, tgt_kp_o3d, transformation)

    return src_kp_o3d, tgt_kp_o3d

def procrustes_align(pc_x, pc_y):
    """
    calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
    :param pc_x: Nx3 input point cloud
    :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
    :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
    """
    transformation = np.eye(4)
    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    #* My implementation starts here ###############
    # 1. get centered pc_x and centered pc_y
    # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
    # 3. estimate rotation
    # 4. estimate translation
    # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)
    center_x = np.mean(pc_x,axis=0)
    center_y = np.mean(pc_y,axis=0)
    X = (pc_x-center_x).T # 3*N
    Y = (pc_y-center_y).T
    U, _, V = np.linalg.svd(Y@X.T)
    if np.isclose(np.linalg.det(U)*np.linalg.det(V.T), 1):
        S=np.eye(U.shape[0], U.shape[1])
    else:
        diag = np.ones(U.shape[0])
        diag[-1] = -1
        S=np.diag(diag)
    R = U @ S @ V
    t = - R @ center_x + center_y
    #* My implementation ends here ###############

    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
    print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())
    transformation[:3,:3] = R
    transformation[:3,3] = t
    return transformation


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptor', default='fcgf', type=str, choices=['fcgf', 'fpfh'])
    args = parser.parse_args()
    config_path = f'snapshot/PointDSC_3DMatch_release/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    groud_truth_root_dir = "/home/wbh/datasets/3DMatch/gt_result"
    output_root_dir = "/home/wbh/datasets/3DMatch/preprocess"
    scene_root_dir = "/home/wbh/datasets/3DMatch/fragments"
    for scene in os.listdir(scene_root_dir):
        pcd_dir = os.path.join(scene_root_dir, scene)
        pcd_files = os.listdir(pcd_dir)
        pcd_files = sorted(pcd_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        num_pcd_files = len(pcd_files)
        gt_file = os.path.join(groud_truth_root_dir, f"{scene}-evaluation","gt.log")
        dict_gt = {}
        current_key = ''
        with open(gt_file, 'r') as gtf:
            for idx, line in enumerate(gtf):
                if idx%5==0:
                    tmp_list = line.strip("\n").split()
                    current_key = f"{tmp_list[0]}-{tmp_list[1]}"
                    dict_gt[current_key]=np.eye(4)
                else:
                    tmp_list = line.strip("\n").split()
                    temp_npy = np.zeros([1,4])
                    for i in range(4):
                        temp_npy[0, i] = float(tmp_list[i])
                    dict_gt[current_key][idx%5-1,:] = temp_npy

        for i in range(1, num_pcd_files):
            for j in range(i):
                if f"{j}-{i}" in dict_gt.keys():
                    gt_ = dict_gt[f"{j}-{i}"]
                    print(f"gt {j}-{i}: ", gt_)
                else: 
                    continue
                pcd1 = os.path.join(pcd_dir, pcd_files[j])
                pcd2 = os.path.join(pcd_dir, pcd_files[i])
                save_path = os.path.join(output_root_dir, scene, args.descriptor)
                cv_file = cv2.FileStorage(f"{save_path}/{j}_{i}.yaml", cv2.FILE_STORAGE_WRITE)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                src_kp_o3d, tgt_kp_o3d = find_match_pcd(pcd1, pcd2, config.downsample, \
                    args.descriptor, device, os.path.join(save_path, f"{j}_{i}"))
                cv_file.write("transform", gt_)
                # draw_registration_result(tgt_kp_o3d, src_kp_o3d, gt_)

            if i>2:
                break
            


    


