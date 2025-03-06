import os
import pickle
from argparse import ArgumentParser
import cv2
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from embodiedqa.visualization.utils import (_9dof_to_box, _box_add_thickness, draw_camera,
                    from_depth_to_point)
from tqdm import tqdm
from embodiedqa.structures.bbox_3d import (batch_points_cam2img,
                                             get_proj_mat_by_coord_type,
                                             points_cam2img, points_img2cam)
from copy import deepcopy
from scipy.ndimage import map_coordinates
#固定随机数种子
np.random.seed(42)


def sample_colors_from_image(points_camera, image):

    h, w, _ = image.shape


    uv_coords = points_camera[:, :2].astype(np.int32)


    uv_coords[:, 0] = np.clip(uv_coords[:, 0], 0, w - 1)  # u
    uv_coords[:, 1] = np.clip(uv_coords[:, 1], 0, h - 1)  # v

    points_color = image[uv_coords[:, 1], uv_coords[:, 0]]/255.0

    return points_color
class RGBD2PointClounds:
    """Visualization tool for Continuous 3D Object Detection task.

    This class serves as the API for visualizing Continuous 3D Object
    Detection task.

    Args:
        dir (str): Root path of the dataset.
        scene (dict): Annotation of the selected scene.
        pcd_downsample (int) : The rate of downsample.
    """

    def __init__(self, dir, scene, pcd_downsample):
        self.dir = dir
        self.scene = scene
        self.downsample = pcd_downsample
    def save(self, out_file=None):
        points_list = []
        colors_list = []
        ids_full = range(len(self.scene['images']))
        ids = [int(i*len(self.scene['images'])/20) for i in range(20)]
        extrinsics = []
        intrinsics = []
        imgs = []
        depth_imgs = []
        for idx in tqdm(ids_full):
            img = self.scene['images'][idx]
            img_path = img['img_path']
            img_path = os.path.join(self.dir, img_path)
            depth_path = img['depth_path']
            depth_path = os.path.join(self.dir,
                                    depth_path)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            rgb_img = cv2.imread(img_path)
            
            rgb_img = rgb_img[:, :, ::-1]
            imgs.append(rgb_img)
            depth_imgs.append(depth_img)
            axis_align_matrix = self.scene['axis_align_matrix']
            extrinsic = axis_align_matrix @ img['cam2global']
            extrinsics.append(extrinsic)
            if 'cam2img' in img:
                intrinsic = img['cam2img']
            else:
                intrinsic = self.scene['cam2img']
            intrinsics.append(intrinsic)
            if 'depth_cam2img' in img:
                depth_intrinsic = img['depth_cam2img']
            else:
                depth_intrinsic = self.scene['depth_cam2img']
            depth_shift = 1000.0
            
            mask = (depth_img > 0).flatten()
            depth_img = depth_img.astype(np.float32) / depth_shift
            
            #点云对齐
            points, colors = from_depth_to_point(rgb_img, depth_img, mask,
                                                intrinsic, depth_intrinsic,
                                                extrinsic)
            points_list.append(points)
            colors_list.append(colors)

        points = np.concatenate(points_list,axis=0)
        colors = np.concatenate(colors_list,axis=0)
        pc = o3d.geometry.PointCloud()
        
        scene_name = self.dir.split('/')[-1]
        if os.path.exists(os.path.join(self.dir,f'{scene_name}_aligned_vert.npy')):
            vert = np.load(os.path.join(self.dir,f'{scene_name}_aligned_vert.npy'))
            points = vert[:,:3]
            colors = vert[:,3:6]/255.0
            
        sample_idx = np.random.choice(points.shape[0], 50000, replace=False)
        points = points[sample_idx]
        colors = colors[sample_idx]
        key_idx = np.random.choice(points.shape[0], 256, replace=False)
        colors[key_idx] = [1,0,0]
        pc.points = o3d.utility.Vector3dVector(deepcopy(points))
        pc.colors = o3d.utility.Vector3dVector(deepcopy(colors))
        # save pc to .ply file
        if out_file is None:
            out_file = os.path.join(self.dir,'clean_pointclounds.ply')
        o3d.io.write_point_cloud(out_file,pc)
        exit()
        valid_masks = []
        agg_color = np.zeros_like(colors)
        invalid_color = [1.0,0,0]
        for i, idx in enumerate(tqdm(ids)):
            img = self.scene['images'][idx]
            img_path = img['img_path']
            base_name = os.path.basename(img_path).replace('.jpg','.ply')
            sub_out_file = os.path.join(self.dir, base_name)
            temp_pc = deepcopy(pc)
            points_camera = np.asarray(temp_pc.transform(self.scene['depth2img']['intrinsic']@self.scene['depth2img']['extrinsic'][idx]).points)
            points_camera[:,:2] = points_camera[:,:2]/points_camera[:,2:3]
            h,w,_ = imgs[idx].shape
            valid_mask = (points_camera[:,2] > 0)&(points_camera[:,2] < depth_imgs[idx].max())&(points_camera[:,0] > 0)&(points_camera[:,0] < w) & (points_camera[:,1] > 0)&(points_camera[:,1] < h)
            valid_masks.append(valid_mask)
            temp_pc = o3d.geometry.PointCloud()
            temp_color = sample_colors_from_image(points_camera, imgs[idx])
            temp_color[~valid_mask] =0.
            agg_color += temp_color
            temp_color[~valid_mask] = invalid_color
            temp_pc.colors = o3d.utility.Vector3dVector(temp_color)
            temp_pc.points = o3d.utility.Vector3dVector(points)
            # o3d.io.write_point_cloud(sub_out_file,temp_pc)
        valid_num = np.stack(valid_masks).sum(0)
        valid_mask = valid_num > 0
        temp_pc = o3d.geometry.PointCloud()
        temp_color = agg_color/np.clip(valid_num[...,None],a_min=1,a_max=None)
        temp_color[~valid_mask] = invalid_color
        temp_pc.colors = o3d.utility.Vector3dVector(temp_color)
        temp_pc.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(self.dir,'aggregate_pointclounds.ply'),temp_pc)
def main(args):
    data_dir = args.scene_dir
    with open(os.path.join(data_dir, 'poses.txt'), 'r') as f:
        poses = f.readlines()
    axis_align_matrix_path = os.path.join(data_dir, 'axis_align_matrix.txt')
    if os.path.isfile(axis_align_matrix_path):
        axis_align_matrix = np.loadtxt(axis_align_matrix_path)
    else:
        axis_align_matrix = np.eye(4).astype(np.float32)
    intrinsic = np.loadtxt(os.path.join(data_dir, 'intrinsic.txt'))#cam2img
    intrinsic = intrinsic.astype(np.float32)
    
    if os.path.isfile(os.path.join(data_dir, 'depth_intrinsic.txt')):
        depth_intrinsic = np.loadtxt(os.path.join(data_dir, 'depth_intrinsic.txt'))#depth_cam2img
        depth_intrinsic = depth_intrinsic.astype(np.float32)
    else:
        depth_intrinsic = intrinsic


    select_num = 20
    n_frames = len(poses)
    stride = n_frames//select_num
    scene = dict(
        axis_align_matrix=axis_align_matrix,
        images=[],
        img_path=[],
        depth_img_path=[],
        depth2img=dict(extrinsic=[],
                       intrinsic=intrinsic,
                       origin=np.array([.0, .0, .5]).astype(np.float32)),
        depth_cam2img=depth_intrinsic,
        depth_shift=1000.0,
        cam2img=intrinsic)
    idx = range(1,n_frames,stride)
    for i in idx:
        timestamp, x, y, z, qx, qy, qz, qw = poses[i].split()#poses of camera
        x, y, z, qx, qy, qz, qw = float(x), float(y), float(z), float(
            qx), float(qy), float(qz), float(qw)
        rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
        transform_matrix = np.identity(4)
        transform_matrix[:3, :3] = rot_matrix @ [[0, 0, 1], [-1, 0, 0],
                                                 [0, -1, 0]]
        transform_matrix[:3, 3] = [x, y, z]  # CAM to NOT ALIGNED GLOBAL

        #scene info
        image_ann = dict(img_path=os.path.join('rgb',timestamp + '.jpg'),
                         depth_path=os.path.join('depth',timestamp + '.png'),
                         cam2global=transform_matrix,
                         cam2img=intrinsic)
        scene['images'].append(image_ann)
        scene['img_path'].append(
            os.path.join('rgb', timestamp + '.jpg'))
        scene['depth_img_path'].append(
            os.path.join('depth', timestamp + '.png'))
        align_global2cam = np.linalg.inv(axis_align_matrix @ transform_matrix)
        scene['depth2img']['extrinsic'].append(
            align_global2cam.astype(np.float32))
    rgb2pc = RGBD2PointClounds(dir=data_dir, scene=scene, pcd_downsample=500)
    rgb2pc.save()
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene_dir',
                        type=str,
                        required=True,
                        help='Demo data directory')
    args = parser.parse_args()
    main(args)
