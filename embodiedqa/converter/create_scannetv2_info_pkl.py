import numpy as np
import re
import os
import jsonlines
import json
import csv
import pickle
from glob import glob
import argparse
SCAN_RAW_BASE = None #From the original ScanNetv2 data
DSPNET_BASE = None #saved path
def is_box_visible(box_3d, intrinsic_matrix, extrinsic_matrix, image_shape):
    """
    Check if the 3D bounding box is visible in the image.

    :param box_3d: np.array of shape (6,), the 3D coordinates of the bounding box center (x, y, z) 
                    and the size in x, y, and z dimensions (x_size, y_size, z_size).
    :param intrinsic_matrix: np.array of shape (4, 4), the camera intrinsic matrix.
    :param extrinsic_matrix: np.array of shape (4, 4), the camera extrinsic matrix.
    :param image_shape: tuple, the shape of the image (height, width).
    :return: bool, True if any part of the bounding box is visible in the image, otherwise False.
    """
    intrinsic_matrix = intrinsic_matrix[:3,:3]
    # Extract box center and dimensions
    x, y, z, x_size, y_size, z_size = box_3d

    # Define the eight corners of the bounding box in 3D space
    corners = np.array([
        [x - x_size / 2, y - y_size / 2, z - z_size / 2],
        [x + x_size / 2, y - y_size / 2, z - z_size / 2],
        [x - x_size / 2, y + y_size / 2, z - z_size / 2],
        [x + x_size / 2, y + y_size / 2, z - z_size / 2],
        [x - x_size / 2, y - y_size / 2, z + z_size / 2],
        [x + x_size / 2, y - y_size / 2, z + z_size / 2],
        [x - x_size / 2, y + y_size / 2, z + z_size / 2],
        [x + x_size / 2, y + y_size / 2, z + z_size / 2]
    ])

    # Transform 3D box coordinates to camera coordinates
    box_3d_hom = np.hstack((corners, np.ones((corners.shape[0], 1))))
    box_cam_coords = (extrinsic_matrix @ box_3d_hom.T).T[:, :3]
    in_front_of_camera = box_cam_coords[:, 2] > 0
    # Project 3D points onto 2D image plane
    box_2d_hom = intrinsic_matrix @ box_cam_coords.T
    box_2d = (box_2d_hom[:2, :] / box_2d_hom[2, :]).T

    # Check if any of the points are within the image boundaries
    height, width = image_shape
    visible = np.any((box_2d[:, 0] >= 0) & (box_2d[:, 0] < width) &
                     (box_2d[:, 1] >= 0) & (box_2d[:, 1] < height) & 
                     in_front_of_camera) 

    return visible
def load_scannet(scan_ids, is_test = False):
    scans = {}
    # attribute
    # inst_labels, inst_locs, inst_colors, pcds
    aligned_flag = '' if is_test else '_aligned'
    for scan_id in scan_ids:
        if is_test:
            inst_ids = []
            inst_labels = []
            inst_locs = []
        else:
            bbox = np.load(os.path.join(SCAN_RAW_BASE, f'scannet_data/{scan_id}{aligned_flag}_bbox.npy'))
            inst_locs = bbox[:,:6]
            inst_labels = bbox[:,6].astype(np.int64) 
            inst_ids = bbox[:,7].astype(np.int64) 
        scans[scan_id] = {
            'inst_ids': inst_ids, #idx of instance
            'inst_labels': inst_labels,
            'inst_locs': inst_locs,     # (n_obj, 6) center xyz, whl
        }
    print("finish loading scannet data")
    return scans

def load_info_txt(path):
    axis_align_matrix = np.eye(4).astype(np.float32)
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('axisAlignment'):
                axis_align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(4, 4)
            # elif line.startswith('colorToDepthExtrinsics'):
            #     color2depth = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(4, 4)
            elif line.startswith('colorHeight'):
                h = int(line.strip().split()[-1])
            elif line.startswith('colorWidth'):
                w = int(line.strip().split()[-1])
            elif line.startswith('numColorFrames'):
                num_frams = int(line.strip().split()[-1])
    image_shape = (h,w)
    return axis_align_matrix, image_shape, num_frams

def read_categories_id_dict(filename, categories='nyu40class',ids='nyu40id'):
    assert os.path.isfile(filename)
    categories_id_dict = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            categories_id_dict[row[categories]] = int(row[ids])
    return categories_id_dict

#主要调用的函数
def format_scannetv2_to_mv3d(split='train'):
    assert split in ['train', 'val', 'test']
    
    # load category file
    cat2int = read_categories_id_dict(filename = os.path.join(SCAN_RAW_BASE, "meta_data/nyu40_labels.csv"),categories='nyu40class',ids='nyu40id')
    int2cat = {i:c for c,i in cat2int.items()}

    if split=='train':
        file_path = os.path.join(SCAN_RAW_BASE,'meta_data/scannetv2_train.txt')
        with open(file_path, 'r') as file:
            lines = file.readlines()
        scan_ids = [line.strip() for line in lines]
        posed_images_dir = 'posed_images'
    elif split=='val':
        file_path = os.path.join(SCAN_RAW_BASE,'meta_data/scannetv2_val.txt')
        with open(file_path, 'r') as file:
            lines = file.readlines()
        scan_ids = [line.strip() for line in lines]
        posed_images_dir = 'posed_images'
    elif split=='test':
        file_path = os.path.join(SCAN_RAW_BASE,'meta_data/scannetv2_test.txt')
        with open(file_path, 'r') as file:
            lines = file.readlines()
        scan_ids = [line.strip() for line in lines]
        posed_images_dir = 'posed_images_test'
    scans = load_scannet(scan_ids, is_test = (split=='test'))
    data = dict(metainfo={},data_list=[])

    data['metainfo']['categories'] = cat2int
    data['metainfo']['DATASET'] = 'SCANNET'
    for scan_id in scan_ids:
        if split == 'test':
            axis_align_matrix, image_shape, num_frams = load_info_txt(os.path.join(SCAN_RAW_BASE,f'scans_test/{scan_id}/{scan_id}.txt'))
        else:
            axis_align_matrix, image_shape, num_frams = load_info_txt(os.path.join(SCAN_RAW_BASE,f'scans/{scan_id}/{scan_id}.txt'))
        intrinsic = np.loadtxt(os.path.join(SCAN_RAW_BASE,f'{posed_images_dir}/{scan_id}/intrinsic.txt'))
        depth_intrinsic = np.loadtxt(os.path.join(SCAN_RAW_BASE,f'{posed_images_dir}/{scan_id}/depth_intrinsic.txt'))
        scan = scans[scan_id]
        instances = []
        
        #获取对齐统一世界的内参外参
        extrinsics = []
        img_path_list = []
        depth_path_list = []
        images_info = []
        for i in list(range(num_frams))[::10]:
            file_id = str(i).zfill(5)
            cam2global_txt_path = os.path.join(SCAN_RAW_BASE,f'{posed_images_dir}/{scan_id}/{file_id}.txt')
            if not os.path.isfile(cam2global_txt_path):
                print(f'{i}/{num_frams}: {cam2global_txt_path} is missing!')
                break
            cam2global_array = np.loadtxt(cam2global_txt_path)
            if np.isnan(cam2global_array).any() or np.isinf(cam2global_array).any():
                print(f'{i}/{num_frams}: cam2global array of {cam2global_txt_path} is nan or inf!')
                continue
            img_path = f"scannet/{posed_images_dir}/{scan_id}/{file_id}.jpg"
            depth_path = f"scannet/{posed_images_dir}/{scan_id}/{file_id}.png"
            align_global2cam = np.linalg.inv(axis_align_matrix @ cam2global_array)
            extrinsics.append(align_global2cam)
            image_info = dict(img_path = img_path,
                              cam2global = cam2global_array,
                              depth_path = depth_path,
                              visible_instance_ids = []
                              )
            images_info.append(image_info)
        
        for j in range(len(scan['inst_labels'])):
            ins = {}
            ins['bbox_3d'] = scan['inst_locs'][j].tolist()
            ins['bbox_label_3d'] = scan['inst_labels'][j] #start from 1
            ins['bbox_id'] = scan['inst_ids'][j] #start from 0
            instances.append(ins)
            for k, image_info in enumerate(images_info):
                is_visible = is_box_visible(scan['inst_locs'][j],extrinsic_matrix=extrinsics[k],intrinsic_matrix=intrinsic, image_shape=image_shape) #TODO: 考虑遮挡问题
                if is_visible:
                    image_info['visible_instance_ids'].append(j)
        data_item = dict(images = images_info,
                         instances = instances,
                         cam2img = intrinsic,
                         axis_align_matrix = axis_align_matrix,
                         depth_cam2img = depth_intrinsic,
                         sample_idx = f'scannet/{scan_id}')
        data['data_list'].append(data_item)
    with open(os.path.join(DSPNET_BASE, f'mv_scannetv2_infos_{split}.pkl'), 'wb') as file:
        pickle.dump(data, file)

if __name__=='__main__':
    #用户输入path参数
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_folder', type=str, default='./data/scannet')
    argparser.add_argument('--output_dir', type=str, default='./data')
    args = argparser.parse_args()
    SCAN_RAW_BASE = args.dataset_folder
    DSPNET_BASE = args.output_dir
    format_scannetv2_to_mv3d(split='train')
    format_scannetv2_to_mv3d(split='val')
    format_scannetv2_to_mv3d(split='test')

    