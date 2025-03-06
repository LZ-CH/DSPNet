from typing import List, Optional, Tuple, Union

import numpy as np
import mmcv
from mmcv.transforms import BaseTransform

from embodiedqa.registry import TRANSFORMS
from embodiedqa.structures.bbox_3d import points_cam2img, points_img2cam
from embodiedqa.structures.points import BasePoints, get_points_type
import torch
import torch.distributed as dist
import os
import time

# from mmcv.ops import furthest_point_sample
# def furthest_point_sample_cpu_compatible(coords, num_samples, device="cuda"):
#     """Ensure furthest point sample runs on CPU-compatible tensors."""
#     coords_tensor = torch.tensor(coords, device=device) # 移动到 GPU
#     with torch.no_grad():
#         sampled_indices = furthest_point_sample(coords_tensor[None, ...], num_samples)[0].cpu().numpy()  # 取回 CPU
#     return sampled_indices


def furthest_point_sample_cpu_compatible(points, n_samples, device="cpu"):
    """
    使用最小距离数组在CPU上优化最远点采样（FPS），优化距离计算、索引访问，提高运行效率。
    
    参数:
        points (np.ndarray): 输入点云，形状为(N, D)，N为点数，D为维度。
        n_samples (int): 需要采样的点数。
        
    返回:
        np.ndarray: 采样点的索引，形状为(n_samples,)。
    """
    n_points = points.shape[0]
    if n_samples > n_points:
        raise ValueError("n_samples 不能超过点云数量")
    
    selected_indices = np.zeros(n_samples, dtype=int)
    
    # 选择第一个为初始点
    initial_idx = 0
    selected_indices[0] = initial_idx

    # 初始化最小距离数组（平方距离）
    min_sq_distances = np.full(n_points, np.inf)
    
    # 计算初始点到所有点的平方距离
    diff = points - points[initial_idx]
    min_sq_distances[:] = np.einsum('ij,ij->i', diff, diff) 

    for i in range(1, n_samples):
        # 选择当前最远点（避免整体搜索，用分块策略优化 np.argmax）
        selected_idx = np.argmax(min_sq_distances)
        selected_indices[i] = selected_idx

        # 计算新点到所有点的平方距离
        new_point = points[selected_idx]
        diff = points - new_point
        new_sq_distances = np.einsum('ij,ij->i', diff, diff)

        # 仅更新更小的距离（避免整体更新）
        mask = new_sq_distances < min_sq_distances
        min_sq_distances[mask] = new_sq_distances[mask]
    
    return selected_indices


@TRANSFORMS.register_module()
class ConvertRGBDToPoints(BaseTransform):
    """Convert depth map to point clouds.

    Args:
        coord_type (str): The type of point coordinates. Defaults to 'CAMERA'.
        use_color (bool): Whether to use color as additional features
            when converting the image to points. Generally speaking, if False,
            only return xyz points. Otherwise, return xyzrgb points.
            Defaults to False.
    """

    def __init__(self,
                 coord_type: str = 'CAMERA',
                 use_color: bool = False) -> None:
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        self.use_color = use_color

    def transform(self, input_dict: dict) -> dict:
        """Call function to normalize color of points.

        Args:
            input_dict (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        depth_img = input_dict['depth_img']
        depth_cam2img = input_dict['depth_cam2img']
        ws = np.arange(depth_img.shape[1])
        hs = np.arange(depth_img.shape[0])
        us, vs = np.meshgrid(ws, hs)
        grid = np.stack(
            [us.astype(np.float32),
             vs.astype(np.float32), depth_img], axis=-1).reshape(-1, 3)
        nonzero_indices = depth_img.reshape(-1).nonzero()[0]
        grid3d = points_img2cam(grid, depth_cam2img)
        points = grid3d[nonzero_indices]
        attribute_dims = None
        if self.use_color:
            img = input_dict['img']
            img_resized = mmcv.imresize(img,(depth_img.shape[1],depth_img.shape[0]))
            rgb_points = img_resized[...,[2,1,0]].reshape(-1,3)
            rgb_points = rgb_points[nonzero_indices]
            points = np.concatenate([points, rgb_points], axis=-1)

            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(points,
                              points_dim=points.shape[-1],
                              attribute_dims=attribute_dims)
        input_dict['points'] = points

        return input_dict
@TRANSFORMS.register_module()
class LoadRGBToPoints(BaseTransform):
    """Convert depth map to point clouds.

    Args:
        coord_type (str): The type of point coordinates. Defaults to 'CAMERA'.
        use_color (bool): Whether to use color as additional features
            when converting the image to points. Generally speaking, if False,
            only return xyz points. Otherwise, return xyzrgb points.
            Defaults to False.
    """

    def __init__(self,
                 coord_type: str = 'CAMERA') -> None:
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type

    def transform(self, input_dict: dict) -> dict:
        """Call function to normalize color of points.

        Args:
            input_dict (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = input_dict['points'].coord.numpy()

        img = input_dict['img']
        h, w = img.shape[0], img.shape[1]
        cam2img = input_dict['cam2img']
        points2d = np.round(points_cam2img(points,
                                            cam2img)).astype(np.int32)
        us = np.clip(points2d[:, 0], a_min=0, a_max=w - 1)
        vs = np.clip(points2d[:, 1], a_min=0, a_max=h - 1)
        rgb_points = img[vs, us]
        points = np.concatenate([points, rgb_points], axis=-1)
        attribute_dims = dict()
        attribute_dims.update(
            dict(color=[
                points.shape[1] - 3,
                points.shape[1] - 2,
                points.shape[1] - 1,
            ]))
        points_class = get_points_type(self.coord_type)
        points = points_class(points,
                              points_dim=points.shape[-1],
                              attribute_dims=attribute_dims)
        input_dict['points'] = points

        return input_dict

@TRANSFORMS.register_module()
class PointSample(BaseTransform):
    """Point sample.

    Sampling data to a certain number.

    Required Keys:

    - points
    - pts_instance_mask (optional)
    - pts_semantic_mask (optional)

    Modified Keys:

    - points
    - pts_instance_mask (optional)
    - pts_semantic_mask (optional)

    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool): Whether the sampling is with or without replacement.
            Defaults to False.
    """

    def __init__(self,
                 num_points: int,
                 sample_range: Optional[float] = None,
                 replace: bool = False,
                 furthest_point_sampling: bool = False,
                 ) -> None:
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace
        self.furthest_point_sampling=furthest_point_sampling
        if self.furthest_point_sampling:
            print("using furthest point sampling in cpu will be so slow, you can set furthest point sampling in data_preprocessors where you can use cuda.")
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = f"cuda:{local_rank}"
    def transform(self, input_dict: dict) -> dict:
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']

        # if the depth map is all-zero
        if len(points) == 0:
            return input_dict
        if self.furthest_point_sampling:
            
            a = time.time()
            points, choices = self._points_furthest_sampling(points,
                                                             self.num_points,
                                                             return_choices=True)
            b = time.time()
            print("time:",b-a)
        else:
            points, choices = self._points_random_sampling(points,
                                                        self.num_points,
                                                        self.sample_range,
                                                        self.replace,
                                                        return_choices=True)
        input_dict['points'] = points

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            input_dict['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        return input_dict

    def _points_random_sampling(
        self,
        points: BasePoints,
        num_samples: Union[int, float],
        sample_range: Optional[float] = None,
        replace: bool = False,
        return_choices: bool = False
    ) -> Union[Tuple[BasePoints, np.ndarray], BasePoints]:
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int, float): Number of samples to be sampled. If
                float, we sample random fraction of points from num_points
                to 100%.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool): Sampling with or without replacement.
                Defaults to False.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:

                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if isinstance(num_samples, float):
            assert num_samples < 1
            num_samples = int(
                np.random.uniform(self.num_points, 1.) *
                points.shape[0])  # TODO: confusion

        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            dist = np.linalg.norm(points.coord.numpy(), axis=1)
            far_inds = np.where(dist >= sample_range)[0]
            near_inds = np.where(dist < sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(far_inds,
                                            num_samples,
                                            replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]
    def _points_furthest_sampling(
        self,
        points: BasePoints,
        num_samples: int,
        return_choices: bool = False
    ) -> Union[Tuple[BasePoints, np.ndarray], BasePoints]:
        """Points furthest point sampling.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int): Number of points to sample.
            return_choices (bool): Whether to return sampled indices. Defaults to False.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:
                - points (:obj:`BasePoints`): Sampled 3D Points.
                - choices (np.ndarray, optional): The sampled indices.
        """
        assert num_samples > 0, "num_samples must be positive."
        # Convert points to numpy array if necessary
        coords = points.coord.numpy()
        
        # If num_samples exceeds available points, duplicate the points
        if num_samples > points.shape[0]:
            sampled_indices = np.arange(points.shape[0])
            repeat_factor = (num_samples // points.shape[0]) + 1
            sampled_indices = np.tile(sampled_indices, repeat_factor)[:num_samples]
        else:
            # Apply furthest point sampling
            sampled_indices = furthest_point_sample_cpu_compatible(coords, num_samples, self.device)
        
        if return_choices:
            return points[sampled_indices], sampled_indices
        else:
            return points[sampled_indices]
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' sample_range={self.sample_range},'
        repr_str += f' replace={self.replace})'

        return repr_str


@TRANSFORMS.register_module()
class PointsRangeFilter(BaseTransform):
    """Filter points by the range.

    Required Keys:

    - points
    - pts_instance_mask (optional)

    Modified Keys:

    - points
    - pts_instance_mask (optional)

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range: List[float]) -> None:
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict) -> dict:
        """Transform function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        if len(clean_points) < 100:
            print('Warning: <100 points after PointsRangeFilter and',
                  'so we keep the original points!')
        else:
            input_dict['points'] = clean_points
            points_mask = points_mask.numpy()

            pts_instance_mask = input_dict.get('pts_instance_mask', None)
            pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

            if pts_instance_mask is not None:
                input_dict['pts_instance_mask'] = pts_instance_mask[
                    points_mask]

            if pts_semantic_mask is not None:
                input_dict['pts_semantic_mask'] = pts_semantic_mask[
                    points_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str
