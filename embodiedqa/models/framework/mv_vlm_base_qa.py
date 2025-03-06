# Copyright (c) OpenRobotLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import time
from copy import deepcopy
from mmengine.model import BaseModel,BaseModule
from mmengine.structures import InstanceData
from mmcv.ops import furthest_point_sample,gather_points
from embodiedqa.models.layers.fusion_layers.point_fusion import (
    batch_point_sample, point_sample, batch_point_sample_in_visible)
from embodiedqa.registry import MODELS
from embodiedqa.structures.bbox_3d import get_proj_mat_by_coord_type
from embodiedqa.utils import ConfigType, OptConfigType
from embodiedqa.utils.typing_config import (ForwardResults, InstanceList,
                                              OptSampleList, SampleList)
from .vision_fusion import VisionFusion

class PositionEmbeddingLearned(BaseModule):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, embed_dims=768):

        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, embed_dims, kernel_size=1),
            nn.BatchNorm1d(embed_dims), nn.ReLU(inplace=True),
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding.transpose(1, 2).contiguous()
@MODELS.register_module()
class MultiViewVLMBase3DQA(BaseModel):
    """MultiViewVLMBase3DQA.

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        max_num_entities (int, optional): The maximum number of entities.
            Defaults to 256.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """
    _version = 1

    def __init__(self,
                 backbone: ConfigType,
                 backbone_text: ConfigType,
                 backbone_lidar: ConfigType,
                 backbone_fusion: ConfigType,
                 qa_head: ConfigType,
                 target_bbox_head: ConfigType = None,
                 target_cls_head: ConfigType = None,
                 situation_predict_head: ConfigType = None,
                 voxel_size: float = 0.01,
                 text_max_length: int = 512,
                 vision_num_queries: int = 256,
                 coord_type: str = 'CAMERA',
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 use_2d: bool = True,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone_lidar = MODELS.build(backbone_lidar)
        self.text_encoder = MODELS.build(backbone_text)
        self.tokenizer = self.text_encoder.get_tokenizer()
        self.use_2d = use_2d
        #MCGR
        self.fusion_encoder = MODELS.build(backbone_fusion)
        
        if self.use_2d:
            self.backbone = MODELS.build(backbone)
            #for TGMF
            self.text_global_att_proj = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size,self.fusion_encoder.config.hidden_size),
                                                    nn.LayerNorm(self.fusion_encoder.config.hidden_size))
            self.img_att_proj = nn.Sequential( nn.Linear(self.backbone.out_channels[-1],self.fusion_encoder.config.hidden_size),
                                            nn.LayerNorm(self.fusion_encoder.config.hidden_size))
            #ADVP
            self.fusion_map = VisionFusion(self.backbone_lidar.fp_channels[-1][-1],self.backbone.out_channels[-1],self.fusion_encoder.config.hidden_size)
            self.visual_feat_map = nn.Linear(self.fusion_encoder.config.hidden_size,self.fusion_encoder.config.hidden_size)
        else:
            self.visual_feat_map = nn.Linear(self.backbone_lidar.fp_channels[-1][-1],self.fusion_encoder.config.hidden_size)
        
        self.text_feat_map = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size,self.fusion_encoder.config.hidden_size),
                                           nn.LayerNorm(self.fusion_encoder.config.hidden_size)
                                           ) 
        
        self.pos_embedding = PositionEmbeddingLearned(3,self.fusion_encoder.config.hidden_size)
        
        self.fusion_encoder_visual_pre_norm = nn.Sequential(nn.LayerNorm(self.fusion_encoder.config.hidden_size),
                                                            nn.Dropout(self.fusion_encoder.config.hidden_dropout_prob)
                                                            )
        
        #dense visual feature
        self.full_visual_feat_map = deepcopy(self.visual_feat_map)
        self.full_pos_embedding = PositionEmbeddingLearned(3,self.fusion_encoder.config.hidden_size)
        self.fusion_encoder_full_visual_pre_norm = nn.Sequential(nn.LayerNorm(self.fusion_encoder.config.hidden_size),
                                                        nn.Dropout(self.fusion_encoder.config.hidden_dropout_prob)
                                                        )
        self.text_max_length = text_max_length
        self.vision_num_queries = vision_num_queries
        if target_bbox_head is not None:
            self.target_bbox_head = MODELS.build(target_bbox_head)
        if target_cls_head is not None:
            self.target_cls_head = MODELS.build(target_cls_head)
        if situation_predict_head is not None:
            self.situation_predict_head = MODELS.build(situation_predict_head)
        
        self.qa_head = MODELS.build(qa_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.coord_type = coord_type
        self.voxel_size = voxel_size
    @property
    def with_backbone(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'backbone') and self.backbone is not None
    @property
    def with_target_bbox_head(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'target_bbox_head') and self.target_bbox_head is not None
    @property
    def with_target_cls_head(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'target_cls_head') and self.target_cls_head is not None
    @property
    def with_situation_predict_head(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'situation_predict_head') and self.situation_predict_head is not None
    def extract_feat(
        self, batch_inputs_dict: Dict[str, Tensor], 
        batch_data_samples: SampleList,
        text_dict: Dict = None
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.

        Args:
            batch_inputs_dict (dict): The model input dict which includes
                'points' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.

        Returns:
            tuple[Tensor] | dict:  For outside 3D object detection, we
                typically obtain a tuple of features from the backbone + neck,
                and for inside 3D object detection, usually a dict containing
                features will be obtained.
        """
        points = batch_inputs_dict['points']
        stack_points = torch.stack(points)#B,N,6
        feat_dict = self.backbone_lidar(stack_points)
        
        if not self.use_2d:
            return feat_dict
        
        # extract img features for dual vision
        img = batch_inputs_dict['imgs']
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_size = img.shape[0]

        if len(img.shape) > 4:  # (B, n_views, C, H, W)
            img = img.reshape([-1] + list(img.shape)[2:])
            img_features_dict = self.backbone(img)# B*n_views,C,H,W
            img_features, img_global_features = img_features_dict['layer_outputs'],img_features_dict['pooler_output']
            img_features = [
                img_feat.reshape([batch_size, -1] + list(img_feat.shape)[1:])
                for img_feat in img_features
            ]
            img_global_features = img_global_features.reshape([batch_size, -1] + list(img_global_features.shape)[1:])#B,n,C
        else:
            img_features_dict = self.backbone(img)
            img_features,img_global_features = img_features_dict['layer_outputs'],img_features_dict['pooler_output']
        all_points_imgfeats = []
        img_feat_valid_flags = []
        points_imgfeats = []
        
        text_global_token = text_dict.get('text_global_token', None)
        text_global_features_for_att = self.text_global_att_proj(text_global_token)
        
        img_features_for_att = self.img_att_proj(img_features[-1].mean(dim=[-1,-2]))#B, n_views, C
        all_extrinsics = []
        for idx in range(len(batch_img_metas)):
            img_meta = batch_img_metas[idx]
              
            img_scale_factor = (img.new_tensor(img_meta['scale_factor'][:2])
                                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (img.new_tensor(img_meta['img_crop_offset'])
                               if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            assert 'extrinsic' in proj_mat.keys()
            assert 'intrinsic' in proj_mat.keys()
            projection = []
            # Support different intrinsic matrices for different images
            # if the original intrinsic is only a matrix
            # we will simply copy it to construct the intrinsic matrix list
            # in MultiViewPipeline
            assert isinstance(proj_mat['intrinsic'], list)
            for proj_idx in range(len(proj_mat['extrinsic'])):
                intrinsic = img.new_tensor(proj_mat['intrinsic'][proj_idx])
                extrinsic = img.new_tensor(proj_mat['extrinsic'][proj_idx])
                projection.append(intrinsic @ extrinsic)
            all_extrinsics.append(img.new_tensor(proj_mat['extrinsic']))#n_views,4,4
            proj_mat = torch.stack(projection)#n_views,4,4
            points_imgfeat, img_feat_valid_flag, img_feat_valid_flag_each = batch_point_sample_in_visible(# (N, C), (N,)
                img_meta,
                img_features=img_features[-1][idx],
                points=feat_dict['fp_xyz'][-1][idx],
                views_points = batch_data_samples[idx].views_points,
                voxel_size = self.voxel_size,
                proj_mat=proj_mat,
                coord_type=self.coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img.shape[-2:],
                img_shape=img_meta['img_shape'][:2],
                aligned=False,
                return_valid_flag=True,
                text_global_features_for_att=text_global_features_for_att[idx],
                img_features_for_att=img_features_for_att[idx])
            points_imgfeats.append(
                points_imgfeat)  # all sample
            img_feat_valid_flags.append(img_feat_valid_flag)# last_level

        points_imgfeats = torch.stack(points_imgfeats)#B,N,C
        img_feat_valid_flags = torch.stack(img_feat_valid_flags)#B,N
        all_extrinsics = torch.stack(all_extrinsics).to(points_imgfeats.device)#B,n_views,4,4
        feat_dict['fp_features'][-1] = self.fusion_map(feat_dict['fp_features'][-1].transpose(1,2),points_imgfeats).transpose(1,2)#B,C,N
        return feat_dict
    def extract_text_feat(
        self, batch_inputs_dict: Dict[str,
                                      Tensor], batch_data_samples: SampleList,):
        text_prompts = [
            data_samples.question for data_samples in batch_data_samples
        ]  # txt list
        tokenized = self.tokenizer.batch_encode_plus(
            text_prompts, padding='longest',max_length=self.text_max_length, truncation=True,
            return_tensors='pt').to(batch_inputs_dict['points'][0].device)
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_feat_map(encoded_text.last_hidden_state)
        text_token_mask = tokenized.attention_mask.bool()
        text_dict = dict(text_feats=text_feats,
                         text_token_mask=text_token_mask,
                         text_global_token=(text_feats*text_token_mask.unsqueeze(2)).sum(1)/text_token_mask.sum(1,keepdim=True)
                         )# (bs, max_text_length)
        return text_dict

    def forward_transformer(self,
                            point_feats: Tensor,
                            point_pos: Tensor,
                            point_mask:Tensor,
                            text_dict: Dict,
                            full_point_feats: Tensor = None,
                            full_point_pos: Tensor = None,
                            full_point_mask: Tensor = None
                            ) -> Dict:
        #feats: mapping and add pos embedding
        point_feats = self.visual_feat_map(point_feats)
        point_feats += self.pos_embedding(point_pos)
        point_feats = self.fusion_encoder_visual_pre_norm(point_feats)
        
        full_point_feats = self.full_visual_feat_map(full_point_feats)
        full_point_feats += self.full_pos_embedding(full_point_pos)
        full_point_feats = self.fusion_encoder_full_visual_pre_norm(full_point_feats)
        
        fusion_encoder_inputs_dict = dict(
            lang_feats = text_dict['text_feats'],
            lang_attention_mask = text_dict['text_token_mask'],
            visual_feats = point_feats,
            visual_attention_mask = point_mask,
            full_visual_feats = full_point_feats,
            full_visual_attention_mask = full_point_mask,
            )
        fusion_output = self.fusion_encoder(**fusion_encoder_inputs_dict)
        head_inputs_dict = dict(fusion_feat_visual=fusion_output['visual_feats'],
                                visual_mask=fusion_encoder_inputs_dict['visual_attention_mask'], 
                                fusion_feat_language=fusion_output['lang_feats'], 
                                language_mask=fusion_encoder_inputs_dict['lang_attention_mask'],
                                fusion_feat_pooler=fusion_output.get('pooler_feat',None)
                                )
        
        return head_inputs_dict

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:        
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples,text_dict=text_dict)

        points = batch_inputs_dict['points']
        batch_size = len(points)
        losses = {}

        full_point_feats = feat_dict['fp_features'][-1].transpose(1,2).contiguous() #B,seed_num,hidden_size
        full_point_pos = feat_dict['fp_xyz'][-1]
        point_mask = None #B,proposal_num

        fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries) #B,proposal_num
        point_feats = gather_points(full_point_feats.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,hidden_size
        point_pos = gather_points(full_point_pos.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,3

        head_inputs_dict = self.forward_transformer(point_feats=point_feats,
                                                    point_pos=point_pos,
                                                    point_mask=point_mask,
                                                    text_dict=text_dict,
                                                    full_point_feats=full_point_feats,
                                                    full_point_pos=full_point_pos)
        qa_losses = self.qa_head.loss(**head_inputs_dict,
                                     ret_fusion_feat=True,
                                     batch_data_samples=batch_data_samples)
        losses.update(qa_losses)
        
        if self.with_target_bbox_head:
            ref_loc_losses = self.target_bbox_head.loss(**head_inputs_dict,
                                                    points=points, 
                                                    aggregated_points=point_pos, 
                                                    batch_data_samples=batch_data_samples)
            losses.update(ref_loc_losses)
        if self.with_target_cls_head:
            fusion_feat = qa_losses['fusion_feat']
            ref_cls_loss = self.target_cls_head.loss(fusion_feat,batch_data_samples=batch_data_samples)
            losses.update(ref_cls_loss)
        if self.with_situation_predict_head:
            fusion_feat = qa_losses['fusion_feat']
            situation_predict_loss = self.situation_predict_head.loss(fusion_feat,batch_data_samples=batch_data_samples)
            losses.update(situation_predict_loss)  
        losses = self.loss_collect(losses)
        return losses
    def loss_collect(self, losses_dict):
        """
        Collects key-value pairs from the input dictionary where the key contains the word 'loss'.

        Args:
            losses_dict (dict): A dictionary containing various key-value pairs.

        Returns:
            dict: A dictionary containing only the key-value pairs where the key contains 'loss'.
        """
        # Create a new dictionary to store the key-value pairs with 'loss' in the key
        filtered_losses = {key: value for key, value in losses_dict.items() if 'loss' in key.lower()}

        return filtered_losses
    def add_prefix_to_loss_dict_keys(self, d, prefix):
        """
        给字典中的所有键添加指定前缀。

        Args:
            d (dict): 需要处理的字典。
            prefix (str): 要添加的前缀字符串。

        Returns:
            dict: 添加前缀后的新字典。
        """
        new_dict = {}
        for key, value in d.items():
            new_key = f"{prefix}{key}"
            if isinstance(value, dict):
                new_dict[new_key] = self.add_prefix_to_loss_dict_keys(value, prefix)
            else:
                if 'loss' in key:
                    new_dict[new_key] = value
                else:
                    new_dict[key] = value
        return new_dict
    
    def predict(self, batch_inputs_dict, batch_data_samples,**kwargs):
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples,text_dict=text_dict)
        full_point_feats = feat_dict['fp_features'][-1].transpose(1,2).contiguous() #B,seed_num,hidden_size
        full_point_pos = feat_dict['fp_xyz'][-1]
        batch_size = full_point_feats.shape[0]
        point_mask = None #B,proposal_num
        fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries) #B,proposal_num
        point_feats = gather_points(full_point_feats.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,hidden_size
        point_pos = gather_points(full_point_pos.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,3

        head_inputs_dict = self.forward_transformer(point_feats=point_feats,
                                                    point_pos=point_pos,
                                                    point_mask=point_mask,
                                                    text_dict=text_dict,
                                                    full_point_feats=full_point_feats,
                                                    full_point_pos=full_point_pos)
        results_list = self.qa_head.predict(**head_inputs_dict,
                                     batch_data_samples=batch_data_samples)

        for data_sample, pred_scores in zip(batch_data_samples,
                                                  results_list):
            data_sample.pred_scores = pred_scores
        return batch_data_samples
    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: Optional[List] = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: Optional[InstanceList] = None,
        data_instances_2d: Optional[InstanceList] = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples