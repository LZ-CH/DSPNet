import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS, TASK_UTILS
from embodiedqa.utils.typing_config import SampleList
from typing import Dict, List, Optional, Tuple, Union
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from embodiedqa.models.losses import chamfer_distance
from mmdet.models.utils import multi_apply

@MODELS.register_module()
class RefLocHead(BaseModule):
    def __init__(self, 
                 bbox_coder: Union[ConfigDict, dict],
                 train_cfg: Optional[dict] = None,
                 num_classes: int = 1,
                 in_channels: int = 768,
                 hidden_channels: int = 768,
                 dropout: float = 0.,
                 ref_loss: dict = dict(type='mmdet.CrossEntropyLoss',
                                       use_sigmoid=True),
                 loss_weight: float = 1.,
                 ):
        super().__init__()
        self.clf_head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
        self.ref_loss = MODELS.build(ref_loss)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.train_cfg = train_cfg
        self.loss_weight = loss_weight    

    def loss(self, fusion_feat_visual, visual_mask, 
             points: List[torch.Tensor], aggregated_points: List[torch.Tensor], 
             batch_data_samples: SampleList, **kwargs):
        # Generate targets using get_loc_target function
        batch_gt_instances_3d = [data_samples.gt_instances_3d for data_samples in batch_data_samples]
        target_labels, target_objectness, target_assignments, target_object_mask = self.get_loc_target(points, aggregated_points, batch_gt_instances_3d)
        # Compute ground truth labels using assignments
        batch_gt_labels = [
            data_samples.gt_instances_3d.target_objects_mask.long()[target_assignments[i]]
            for i, data_samples in enumerate(batch_data_samples)
        ]
        batch_gt_labels = torch.stack(batch_gt_labels)
        
        # Compute logits
        logits = self.clf_head(fusion_feat_visual).squeeze(-1)  # B, N

        if visual_mask is not None:
            logits = logits.masked_fill(~visual_mask.bool(),-1e6)
            target_objectness = target_objectness*visual_mask.float()
        # Compute the loss using internal helper function
        ref_loc_loss = self.loss_weight*self.group_cross_entropy_loss(logits, batch_gt_labels, target_objectness)
        
        loss = dict(ref_loc_loss=ref_loc_loss)
        return loss
    def predict(self, fusion_feat_visual, aggregated_points: List[torch.Tensor], **kwargs):
        batch_gt_labels = torch.stack(batch_gt_labels)
        # Compute logits
        logits = self.clf_head(fusion_feat_visual).squeeze(-1)  # B, N
        scores = F.softmax(logits,-1)
        # List to store coordinate sequences for each batch
        result_coordinates = []
        # Iterate over each batch
        for b in range(scores.shape[0]):
            # Get scores and points for the current batch
            batch_scores = scores[b]  # Shape: (N,)
            batch_points = aggregated_points[b]  # Shape: (N, 3)

            # Sort points and scores by score in descending order
            sorted_indices = torch.argsort(batch_scores, descending=True)
            sorted_scores = batch_scores[sorted_indices]
            sorted_points = batch_points[sorted_indices]

            # Find the subset of points where cumulative score exceeds 0.5
            cumulative_sum = torch.cumsum(sorted_scores, dim=0)
            cutoff_index = (cumulative_sum >= 0.5).nonzero(as_tuple=True)[0].item()

            # Get the subset of points up to the cutoff index
            selected_points = sorted_points[:cutoff_index + 1]
            result_coordinates.append(selected_points)

        return result_coordinates
    def group_cross_entropy_loss(self, logits, gt_labels, target_objectness_masks):
        batch_mask = (gt_labels.long() != 0).any(dim=1).float()  # B,
        pred_score = F.softmax(logits, dim=1)
        pred_score = (pred_score * gt_labels.float() * target_objectness_masks.float()).sum(1)  # B,
        loss = -(pred_score + 1e-6).log()
        loss = (loss * batch_mask).sum() / batch_mask.sum().clamp(1)
        return loss

    def get_loc_target(
        self,
        points: List[torch.Tensor],
        aggregated_points: List[torch.Tensor],
        batch_gt_instances_3d: List[InstanceData]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate location targets from points, aggregated points, and ground truth instances.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            aggregated_points (list[torch.Tensor]): Aggregated points from vote aggregation layer.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of gt_instances. 
                It usually includes ``bboxes`` and ``labels`` attributes.

        Returns:
            Tuple[torch.Tensor]: target_labels, target_objectness, target_assignments, target_object_mask.
        """
        # Extract gt labels and bboxes from batch_gt_instances_3d
        batch_gt_labels_3d = [gt.labels_3d for gt in batch_gt_instances_3d]
        batch_gt_bboxes_3d = [gt.bboxes_3d for gt in batch_gt_instances_3d]
        (target_labels, target_objectness, target_assignments, target_object_masks) = multi_apply(
                                                self._get_loc_targets_single,
                                                points, 
                                                batch_gt_bboxes_3d, 
                                                batch_gt_labels_3d, 
                                                aggregated_points
                                                )

        # # Pad targets as needed
        target_labels = torch.stack(target_labels)
        target_objectness = torch.stack(target_objectness)
        target_assignments = torch.stack(target_assignments)
        target_object_masks = torch.stack(target_object_masks)

        return target_labels, target_objectness, target_assignments, target_object_masks

    def _get_loc_targets_single(
        self,
        points: torch.Tensor,
        gt_bboxes_3d: InstanceData,
        gt_labels_3d: torch.Tensor,
        aggregated_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate location targets for a single batch.

        Args:
            points (torch.Tensor): Points of the batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes of the batch.
            gt_labels_3d (torch.Tensor): Labels of the batch.
            aggregated_points (torch.Tensor): Aggregated points from vote aggregation layer.

        Returns:
            Tuple[torch.Tensor]: target_labels, target_objectness, target_assignments, target_object_mask.
        """
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # Encode ground truth boxes and labels to center targets
        center_targets, size_class_targets, size_res_targets, dir_class_targets, dir_res_targets = \
            self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        # Compute Chamfer distance between aggregated points and center targets
        proposal_num = aggregated_points.shape[0]
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction='none')
        target_assignments = assignment.squeeze(0)
        euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

        # Determine objectness targets
        target_objectness = points.new_zeros((proposal_num), dtype=torch.long)
        target_objectness[euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1

        # Determine object mask (valid points based on distance thresholds)
        target_object_mask = points.new_zeros((proposal_num))
        target_object_mask[euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1.0
        target_object_mask[euclidean_distance1 > self.train_cfg['neg_distance_thr']] = 1.0

        # Determine mask targets
        target_labels = gt_labels_3d[target_assignments]

        return (target_labels.long(), target_objectness, target_assignments, target_object_mask)
