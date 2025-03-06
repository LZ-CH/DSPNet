import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmengine.model import BaseModel, BaseModule
from embodiedqa.registry import MODELS
from embodiedqa.utils.typing_config import SampleList

@MODELS.register_module()
class SituationPredictHead(BaseModule):
    def __init__(self, 
                 pos_dims: int = 3,
                 rot_dims: int = 4,
                 in_channels: int = 768,
                 hidden_channels: int = 768,
                 dropout: float = 0.,
                 loss_weight_pos = 1.0,
                 loss_weight_rot = 1.0,
                 aux_loss_weight = 1.0,
                 ):
        super().__init__()
        self.pos_dims = pos_dims
        self.rot_dims = rot_dims
        self.loss_weight_pos = loss_weight_pos
        self.loss_weight_rot = loss_weight_rot
        self.aux_loss_weight = aux_loss_weight
        self.reg_head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, pos_dims + rot_dims),
        )

    def loss(self, fusion_feat, batch_data_samples, **kwargs):
        gt_situation_label = [
            data_samples.gt_situation.situation_label
            for i, data_samples in enumerate(batch_data_samples) 
        ]
        batch_gt_situation_label = torch.stack(gt_situation_label)  # B, pos_dims + rot_dims

        logits = self.reg_head(fusion_feat)

        situation_predict_loss = self.aux_loss_weight * self.compute_situation_predict_loss(logits, batch_gt_situation_label)
        
        loss = dict(situation_predict_loss=situation_predict_loss)
        return loss

    def compute_situation_predict_loss(self, logits, batch_gt_situation_label):
        # Position loss
        loss_position = F.mse_loss(logits[:, :self.pos_dims], batch_gt_situation_label[:, :self.pos_dims], reduction="mean")

        if self.rot_dims == 6:
            # Predict 6D vectors
            pred_vectors = logits[:, self.pos_dims:].view(-1, 6)
            
            # Normalize predicted vectors to form orthogonal basis using Gram-Schmidt process
            pred_vec1 = F.normalize(pred_vectors[:, 0:3], dim=-1)  # First column vector
            pred_vec2 = pred_vectors[:, 3:6]  # Second vector, not normalized yet

            # Orthogonalize second vector against the first
            pred_vec2 = pred_vec2 - (pred_vec1 * pred_vec2).sum(dim=-1, keepdim=True) * pred_vec1
            pred_vec2 = F.normalize(pred_vec2, dim=-1)  # Normalize to unit length

            # Create predicted rotation matrix
            pred_rotation_matrix = torch.stack((pred_vec1, pred_vec2, torch.cross(pred_vec1, pred_vec2, dim=-1)), dim=-1)

            # Convert ground truth quaternion to rotation matrix
            gt_quaternions = batch_gt_situation_label[:, self.pos_dims:]  # Ground truth quaternion (_x, _y, _z, _w)
            gt_rotation_matrix = self.quaternion_to_rotation_matrix(gt_quaternions)

            # Compute the rotation loss (e.g., using Frobenius norm)
            loss_rotation = F.mse_loss(pred_rotation_matrix, gt_rotation_matrix, reduction="mean")

        elif self.rot_dims == 4:
            # Quaternion loss
            loss_quaternion = F.mse_loss(logits[:, self.pos_dims:], batch_gt_situation_label[:, self.pos_dims:], reduction="mean")

        # Combine position and rotation losses
        situation_predict_loss = (
            self.loss_weight_pos * loss_position + 
            self.loss_weight_rot * (loss_rotation if self.rot_dims == 6 else loss_quaternion)
        )
        return situation_predict_loss

    def quaternion_to_rotation_matrix(self, quaternions):
        # Normalize the quaternion to avoid scaling effects
        quaternions = F.normalize(quaternions, p=2, dim=-1)
        
        # Extract the values
        x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        # Compute the rotation matrix elements
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        # Form the rotation matrix
        rot_matrix = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
        ], dim=-1).view(-1, 3, 3)
        
        return rot_matrix