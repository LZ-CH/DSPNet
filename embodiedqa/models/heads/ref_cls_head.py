import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule
from embodiedqa.registry import MODELS
from embodiedqa.utils.typing_config import SampleList

@MODELS.register_module()
class RefClsHead(BaseModule):
    def __init__(self, 
                num_classes: int = 18,
                in_channels: int = 768,
                hidden_channels: int = 768,
                dropout: float = 0.,
                ref_loss: dict = dict(type='mmdet.CrossEntropyLoss',
                                          use_sigmoid=True),
                loss_weight: float = 1.,
                ):
        super().__init__()
        self.num_classes = num_classes
        self.clf_head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
        self.ref_loss = MODELS.build(ref_loss)
        self.loss_weight = loss_weight
    def loss(self, fusion_feat, batch_data_samples, **kwargs):
        target_object_labels = [
            (data_samples.gt_instances_3d.labels_3d[data_samples.gt_instances_3d.target_objects_mask.bool()]).unique()
            for i,data_samples in enumerate(batch_data_samples) 
        ]
        batch_gt_labels = torch.zeros((fusion_feat.shape[0],self.num_classes),device=fusion_feat.device)
        for i in range(batch_gt_labels.shape[0]):
            batch_gt_labels[i][target_object_labels[i]]=1

        logits = self.clf_head(fusion_feat)#B,cls
        
        # ref_cls_loss = self.ref_loss(logits, batch_gt_labels)/batch_gt_labels.shape[0]
        
        # pred_score_log = F.log_softmax(logits,dim=1)
        # batch_gt_labels = batch_gt_labels/batch_gt_labels.sum(1,keepdim=True).clamp(min=1)#B,C
        # ref_cls_loss = -(batch_gt_labels*pred_score_log).sum()/logits.shape[0]
        ref_cls_loss = self.loss_weight*self.group_cross_entropy_loss(logits,batch_gt_labels.float())
        
        loss = dict(ref_cls_loss=ref_cls_loss)
        return loss
    def group_cross_entropy_loss(self, logits, gt_labels):
        batch_mask = (gt_labels.long() != 0).any(dim=1).float()#B,
        pred_score = F.softmax(logits,dim=1)
        pred_score = (pred_score*gt_labels).sum(1)#B,
        loss = -(pred_score + 1e-6).log()
        loss = (loss*batch_mask).sum()/batch_mask.sum().clamp(1)
        return loss