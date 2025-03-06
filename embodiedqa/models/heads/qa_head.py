import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModel, BaseModule
from embodiedqa.registry import MODELS
from embodiedqa.utils.typing_config import SampleList
from embodiedqa.models.common import FC,MLP,AttFlat
eps = 1e-6    
@MODELS.register_module()
class QAHead(BaseModule):
    def __init__(self, 
                num_classes: int,
                in_channels: int = 768,
                hidden_channels: int = 768,
                glimpse: int = 1,
                dropout: float = 0.,
                only_use_fusion_feat_pooler: bool = False,
                cls_loss: dict = dict(type='BCEWithLogitsLoss',
                                      params=dict(reduction='sum'),
                                    )
                ):
        """
        Args:
            num_classes (int): The number of classes.
            in_channels (int, optional): The number of input channels. Defaults to 768.
            hidden_channels (int, optional): The number of hidden channels. Defaults to 768.
            glimpse (int, optional): The number of glimpses. Defaults to 1.
            dropout (float, optional): The dropout rate. Defaults to 0..
            only_use_fusion_feat_pooler (bool, optional): Whether to only use the fusion feature pooler. Defaults to False.
            cls_loss (dict, optional): The configuration of the classification loss. Defaults to dict(type='BCEWithLogitsLoss', params=dict(reduction='sum')).
        """
        super().__init__()
        self.only_use_fusion_feat_pooler = only_use_fusion_feat_pooler
        if not self.only_use_fusion_feat_pooler:
            self.attflat_visual = AttFlat(in_channels, in_channels//2, glimpse, in_channels, 0.1)
            self.attflat_lang = AttFlat(in_channels, in_channels//2, glimpse, in_channels, 0.1)
        # self.fusion_norm = nn.LayerNorm(in_channels)
        self.clf_head = nn.Sequential(
            nn.Linear((2-int(self.only_use_fusion_feat_pooler))*in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
        self.cls_loss = getattr(nn, cls_loss['type'])(**cls_loss['params'])
    def loss(self, fusion_feat_visual,visual_mask, fusion_feat_language, language_mask, batch_data_samples, fusion_feat_pooler, ret_fusion_feat=False, **kwargs):
        """
        Args:
            fusion_feat_visual (Tensor): The fusion feature visual Tensor.
            visual_mask (Tensor): The visual mask Tensor.
            fusion_feat_language (Tensor): The fusion feature language Tensor.
            language_mask (Tensor): The language mask Tensor.
            batch_data_samples (List[DataSample]): The batch of DataSamples.
            fusion_feat_pooler (Tensor): The fusion feature pooler Tensor.
            ret_fusion_feat (bool, optional): Whether to return the fusion feature. Defaults to False.

        Returns:
            Dict[str, Tensor]: The loss dict containing the qa cls loss, and optionally the fusion feature.
        """
        batch_gt_answer_labels = [
            data_samples.gt_answer.answer_labels
            for data_samples in batch_data_samples
        ]
        batch_gt_answer_labels = torch.stack(batch_gt_answer_labels).float()
        logits,fusion_feat = self.forward(fusion_feat_visual,visual_mask, fusion_feat_language, language_mask, fusion_feat_pooler, batch_data_samples)
        
        # qa_cls_loss = self.cls_loss(logits, batch_gt_answer_labels)/batch_gt_answer_labels.shape[0]
        
        # pred_score_log = F.log_softmax(logits,dim=1)
        # batch_gt_answer_labels = batch_gt_answer_labels/batch_gt_answer_labels.sum(1,keepdim=True).clamp(min=1)#B,C
        # qa_cls_loss = -(batch_gt_answer_labels*pred_score_log).sum()/logits.shape[0]
        qa_cls_loss = self.group_cross_entropy_loss(logits,batch_gt_answer_labels)
        
        loss = dict(qa_cls_loss=qa_cls_loss)
        if ret_fusion_feat:
            loss['fusion_feat']=fusion_feat
        return loss
    def group_cross_entropy_loss(self, logits, gt_answer_labels):
        pred_score = F.softmax(logits,dim=1)
        pred_score = (pred_score*gt_answer_labels).sum(1)#B,
        loss = -(pred_score + eps).log()
        return loss.mean()
    def predict(self, fusion_feat_visual,visual_mask, fusion_feat_language, language_mask, fusion_feat_pooler, batch_data_samples, **kwargs):
        logits,_ = self.forward(fusion_feat_visual,visual_mask, fusion_feat_language, language_mask,fusion_feat_pooler, batch_data_samples)
        pred_scores = F.softmax(logits,dim=1)
        return pred_scores
    def forward(self, fusion_feat_visual,visual_mask, fusion_feat_language, language_mask, fusion_feat_pooler, batch_data_samples, **kwargs):
        if self.only_use_fusion_feat_pooler:
            assert fusion_feat_pooler is not None
            fusion_feat = fusion_feat_pooler
        else:
            fusion_feat_visual = self.attflat_visual(fusion_feat_visual,visual_mask)
            fusion_feat_language = self.attflat_lang(fusion_feat_language,language_mask)
            fusion_feat = torch.cat([fusion_feat_visual,fusion_feat_language],dim=-1)
        logits = self.clf_head(fusion_feat)
        return logits, fusion_feat

