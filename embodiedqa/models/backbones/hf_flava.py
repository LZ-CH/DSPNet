from transformers import FlavaTextModel, FlavaImageModel, FlavaMultimodalModel, AutoProcessor, FlavaImageConfig, FlavaTextConfig, FlavaMultimodalConfig
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
import torch
import torch.nn as nn


@MODELS.register_module()
class FlavaVisionModelWrapper(BaseModule):
    def __init__(self, name="facebook/flava-full",out_channels=[64,128,256,512],add_map=False,used_hidden_layers=[12,12,12,12],frozen=True,use_img_feat_affine=False):
        super().__init__()
        assert len(out_channels)==len(used_hidden_layers)
        self.out_channels = out_channels
        self.config = FlavaImageConfig.from_pretrained(name)
        self.used_hidden_layers = used_hidden_layers
        self.config.output_hidden_states = not all([item==self.config.num_hidden_layers for item in used_hidden_layers])
        self.vision_model = FlavaImageModel.from_pretrained(name,config = self.config)
        self.out_proj = nn.ModuleList([nn.Sequential(nn.Linear(self.config.hidden_size, oc, bias=True),
                                                nn.LayerNorm(oc)) for oc in out_channels],
                                      )
        if frozen:
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = False  
    def forward(self,img):
        raw_out = self.vision_model(img)
        if self.config.output_hidden_states:
            feats = raw_out.hidden_states
        else:
            feat = raw_out.last_hidden_state
        outs = []
        k = self.config.image_size//self.config.patch_size
        for i in range(len(self.out_proj)):
            if self.config.output_hidden_states:
                feat = feats[self.used_hidden_layers[i]]
            feat_proj = self.out_proj[i](feat[:,1:,:]) #B,L,D
            feat_proj = feat_proj.transpose(1,2) #B,D,L
            feat_proj = feat_proj.reshape(feat_proj.shape[0],feat_proj.shape[1],k,k) #B,D,H,W
            outs.append(feat_proj)
        out_dict = dict(layer_outputs=outs,pooler_output=raw_out.pooler_output,raw_output=feat)
        return out_dict
        

@MODELS.register_module()
class FlavaTextModelWrapper(BaseModule):
    def __init__(self, name="facebook/flava-full", frozen=True):
        super().__init__()
        processor = AutoProcessor.from_pretrained(name)
        self.config = FlavaTextConfig.from_pretrained(name)
        self.text_model = FlavaTextModel.from_pretrained(name,config = self.config)
        self.tokenizer = processor.tokenizer
        self.frozen = frozen
        if frozen:
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False  
    def forward(self,*args,**kwargs):
        return self.text_model(*args,**kwargs)
    def get_tokenizer(self):
        return self.tokenizer


@MODELS.register_module()
class FlavaMultimodalModelWrapper(BaseModule):
    def __init__(self, name="facebook/flava-full", add_pooling_layer=True, frozen=True):
        super().__init__()
        # 加载配置
        self.config = FlavaMultimodalConfig.from_pretrained(name)
        
        # 加载 FlavaMultimodalModel
        self.model = FlavaMultimodalModel.from_pretrained(name, config=self.config, add_pooling_layer=add_pooling_layer)
        
        self.add_pooling_layer = add_pooling_layer
        self.frozen = frozen
        
        # 如果设置为冻结模型参数
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False  

    def forward(self, visual_feats, visual_attention_mask, lang_feats, lang_attention_mask, **kwargs):
        # 将视觉特征放在前面，语言特征放在后面进行拼接
        hidden_states = torch.cat([visual_feats, lang_feats], dim=1)
        
        # 拼接 attention_mask，视觉特征在前，语言特征在后
        attention_mask = torch.cat([visual_attention_mask, lang_attention_mask], dim=1)

        # 如果启用了 pooling layer，需要在 attention_mask 前添加一个 1
        if self.model.use_cls_token:
            cls_token_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
            attention_mask = torch.cat([cls_token_mask, attention_mask], dim=1)
        
        # 调用 FlavaMultimodalModel 的 forward 方法
        outputs = self.model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
        )
                
        if not self.model.use_cls_token:
            pooler_feat = None
            start_idx = 0
        else:
            # 拆分 CLS token 的输出和其他输出
            if self.add_pooling_layer:
                pooler_feat = outputs.pooler_output
            else:
                pooler_feat = outputs.last_hidden_state[:, 0, :]  # 取出 CLS token 的输出
            start_idx = 1
        
        visual_feats = outputs.last_hidden_state[:, start_idx:visual_feats.size(1)+start_idx, :]  # 拆分视觉特征输出
        lang_feats = outputs.last_hidden_state[:, visual_feats.size(1)+start_idx:, :]  # 拆分语言特征输出

        # 构建输出字典
        output = dict(
            pooler_feat=pooler_feat,
            visual_feats=visual_feats,
            lang_feats=lang_feats,
        )
        return output

@MODELS.register_module()
class FlavaMultimodalModelTwinWrapper(BaseModule):
    def __init__(self, name="facebook/flava-full", add_pooling_layer=True, frozen=False):
        super().__init__()
        # 加载配置
        self.config = FlavaMultimodalConfig.from_pretrained(name)
        
        # 加载 FlavaMultimodalModel
        self.model = FlavaMultimodalModel.from_pretrained(name, config=self.config, add_pooling_layer=add_pooling_layer)
        
        self.add_pooling_layer = add_pooling_layer
        self.frozen = frozen
        
        # 如果设置为冻结模型参数
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False  

    def forward(self, visual_feats, visual_attention_mask,visual_feats_2d, visual_attention_mask_2d, lang_feats, lang_attention_mask, **kwargs):
        
        if lang_attention_mask is None:
            lang_attention_mask = torch.ones((lang_feats.shape[0],lang_feats.shape[1]),device=lang_feats.device)
        if visual_attention_mask is None:
            visual_attention_mask = torch.ones((visual_feats.shape[0],visual_feats.shape[1]),device=visual_feats.device)
        if visual_attention_mask_2d is None:
            visual_attention_mask_2d = torch.ones((visual_feats_2d.shape[0],visual_feats_2d.shape[1]),device=visual_feats_2d.device)
        
        # # 将视觉特征放在前面，语言特征放在后面进行拼接
        # hidden_states = torch.cat([visual_feats,visual_feats_2d, lang_feats], dim=1)
        
        # # 拼接 attention_mask，视觉特征在前，语言特征在后
        # attention_mask = torch.cat([visual_attention_mask,visual_attention_mask_2d, lang_attention_mask], dim=1)

        # 将视觉特征放在前面，语言特征放在后面进行拼接
        hidden_states = torch.cat([visual_feats_2d, lang_feats], dim=1)
        
        # 拼接 attention_mask，视觉特征在前，语言特征在后
        attention_mask = torch.cat([visual_attention_mask_2d, lang_attention_mask], dim=1)

        # 如果启用了 pooling layer，需要在 attention_mask 前添加一个 1
        if self.model.use_cls_token:
            cls_token_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
            attention_mask = torch.cat([cls_token_mask, attention_mask], dim=1)
        
        # 调用 FlavaMultimodalModel 的 forward 方法
        outputs = self.model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
        )
                
        if not self.model.use_cls_token:
            pooler_feat = None
            start_idx = 0
        else:
            # 拆分 CLS token 的输出和其他输出
            if self.add_pooling_layer:
                pooler_feat = outputs.pooler_output
            else:
                pooler_feat = outputs.last_hidden_state[:, 0, :]  # 取出 CLS token 的输出
            start_idx = 1
        
        visual_feats = None
        visual_feats_2d = outputs.last_hidden_state[:, start_idx:visual_feats_2d.size(1)+start_idx, :]  # 拆分视觉特征输出
        lang_feats = outputs.last_hidden_state[:, visual_feats_2d.size(1)+start_idx:, :]  # 拆分语言特征输出

        # 构建输出字典
        output = dict(
            pooler_feat=pooler_feat,
            visual_feats=visual_feats,
            visual_feats_2d=visual_feats_2d,
            lang_feats=lang_feats,
        )
        return output