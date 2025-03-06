from transformers import CLIPTextModel, CLIPVisionModel, CLIPTokenizerFast,CLIPVisionConfig,CLIPTextConfig
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
import torch
import torch.nn as nn

@MODELS.register_module()
class CLIPVisionModelWrapper(BaseModule):
    def __init__(self, name="openai/clip-vit-base-patch16",out_channels=[64,128,256,512],used_hidden_layers=[12,12,12,12],frozen=True,use_img_feat_affine=False):
        super().__init__()
        assert len(out_channels)==len(used_hidden_layers)
        self.config = CLIPVisionConfig.from_pretrained(name)
        self.used_hidden_layers = used_hidden_layers
        self.config.output_hidden_states = not all([item==self.config.num_hidden_layers for item in used_hidden_layers])
        self.vision_model = CLIPVisionModel.from_pretrained(name,config = self.config)
        self.out_proj = nn.ModuleList([nn.Sequential(nn.Linear(self.config.hidden_size, oc, bias=True),
                                                nn.LayerNorm(oc)) for oc in out_channels],
                                      )
        if frozen:
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = False  
    def forward(self,img):
        if self.config.output_hidden_states:
            feats = self.vision_model(img).hidden_states
        else:
            feat = self.vision_model(img).last_hidden_state
        outs = []
        k = self.config.image_size//self.config.patch_size
        for i in range(len(self.out_proj)):
            if self.config.output_hidden_states:
                feat = feats[self.used_hidden_layers[i]]
            feat_proj = self.out_proj[i](feat[:,1:,:]) #B,L,D
            feat_proj = feat_proj.transpose(1,2) #B,D,L
            feat_proj = feat_proj.reshape(feat_proj.shape[0],feat_proj.shape[1],k,k) #B,D,H,W
            outs.append(feat_proj)
        out_dict = dict(layer_outputs=outs,pooler_output=feat[:, 0, :])
        return out_dict
        

@MODELS.register_module()
class CLIPTextModelWrapper(BaseModule):
    def __init__(self, name="openai/clip-vit-base-patch16", frozen=True):
        super().__init__()
        self.config = CLIPTextConfig.from_pretrained(name)
        self.text_model = CLIPTextModel.from_pretrained(name,config = self.config)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(name)
        self.frozen = frozen
        if frozen:
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False  
    def forward(self,*args,**kwargs):
        return self.text_model(*args,**kwargs)
    def get_tokenizer(self):
        return self.tokenizer
