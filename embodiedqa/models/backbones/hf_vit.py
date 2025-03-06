from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
'''
google/vit-base-patch16-224-in21k
google/vit-base-patch16-224
microsoft/beit-base-patch16-224-pt22k-ft22k
facebook/dinov2-base
'''

@MODELS.register_module()
class ViTModelWrapper(BaseModule):
    def __init__(self, name="google/vit-base-patch16-224", out_channels=[64, 128, 256, 512], add_map=False, used_hidden_layers=[12, 12, 12, 12], frozen=True, learnable_parameter_keys=[]):
        """
        Initialize a ViTModelWrapper.

        Args:
            name (str): The name of the transformer model to use.
            out_channels (list[int]): The output channels of the model.
            add_map (bool): Whether to add a map to the output.
            used_hidden_layers (list[int]): The layers to use for the output.
            frozen (bool): Whether to freeze the model parameters.
            learnable_parameter_keys (list[str]): The keys of the parameters
                that are set to be learnable.
        """
        super().__init__()
        assert len(out_channels) == len(used_hidden_layers)
        self.out_channels = out_channels
        self.config = AutoConfig.from_pretrained(name)
        self.used_hidden_layers = used_hidden_layers
        self.config.output_hidden_states = not all([item==self.config.num_hidden_layers for item in used_hidden_layers])
        self.vision_model = AutoModel.from_pretrained(name, config=self.config)
        self.add_map = add_map
        if self.add_map:
            self.out_proj = nn.ModuleList([nn.Sequential(nn.Linear(self.config.hidden_size, oc, bias=True),
                                                    nn.LayerNorm(oc)) for oc in out_channels],
                                        )
        self.frozen = frozen
        if self.frozen:
            for param_name, param in self.vision_model.named_parameters():
                if not any(k in param_name for k in learnable_parameter_keys):
                    param.requires_grad = False

    def forward(self, img):
        """
        Forward function.

        Args:
            img (torch.Tensor): The input image.

        Returns:
            dict: A dictionary containing the output features.
        """
        
        if self.frozen:
            with torch.no_grad():
                raw_out = self.vision_model(img)
        else:
            raw_out = self.vision_model(img)
        if self.config.output_hidden_states:
            feats = raw_out.hidden_states
        else:
            feat = raw_out.last_hidden_state
        outs = []
        # k = self.config.image_size//self.config.patch_size
        k = img.shape[-1]//self.config.patch_size
        if self.add_map:
            for i in range(len(self.out_proj)):
                if self.config.output_hidden_states:
                    feat = feats[self.used_hidden_layers[i]]
                feat_proj = self.out_proj[i](feat[:,1:,:]) #B,L,D
                feat_proj = feat_proj.transpose(1,2) #B,D,L
                feat_proj = feat_proj.reshape(feat_proj.shape[0],feat_proj.shape[1],k,k) #B,D,H,W
                outs.append(feat_proj)
        else:
                feat_proj = feat[:,1:,:].transpose(1,2) #B,D,L
                feat_proj = feat_proj.reshape(feat_proj.shape[0],feat_proj.shape[1],k,k) #B,D,H,W
                outs.append(feat_proj)
        out_dict = dict(layer_outputs=outs,pooler_output=raw_out.pooler_output,raw_output=feat)
        return out_dict
