from transformers import SwinModel, AutoConfig
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
@MODELS.register_module()
class SwinModelWrapper(BaseModule):
    def __init__(self, name="microsoft/swin-base-patch4-window7-224-in22k", out_channels=[1024], add_map=False, frozen=True, learnable_parameter_keys=[]):
        """
        Initialize a SwinModelWrapper.

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
        self.out_channels = out_channels
        self.config = AutoConfig.from_pretrained(name)
        self.config.output_hidden_states = False
        self.vision_model = SwinModel.from_pretrained(name, config=self.config)
        self.add_map = add_map
        if self.add_map:
            self.out_proj = nn.ModuleList([nn.Sequential(nn.Linear(self.config.hidden_size, oc, bias=True),
                                                         nn.LayerNorm(oc)) for oc in out_channels])
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
        feat = raw_out.last_hidden_state
        outs = []
        # k = img.shape[-1] // self.config.patch_size
        k = self.config.window_size
        if self.add_map:
            for i in range(len(self.out_proj)):
                feat_proj = self.out_proj[i](feat)  # B,L,D
                feat_proj = feat_proj.transpose(1, 2)  # B,D,L
                feat_proj = feat_proj.reshape(feat_proj.shape[0], feat_proj.shape[1], k, k)  # B,D,H,W
                outs.append(feat_proj)
        else:
            feat_proj = feat.transpose(1, 2)  # B,D,L
            feat_proj = feat_proj.reshape(feat_proj.shape[0], feat_proj.shape[1], k, k)  # B,D,H,W
            outs.append(feat_proj)
        out_dict = dict(layer_outputs=outs, pooler_output=raw_out.pooler_output,raw_output=feat)
        return out_dict
