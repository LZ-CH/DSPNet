from transformers import AutoModel, AutoConfig, SegformerForSemanticSegmentation
import torch.nn as nn
import torch
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
import math
@MODELS.register_module()
class SegFormerModelWrapper(BaseModule):
    def __init__(self, name="nvidia/segformer-b3-finetuned-ade-512-512", out_channels=[256], frozen=True, learnable_parameter_keys=['decode_head.classifier'],**kwargs):
        """
        Initialize a SegFormer model wrapper.

        Args:
            name (str): Huggingface model name. Defaults to "nvidia/segformer-b3-finetuned-ade-512-512".
            out_channels (list): A list of number of channels for each layer. Defaults to [256].
            frozen (bool): Set to ``True`` to freeze all the parameters except those in
                ``learnable_parameter_keys``. Defaults to ``True``.
            learnable_parameter_keys (list): A list of parameter names that are set to be learnable.
                Defaults to ``['decode_head.classifier']``.
            **kwargs: Additional keyword arguments passed to the :class:`BaseModule` constructor.
        """
        super().__init__()
        self.out_channels = out_channels
        self.config = AutoConfig.from_pretrained(name)
        self.config.num_labels = out_channels[-1]
        self.vision_model = SegformerForSemanticSegmentation.from_pretrained(name, config=self.config, ignore_mismatched_sizes=True)
        self.bn = nn.BatchNorm2d(out_channels[-1])
        if frozen:
            for param_name, param in self.vision_model.named_parameters():
                if not any(k in param_name for k in learnable_parameter_keys):
                    param.requires_grad = False

    def forward(self, img):
        with torch.no_grad():
            encoder_hidden_states = self.vision_model.segformer(
                img,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=True,
            ).hidden_states

            #decoder head
            
            batch_size = encoder_hidden_states[-1].shape[0]
            all_hidden_states = ()
            for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.vision_model.decode_head.linear_c):
                if self.vision_model.decode_head.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                    height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                    encoder_hidden_state = (
                        encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                    )

                # unify channel dimension
                height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
                encoder_hidden_state = mlp(encoder_hidden_state)
                encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
                encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
                # upsample
                encoder_hidden_state = nn.functional.interpolate(
                    encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
                )
                all_hidden_states += (encoder_hidden_state,)

            hidden_states = self.vision_model.decode_head.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
            hidden_states = self.vision_model.decode_head.batch_norm(hidden_states)
            hidden_states = self.vision_model.decode_head.activation(hidden_states)
            hidden_states = self.vision_model.decode_head.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.vision_model.decode_head.classifier(hidden_states)
        feat = self.bn(logits)
        out_dict = dict(layer_outputs=[feat],pooler_output=feat.mean(dim=[2,3]))
        return out_dict
