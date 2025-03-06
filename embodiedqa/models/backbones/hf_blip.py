from transformers import BlipTextModel, BlipVisionModel, BlipProcessor, BlipForQuestionAnswering, BlipConfig
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
import torch
import torch.nn as nn
import re

def rename_key(key):
    if "visual_encoder" in key:
        key = re.sub("visual_encoder*", "vision_model.encoder", key)
    if "blocks" in key:
        key = re.sub(r"blocks", "layers", key)
    if "attn" in key:
        key = re.sub(r"attn", "self_attn", key)
    if "norm1" in key:
        key = re.sub(r"norm1", "layer_norm1", key)
    if "norm2" in key:
        key = re.sub(r"norm2", "layer_norm2", key)
    if "encoder.norm" in key:
        key = re.sub(r"encoder.norm", "post_layernorm", key)
    if "encoder.patch_embed.proj" in key:
        key = re.sub(r"encoder.patch_embed.proj", "embeddings.patch_embedding", key)
    if "encoder.pos_embed" in key:
        key = re.sub(r"encoder.pos_embed", "embeddings.position_embedding", key)
    if "encoder.cls_token" in key:
        key = re.sub(r"encoder.cls_token", "embeddings.class_embedding", key)
    if "self_attn" in key:
        key = re.sub(r"self_attn.proj", "self_attn.projection", key)
    return key

@MODELS.register_module()
class BlipVisionModelWrapper(BaseModule):
    def __init__(self, name="Salesforce/blip-vqa-base", out_channels=[96, 192, 384, 768], used_hidden_layers=[12, 12, 12, 12], frozen=True, learnable_parameter_keys=[], load_raw_blip_pretrained_checkpoints=None):
        super().__init__()
        assert len(out_channels) == len(used_hidden_layers)
        self.out_channels = out_channels
        if load_raw_blip_pretrained_checkpoints is not None:
            # Load raw BLIP checkpoint and rename keys
            checkpoint = torch.load(load_raw_blip_pretrained_checkpoints, map_location="cpu")
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            renamed_state_dict = {rename_key(k): v for k, v in state_dict.items()}
            vision_model_state_dict = {}
            for k,v in renamed_state_dict.items():
                if 'vision_model' in k and 'encoder_m' not in k:
                    vision_model_state_dict[k.replace('vision_model.','')]=v
            
            # Initialize empty BlipForQuestionAnswering with the correct config
            config = BlipConfig.from_pretrained(name).vision_config
            config.image_size = 224
            self.vision_model = BlipVisionModel(config)
            self.vision_model.load_state_dict(vision_model_state_dict, strict=True)
            print(f'Successfully loaded state dict into the blip vision model from {load_raw_blip_pretrained_checkpoints}')
        else:
            self.vision_model = BlipForQuestionAnswering.from_pretrained(name).vision_model

        self.config = self.vision_model.config
        self.used_hidden_layers = used_hidden_layers
        self.config.output_hidden_states = not all([item == self.config.num_hidden_layers for item in used_hidden_layers])
        self.out_proj = nn.ModuleList([nn.Sequential(nn.Linear(self.config.hidden_size, oc, bias=True), nn.LayerNorm(oc)) for oc in out_channels])
        self.frozen = frozen
        if frozen:
            for param_name, param in self.vision_model.named_parameters():
                if not any(k in param_name for k in learnable_parameter_keys):
                    param.requires_grad = False

    def forward(self, img):
        assert img.shape[2] == img.shape[3]
        
        interpolate_pos_encoding = img.shape[2] != self.config.image_size
        outputs = self.vision_model(img, interpolate_pos_encoding=interpolate_pos_encoding, output_hidden_states=self.config.output_hidden_states)
        
        if self.config.output_hidden_states:
            feats = outputs.hidden_states
        else:
            feat = outputs.last_hidden_state

        outs = []
        k = img.shape[2] // self.config.patch_size
        for i in range(len(self.out_proj)):
            if self.config.output_hidden_states:
                feat = feats[self.used_hidden_layers[i] - 1]
            feat_proj = self.out_proj[i](feat[:, 1:, :])  # B, L, D
            feat_proj = feat_proj.transpose(1, 2)  # B, D, L
            feat_proj = feat_proj.reshape(feat_proj.shape[0], feat_proj.shape[1], k, k)  # B, D, H, W
            outs.append(feat_proj)
        out_dict = dict(layer_outputs=outs,pooler_output=feat[:, 0, :])
        return out_dict

@MODELS.register_module()
class BlipTextModelWrapper(BaseModule):
    def __init__(self, name="Salesforce/blip-vqa-base", frozen=True, learnable_parameter_keys=[], num_hidden_layers=12, load_raw_blip_pretrained_checkpoints=None):
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(name)
        config = BlipConfig.from_pretrained(name)
        config.text_config.num_hidden_layers = num_hidden_layers

        if load_raw_blip_pretrained_checkpoints is not None:
            # Load raw BLIP checkpoint and rename keys
            checkpoint = torch.load(load_raw_blip_pretrained_checkpoints, map_location="cpu")
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            renamed_state_dict = {rename_key(k): v for k, v in state_dict.items()}
            text_model_state_dict = {}
            for k,v in renamed_state_dict.items():
                if 'text_encoder' in k and 'encoder_m' not in k:
                    text_model_state_dict[k.replace('text_encoder.','')]=v
            # Initialize empty BlipForQuestionAnswering with the correct config
            self.text_model = BlipTextModel(config.text_config)
            missing_keys, unexpected_keys = self.text_model.load_state_dict(text_model_state_dict, strict=False)
            print(f'Successfully loaded state dict into the blip text model from {load_raw_blip_pretrained_checkpoints}')
            if missing_keys:
                print("Missing keys:")
                for key in missing_keys:
                    print(f"  {key}")

            if unexpected_keys:
                print("Unexpected keys:")
                for key in unexpected_keys:
                    print(f"  {key}")
        else:
            self.text_model = BlipForQuestionAnswering.from_pretrained(name, config=config).text_encoder

        self.config = self.text_model.config
        self.tokenizer = self.processor.tokenizer

        if frozen:
            for param_name, param in self.text_model.named_parameters():
                if not any(k in param_name for k in learnable_parameter_keys):
                    param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.text_model(*args, **kwargs)

    def get_tokenizer(self):
        return self.tokenizer

# 测试示例
if __name__ == "__main__":
    vision_model_wrapper = BlipVisionModelWrapper(load_raw_blip_pretrained_checkpoints="/data1/luojingzhou/datasets/blip/checkpoints/model_base_capfilt_large.pth")
    text_model_wrapper = BlipTextModelWrapper(load_raw_blip_pretrained_checkpoints="/data1/luojingzhou/datasets/blip/checkpoints/model_base_capfilt_large.pth")
    
    # 测试是否可以正常加载并运行
    dummy_img = torch.randn(1, 3, 224, 224)  # 示例输入
    vision_out = vision_model_wrapper(dummy_img)
    print("Vision model output shapes:", [o.shape for o in vision_out])

    dummy_text = torch.randint(0, 1000, (1, 10))  # 示例输入
    text_out = text_model_wrapper(input_ids=dummy_text)
    print("Text model output shape:", text_out.last_hidden_state.shape)

