import torch
import torch.nn as nn
import einops

#Adaptive Dual-vision Perception (ADVP) module
class VisionFusion(nn.Module):
    def __init__(self, lidar_channels, camera_channels, out_channels):
        super().__init__()
        
        # self.camera_weight = nn.Sequential(
        #     nn.Linear(camera_channels, camera_channels//2),
        #     nn.LayerNorm(camera_channels//2),
        #     nn.GELU(),
        #     nn.Linear(camera_channels//2, 1),
        #     nn.Sigmoid()
        # )
        
        # Channel-wise attention layer at each point
        self.channel_attention = nn.Sequential(
            nn.Linear(lidar_channels + camera_channels, (lidar_channels + camera_channels) // 16),
            nn.LayerNorm((lidar_channels + camera_channels) // 16),
            nn.GELU(),
            nn.Linear((lidar_channels + camera_channels) // 16, lidar_channels + camera_channels),
            nn.Sigmoid()
        )
        
        # maping
        self.final_mapping = nn.Sequential(
            nn.Linear(lidar_channels + camera_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

        # norm
        self.ln_fused = nn.LayerNorm(lidar_channels + camera_channels)
        self.bn_camera = nn.BatchNorm1d(camera_channels)
    def _init_weights(self):
        """Initialize the weights of the fusion blocks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    def forward(self, lidar_features, camera_features):
        
        B, N, _ = lidar_features.shape
        
        camera_features = self.bn_camera(camera_features.transpose(1, 2)).transpose(1, 2)
        
        fused_features = torch.cat([lidar_features, camera_features], dim=-1)
        
        
        # # 应用逐点通道注意力
        channel_attn = self.channel_attention(fused_features)
        fused_features = fused_features * channel_attn
        fused_features = self.ln_fused(fused_features)
        
        # 最终映射
        output = self.final_mapping(fused_features)
        return output
