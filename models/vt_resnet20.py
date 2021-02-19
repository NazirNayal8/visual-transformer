import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Union, List
from .visual_transformer import VisualTransformer
from .resnet_small import ResNet20

class VTResNet20(nn.Module):
    """
    An implementation of Token-Based Visual Transformer by We et al. on top of ResNet20
    Essentially the last stage of resnet is replaced by VisualTransformer blocks which 
    is repeated for the same number of times ResNet20 blocks are repeated in the last
    stage of ResNet.
    """
    
    def __init__(
        self, 
        vt_num_layers: int,
        tokens: int,
        token_channels: int,
        input_dim: int,
        vt_channels: int,
        transformer_enc_layers: int,
        transformer_heads: int,
        transformer_fc_dim: int = 128,
        transformer_dropout: int = 0.5,
        image_channels: int = 3,
        num_classes: int = 1000,
        resnet_pretrained: bool = True,
        freeze_resnet: bool = True
    ) -> None:
        super().__init__()
        
        self.norm_layer = nn.BatchNorm2d
        self.tokens = tokens
        self.vt_inplanes = 32
        self.vt_channels = vt_channels
        self.vt_num_layers = vt_num_layers

        self.vt_layer_res = input_dim // 2
        
        self.resnet20 = None
        
        if resnet_pretrained:
            self.resnet20 = self.pretrained_ResNet20()
        else:
            self.resnet20 = ResNet20()
            for m in self.resnet20.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        if freeze_resnet and (not resnet_pretrained):
            raise ValueError("ResNet weights cannot be freezed unless pretrained model is used")
        
        if freeze_resnet:
            for n, p in self.resnet20.named_parameters():
                if "conv1" in n or "bn1" in n or "layer1" in n:
                    p.requires_grad = False
            
        
        self.vt_layers = nn.ModuleList()
        self.vt_layers.append(
            VisualTransformer(
                in_channels=self.vt_inplanes,
                out_channels=self.vt_channels,
                token_channels=token_channels,
                tokens=tokens,
                tokenizer_type='filter',
                attn_dim=token_channels,
                transformer_enc_layers=transformer_enc_layers,
                transformer_heads=transformer_heads,
                transformer_fc_dim=transformer_fc_dim,
                transformer_dropout=transformer_dropout,
                is_projected=True
            )
        )
        
        for _ in range(1, vt_num_layers):
            self.vt_layers.append(
                VisualTransformer(
                    in_channels= self.vt_channels,
                    out_channels= self.vt_channels,
                    token_channels=token_channels,
                    tokens=tokens,
                    tokenizer_type='recurrent',
                    attn_dim=token_channels,
                    transformer_enc_layers=transformer_enc_layers,
                    transformer_heads=transformer_heads,
                    transformer_fc_dim=transformer_fc_dim,
                    transformer_dropout=transformer_dropout,
                    is_projected=True
                )
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.vt_channels, num_classes)
        
    
    def pretrained_ResNet20(self):
        res = ResNet20()
        checkpoint = torch.load('./saved_models/ckpt_ResNet20_small_backbone_8x8.pth')
        res.load_state_dict(checkpoint)
        return res
    
    def forward(self, x: Tensor) -> Tensor:
        
        x = self.resnet20(x)

        N, C, H, W = x.shape
        # flatten pixels
        
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        
        x, t = self.vt_layers[0](x)
        
        for i in range(1, self.vt_num_layers):
            x, t = self.vt_layers[i](x, t)
        
        x = x.permute(0, 2, 1)
        x = x.reshape(N, self.vt_channels, self.vt_layer_res, self.vt_layer_res)
          
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
