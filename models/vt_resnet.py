import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Union, List
from .visual_transformer import VisualTransformer
from .resnet import BasicBlock, Bottleneck, resnet18_blocks, conv1x1, conv3x3

class VTResNet(nn.Module):
    """
    An implementation of Token-Based Visual Transformer by We et al. on top of ResNet
    Essentially the last stage of resnet is replaced by VisualTransformer blocks which 
    is repeated for the same number of times ResNet blocks are repeated in the last
    stage of ResNet.
    """
    
    def __init__(
        self, 
        resnet_block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        tokens: int,
        token_channels: int,
        input_dim: int,
        layer_planes: List[int],
        transformer_enc_layers: int,
        transformer_heads: int,
        transformer_fc_dim: int = 2024,
        transformer_dropout: int = 0.5,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        
        self.norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.layers = layers
        
        # defaut values for resnet are [64, 128, 256, 512]
        self.layer1_planes = layer_planes[0]
        self.layer2_planes = layer_planes[1]
        self.layer3_planes = layer_planes[2]
        self.layer4_planes = layer_planes[3]
        
        # for input of size 224 default values are [56, 28, 14, 14]
        self.layer1_res = input_dim // 4 
        self.layer2_res = input_dim // 8
        self.layer3_res = input_dim // 16
        self.layer4_res = input_dim // 16
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.resnet_layer1 = self._make_layer(
            block=resnet_block, 
            planes= self.layer1_planes,
            blocks= layers[0]
        )
        
        self.resnet_layer2 = self._make_layer(
            block=resnet_block, 
            planes= self.layer2_planes,
            blocks= layers[1],
            stride=2,
        )
        
        self.resnet_layer3 = self._make_layer(
            block=resnet_block, 
            planes= self.layer3_planes,
            blocks= layers[2],
            stride=2,
        )

        self.bn = nn.BatchNorm2d(self.layer3_planes)
        
        self.vt_layers = nn.ModuleList()
        self.vt_layers.append(
            VisualTransformer(
                in_channels=self.inplanes,
                out_channels=self.layer4_planes,
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
        
        for _ in range(1, layers[3]):
            self.vt_layers.append(
                VisualTransformer(
                    in_channels=self.layer4_planes,
                    out_channels=self.layer4_planes,
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
        self.fc = nn.Linear(self.layer4_planes, num_classes)
        
        # intialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self.norm_layer
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        
        x = self.bn(x)

        N, C, H, W = x.shape
        # flatten pixels
        x = x.view(N, H * W, -1)
        
        x, t = self.vt_layers[0](x)
        
        for i in range(1, self.layers[3]):
            x, t = self.vt_layers[i](x, t)
         
        x = x.reshape(N, self.layer4_planes, self.layer4_res, self.layer4_res)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
