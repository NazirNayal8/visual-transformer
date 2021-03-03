import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Union, List, Any
from .visual_transformer import VisualTransformer
from .resnet import ResNet, BasicBlock, Bottleneck, resnet18_blocks, conv1x1, conv3x3
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class VTResNet(nn.Module):
    """
    An implementation of Token-Based Visual Transformer by We et al. on top of ResNet
    Essentially the last stage of resnet is replaced by VisualTransformer blocks which 
    is repeated for the same number of times ResNet blocks are repeated in the last
    stage of ResNet.
    """
    
    def __init__(
        self, 
        resnet_layer: nn.Module,
        vt_layers_num: int,
        tokens: int,
        token_channels: int,
        input_dim: int,
        vt_channels: int,
        transformer_enc_layers: int,
        transformer_heads: int,
        transformer_fc_dim: int = 1024,
        transformer_dropout: int = 0.5,
        image_channels: int = 3,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        
        self.norm_layer = nn.BatchNorm2d
        self.in_channels = vt_channels // 2
        self.tokens = tokens
        self.vt_channels = vt_channels
        self.vt_layers_num = vt_layers_num
        # feature map resolution
        self.vt_layer_res = input_dim // 16
        
        self.resnet = resnet_layer    
        self.bn = nn.BatchNorm2d(self.in_channels)
        
        self.vt_layers = nn.ModuleList()
        self.vt_layers.append(
            VisualTransformer(
                in_channels=self.in_channels,
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
        
        for _ in range(1, self.vt_layers_num):
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
        
        # intialize weights
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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
        x = self.resnet(x)
        x = self.bn(x)

        N, C, H, W = x.shape
        
        # flatten pixels
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        
        x, t = self.vt_layers[0](x)
        
        for i in range(1, self.vt_layers_num):
            x, t = self.vt_layers[i](x, t)
    
        x = x.reshape(N, self.vt_channels, self.vt_layer_res, self.vt_layer_res)
          
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def create_model(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    freeze: str,
    progress: bool,
    **kwargs: Any
) -> VTResNet:
    """
    A helper function to prepare the ResNet Backbone before creating 
    the VT Module.
    """
    resnet = ResNet(block, layers)
    
    if freeze not in ['no_freeze', 'partial_freeze', 'full_freeze']:
        raise ValueError('Freeze value undefined')
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        resnet.load_state_dict(state_dict)
        
        if freeze == 'partial_freeze':
            for n, p in resnet.named_parameters():
                if "conv1" in n or "bn1" in n or "layer1" in n:
                    p.requires_grad = False
        elif freeze == 'full_freeze':
            for n, p in resnet.named_parameters():
                if "conv" in n or "bn" in n or "layer" in n:
                    p.requires_grad = False
        
    else:
        for m in resnet.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
    return VTResNet(resnet, layers[-1], **kwargs)

def vt_resnet18(pretrained: bool = False, freeze: str = 'no_freeze', progress: bool = True, **kwargs: Any) -> VTResNet:
    """
    Create a VTResNet Model with ResNet18 as a convolutional backbone.
    """
    return create_model('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, freeze, progress, **kwargs)
    
def vt_resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VTResNet:
    """
    Create a VTResNet Model with ResNet34 as a convolutional backbone.
    """
    return create_model('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def vt_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VTResNet:
    """
    Create a VTResNet Model with ResNet50 as a convolutional backbone.
    """
    return create_model('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def vt_resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VTResNet:
    """
    Create a VTResNet Model with ResNet101 as a convolutional backbone.
    """
    return create_model('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def vt_resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VTResNet:
    """
    Create a VTResNet Model with ResNet152 as a convolutional backbone.
    """
    return create_model('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)
