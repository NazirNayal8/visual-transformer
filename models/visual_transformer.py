import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math
from .transformer import Transformer, SelfAttention
from .tokenizer import FilterTokenizer, RecurrentTokenizer
from .projector import Projector

# Adapted from 
# https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

# Adapted from 
# https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class VisualTransformer(nn.Module):
    """
    An implementation of the Token-Based Visual Transformer Module by Wu et al.
    It takes a feature map as input, and depending on whether 
    Parameters:
    - inchannels: number of input channels of feature maps
    - tokens: number of visual tokens to be extracted
    - attn_dim: dimension of projections used in self attention in the transformer
    - tokenization_rounds: number of recurrent iterations for which tokenization is applied
        it includes the first round (filter based) and other rounds (recurrent). 
        (Must be greater than or equal to 1)
    - is_projected: a boolean equal to True with the output is expected to be projected
        back into a spatial map, and False if the output should represent the visual tokens
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        token_channels: int, 
        tokens: int, 
        tokenizer_type: str,
        attn_dim: int, 
        transformer_enc_layers: int,
        transformer_heads: int,
        transformer_fc_dim: int,
        transformer_dropout: int,
        is_projected: bool = True,
    ) -> None:
        super(VisualTransformer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.token_channels = token_channels
        self.attn_dim = attn_dim
        self.is_projected = is_projected
        self.tokens = tokens
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type not in ['recurrent','filter']:
            raise ValueError('tokenizer type must be either recurrent of filter.')
                        
        self.tokenizer = None
        if tokenizer_type == 'recurrent':
            self.tokenizer = RecurrentTokenizer(in_channels, token_channels)
        else:
            self.tokenizer = FilterTokenizer(in_channels, token_channels, tokens)
        
        
        # self.transformer = SelfAttention(token_channels, token_channels)
        
        #Transformer(token_channels, attn_dim)
        self.transformer = nn.Transformer(
            token_channels, 
            nhead=transformer_heads, 
            num_encoder_layers=transformer_enc_layers, 
            num_decoder_layers=0, 
            dim_feedforward=transformer_fc_dim,
            dropout=transformer_dropout
        )

        # self.transformer = Transformer(
        #     token_channels=token_channels,
        #     attn_dim=attn_dim,
        #     dropout=transformer_dropout
        # )
        
        self.projector = None
        if is_projected:
            self.projector = Projector(in_channels, out_channels, token_channels)
    
    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """
        Expected Input:
        - x : input feature maps, of size (N, HW, C)
        
        Expected Output depends on value of self.is_projected.
        If self.is_projected is True:
            - X_out: refined feaure maps, of size (N, HW, C)
        if self.is_projected is False:
            - T_out: visual tokens, of size (N, L, C)
            
        where:
        - N : batch size
        - HW: number of pixels
        - C : number of channels
        - L : number of tokens
        """
        # apply tokenizer
        if self.tokenizer_type == 'filter':
            t = self.tokenizer(x) 
        else:
            t = self.tokenizer(x, t)
        # (N, L, C) -> (L, N, C)
        t = t.permute(1, 0, 2)
        # apply transformer
        t_out = self.transformer(t, t)
        
        t_out = t_out.permute(1, 0, 2) 
        t = t.permute(1, 0, 2)
        
        if self.is_projected:
            out = self.projector(x, t_out)
        
        return out, t
        