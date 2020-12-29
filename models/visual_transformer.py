import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from .transformer import Transformer
from .tokenizer import FilterTokenizer, RecurrentTokenizer
from .projector import Projector

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
        is_projected: bool = True,
    ) -> None:
        super(VisualTransformer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.token_channels = token_channels
        self.attn_dim = attn_dim
        self.is_projected = is_projected
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type not in ['recurrent','filter']:
            raise ValueError('tokenizer type must be either recurrent of filter.')
                        
        self.tokenizer = None
        if tokenizer_type == 'recurrent':
            self.tokenizer = RecurrentTokenizer(in_channels, token_channels)
        else:
            self.tokenizer = FilterTokenizer(in_channels, token_channels, tokens)
        
        # Transformer(token_channels, attn_dim)
        self.transformer = nn.Transformer(
            token_channels, 
            nhead=token_channels, 
            num_encoder_layers=4, 
            num_decoder_layers=0, 
            dim_feedforward=1024,
            dropout=0.5
        )
        
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
        # apply transformer
        out = self.transformer(t, t)
        
        if self.is_projected:
            out = self.projector(x, t)
        
        return out, t
        