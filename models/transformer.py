import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttention(nn.Module):
    """
    PyTorch Module that implements the Self Attention Mechanism which is modified according to
    the Token-Based Visual Transformer. It takes a matrix T_in of size (L,D), which represents
    the extracted visual tokens, and it outputs T_out of size (L,D), which represents the 
    visual tokens after processing them.
    - L : number of tokens
    - D : number of token channels
    """
    def __init__(self, channels: int, attn_dim: int) -> None:
        super(SelfAttention, self).__init__()
        
        self.channels = channels
        self.attn_dim = attn_dim
        
        # Projections are of dimensions (C, D) , where D is the projection dimension
        # Bias is set to false because these only represent a projection
        self.query_linear = nn.Linear(channels, attn_dim, bias=False)
        self.key_linear = nn.Linear(channels, attn_dim, bias=False)
        
        nn.init.kaiming_normal_(self.query_linear.weight)
        nn.init.kaiming_normal_(self.query_linear.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Expected Input:
        - x: of size (N, L, C) 
        
        Expected Output:
        - T_out : of size (N, L, C)
        """
        key = self.key_linear(x) # of size (N, L, D)
        query = self.query_linear(x) # of size (N, L, D)
        query = torch.transpose(query, 1, 2) # of size (N, D, L)
        
        T_out = key.matmul(query) # of size (N, L, L)
        # Note: since both dimensions are L, try to make sure we are computing softmax
        # for the correct dimension
        T_out = T_out.softmax(dim=1) 
        
        T_out = T_out.matmul(x) # of size (N, L, C)

        return T_out
        
class Transformer(nn.Module):
    """
    An implementation of a modified version of the Transformer Encoder by Vaswani et al.
    This modification is made for the Token-Based Visual Transformer by Wu et al.
    
    It takes an input T of size (L, D), and outputs a tensor of the same size after 
    processing it with the transformer. where:
    - L: number of tokens
    - D: number of token channels
    """
    def __init__(self, token_channels: int, attn_dim: int, dropout: int) -> None:
        super(Transformer, self).__init__()
        
        self.token_channels = token_channels
        self.attn_dim = attn_dim
        
        self.attention = SelfAttention(token_channels, attn_dim)
        self.linear1 = nn.Linear(token_channels, token_channels)
        self.linear2 = nn.Linear(token_channels, token_channels)

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        
        self.layer_norm1 = nn.LayerNorm(token_channels)
        self.layer_norm2 = nn.LayerNorm(token_channels)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Expected Input:
        - x: of size (N, L, D)
        
        Expected Output:
        - T_out: of size (N, L, D)
        
        where:
        - N : batch size
        - L : number of tokens
        - D : number of token channels
        """
        a = x + self.attention(x) # of size (N, L, D)

        a = self.layer_norm1(a) # of size (N, L, D)
       
        b = self.linear1(a) # of size (N, L, D)
        b = F.relu(b)
        b = self.linear2(b) # of size (N, L, D)
        a = a + b # of size (N, L, D)
        a = self.layer_norm2(a) # of size (N, L, D)
        
        return a
    