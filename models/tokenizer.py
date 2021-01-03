import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor

class FilterTokenizer(nn.Module):
    """
    The Filter Tokenizer extracts visual tokens using point-wise convolutions.
    It takes input of size (HW, C) and outputs a tensor of size (L, D) where:
    - HW : height x width, which represents the number of pixels
    - C : number of input channels
    - L : number of tokens
    - D : number of token channels
    """
    def __init__(self, in_channels: int, token_channels: int, tokens: int) -> None:
        super(FilterTokenizer, self).__init__()
        
        self.tokens = tokens
        self.in_channels = in_channels
        self.token_channels = token_channels
    
        self.conv1 = nn.Conv2d(in_channels, tokens, kernel_size=1)
        self.linear1 = nn.Linear(in_channels, token_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Expected Input Dimensions: (N, C, H, W), where:
        - N: batch size
        - HW: number of pixels
        - C: number of input feature map channels
        
        Expected Output Dimensions: (N, L, D), where:
        - L: number of tokens
        - D: number of token channels
        """

        N, C, H, W = x.shape

        a = self.conv1(x) # of size (N, L, H, W)

        a = a.view(N, self.tokens, H * W)
        a = a.softmax(dim=2) # softmax for HW dimension, such that every group l features sum to 1
        
        b = x.view(N, H * W, C)
        a = a.matmul(b)  # of size (N, L, C)
        
        a = self.linear1(a) # of size (N, L, D)

        return a

class RecurrentTokenizer(nn.Module):
    """
    The Recurrent Tokenizer extracts visual tokens by recurrently using tokens generated from
    previous iteration.
    It takes input of size (HW, C), and Tokens matrix of size (L, D) and outputs a tensor 
    of size (L, C) where:
    - HW : height x width, which represents the number of pixels
    - C : number of input feature map channels
    - L : number of tokens
    - D : number of token channels
    """
    def __init__(self, in_channels: int, token_channels: int) -> None:
        super(RecurrentTokenizer, self).__init__()
        
        self.token_channels = token_channels
        self.linear1 = nn.Linear(token_channels, token_channels)
        self.conv1 = nn.Conv2d(in_channels, token_channels, kernel_size=1)
 
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Expected Input:
        - x : image features input, of size (N, C, H, W)
        - t : token features extracted previously, of size (N, L, D)
        
        Expected  Output:
        - t_new : new token features, of size(N, L, D)
        
        where:
        - N : batch size
        - HW: number of pixels
        """

        N, C, H, W = x.shape
    
        a = self.linear1(t) # of size (N, L, D)
        
        x = self.conv1(x) # of size (N, D, H, W)
       
        a = torch.transpose(a, 1, 2) # transpose by swapping dimensions to become (N, D, L)
        
        b = x.view(N, H * W, self.token_channels)

        a = b.matmul(a) # of size (N, HW, L)
        a = a.softmax(dim=2) # softmax for HW dimension, such that every group l features sum to 1
        a = torch.transpose(a, 1, 2) # transpose by swapping dimensions to become (N, L, HW)
        
        t = a.matmul(b) # of size (N, L, D)

        return t
    
