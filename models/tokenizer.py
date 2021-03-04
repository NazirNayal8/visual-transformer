import torch
import torch.nn as nn
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

        self.linear1 = nn.Linear(in_channels, tokens)
        self.linear2 = nn.Linear(in_channels, token_channels)

        self.cache1 = None
        self.cache2 = None
        self.token_cache = None

        # initialize weights
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Expected Input Dimensions: (N, HW, C), where:
        - N: batch size
        - HW: number of pixels
        - C: number of input feature map channels
        
        Expected Output Dimensions: (N, L, D), where:
        - L: number of tokens
        - D: number of token channels
        """

        a = self.linear1(x)  # of size (N, HW, L)
        self.cache1 = a
        a = a.softmax(dim=1)  # softmax for HW dimension, such that every group l features sum to 1
        self.cache2 = a
        a = torch.transpose(a, 1, 2)  # swap dimensions 1 and 2, of size (N, L, HW)
        a = a.matmul(x)  # of size (N, L, C)
        a = self.linear2(a)  # of size (N, L, D)

        self.token_cache = a
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
        self.linear2 = nn.Linear(in_channels, token_channels)

        self.cache1 = None
        self.cache2 = None
        self.token_cache = None

        # initialize weights
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Expected Input:
        - x : image features input, of size (N, HW, C)
        - t : token features extracted previously, of size (N, L, D)
        
        Expected  Output:
        - t_new : new token features, of size(N, L, D)
        
        where:
        - N : batch size
        - HW: number of pixels
        """

        a = self.linear1(t)  # of size (N, L, D)

        b = self.linear2(x)  # of size (N, HW, D)

        a = torch.transpose(a, 1, 2)  # transpose by swapping dimensions to become (N, D, L)

        a = b.matmul(a)  # of size (N, HW, L)
        self.cache1 = a
        a = a.softmax(dim=1)  # softmax for HW dimension, such that every group l features sum to 1
        self.cache2 = a
        a = torch.transpose(a, 1, 2)  # transpose by swapping dimensions to become (N, L, HW)
        b = a.matmul(b)  # of size (N, L, D)

        return b
