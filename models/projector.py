import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  Optional
from torch import Tensor
from .custom_layers import TransposeLayer

class Projector(nn.Module):
    """
    Projector is a component in the Token-Based Visual Transformer. It is used to
    use the tokens processed by the transformer to refine the feature map pixel
    representation by the information extracted from the visual tokens. It suitable
    in cases where pixel spatial properties need to be preserved, like in segmentation
    It takes the input feature map, of size (HW, C_in), and the output of the transformer
    layer (visual tokens), of size (L, D) and outputs. where:
    - L : number of tokens
    - C_in : number of feature map input channels
    - HW: number of pixels
    - C_out: number of feature map output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int, token_channels: int) -> None:
        super(Projector, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.token_channels = token_channels

        self.cache = None
        
        self.linear1 = nn.Linear(in_channels, token_channels, bias=False) # modifies feature map (query)
        self.linear2 = nn.Linear(token_channels , token_channels, bias=False) # modifies tokens (key)
        self.linear3 = nn.Linear(token_channels, out_channels) # modifies tokens (value)
        
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)

        self.norm = nn.BatchNorm1d(out_channels)
 
        # if input size is not same as output size
        # we use downsample to adjust the size of the input feature map
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Linear(in_channels, out_channels),
            )
        
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Expected Input:
        - x : feature map of size (N, HW, C_in)
        - t : visual tokens of size (N, L, D)
        
        Expected Output:
        - x_out : refined feature map of size (N, HW, C_out)
        """    
            
        x_q = self.linear1(x) # of size (N, HW, C_out)
        t_q = self.linear2(t) # of size (N, L, C_out)

        t_q = torch.transpose(t_q, 1, 2) # of size (N, C_out, L) 
        a = x_q.matmul(t_q) # of size (N, HW, L)
        a = a.softmax(dim=2) # of size (N, HW, L)
        self.cache = a
        
        t = self.linear3(t) # of size (N, L, C_out)

        a = a.matmul(t) # of shape (N, HW, C_out)
        
        if self.downsample != None:
            x = self.downsample(x)
            
        x = x + a # of shape (N, HW, C)
        
        x = torch.transpose(x, 1, 2)
        x = self.norm(x)
        x = torch.transpose(x, 1, 2)
        x = F.relu(x)
        
        return x
        