# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.necks.channel_mapper import ChannelMapper
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule,ModuleList
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS





class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = ConvModule(2,1,kernel_size=(1,1), stride=1,norm_cfg=dict(type="BN2d",num_features=1))
        
        self.sgap = nn.AvgPool2d(2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        mx = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        combined = torch.cat([mx, avg], dim=1)
        fmap = self.conv(combined)
        weight_map = torch.sigmoid(fmap)
        out = (x * weight_map).mean(dim=(-2, -1))
        
        return out, x * weight_map

class TokenLearner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = [SpatialAttention() for _ in range(S)]
        
    def forward(self, x):
        B, _, _, C = x.shape
        Z = torch.Tensor(B, self.S, C)
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x) # [B, C]
            Z[:, i, :] = Ai
        return Z
    
class TokenFuser(nn.Module):
    def __init__(self, C, S) -> None:
        super().__init__()
        self.projection = nn.Linear(S, S, bias=False)
        self.Bi = nn.Linear(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S
        
    def forward(self, y, x):
        B, S, C = y.shape
        B, H, W, C = x.shape
        
        Y = self.projection(y.view(B, C, S)).view(B, S, C)
        Bw = torch.sigmoid(self.Bi(x)).view(B, H*W, S) # [B, HW, S]
        BwY = torch.matmul(Bw, Y)
        
        _, xj = self.spatial_attn(x)
        xj = xj.view(B, H*W, C)
        
        out = (BwY + xj).view(B, H, W, C)
        
        return out 
@NECKS.register_module()
class TokenSelector(BaseModule):
    r"""
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels=96,
                 out_channels=64,
                 window_size=[4,8,16],
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 num_outs=4,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(TokenSelector, self).__init__(init_cfg)
        #self.ch=ChannelMapper(in_channels,out_channels,kernel_size,conv_cfg,norm_cfg,act_cfg,num_outs=len(window_size)+1,init_cfg=init_cfg)
        self.channels=in_channels
        self.window_size = window_size
        self.Tks=ModuleList()
        self.Tfs=ModuleList()
        for i,c in enumerate(window_size):
            self.Tks.append(TokenLearner(S=1))



    def forward(self, input):
        """inputs list of form B,C,H,W"""
        outs=[]
        x=input[0]
        outs.append(x)
        x=x.permute(0,2,3,1) #B,H,W,C
        B,H,W,C=x.shape
        for i,WS in enumerate(self.window_size):
            
            # Pad to a multiple of window size
            pad_r = (WS - W % WS) % WS
            pad_b = (WS - H % WS) % WS
            x_pad = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            H_pad, W_pad = x_pad.shape[1], x_pad.shape[2]
            # nW*B, window_size, window_size, C
            x_windows = self.window_partition(x_pad,WS)
            # nW*B, window_size*window_size, C
            x_windows = x_windows.view(-1,WS,WS, C)
            
            out_win=self.Tks[i](x_windows)
            rev_x = self.window_reverse(out_win, H_pad, W_pad,WS)
            rev_x=rev_x.permute(0,3,1,2)
            outs.append(rev_x)
        for o in outs:
            print(o.shape)
        return tuple(outs)
    def window_reverse(self, windows, H, W,window_size):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size,-1)
        return x

    def window_partition(self, x,window_size):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        print(x.shape)
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows
        
