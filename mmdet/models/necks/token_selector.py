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
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1,1), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        
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
@NECKS.register_module()
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
                 in_channels,
                 out_channels=256,
                 num_tokens=[32,16,8,4],
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 num_outs=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(TokenSelector, self).__init__(init_cfg)
        self.ch=ChannelMapper(in_channels,out_channels,kernel_size,conv_cfg,norm_cfg,act_cfg,num_outs,init_cfg)
        self.channels=in_channels
        self.Tks=ModuleList()
        self.Tfs=ModuleList()
        for i,c in enumerate(in_channels):
            self.Tks.append(TokenLearner(S=num_tokens[i]))
            self.Tfs.append(TokenFuser(c,S=num_tokens[i]))
    def forward(self, inputs):
        """Forward function."""
        outs=[]
        for i,x in enumerate(inputs):
            #check dim order
            x=x.permute(0,2,3,1)
            out=self.Tks[i](x)
            B,S,C=out.shape
            out=out.permute(0,2,1)
            out=out.reshape(B,C,S//2,S//2)
            #check dim order
            outs.append(out)
        for o in outs:
            print(o.shape)
        sorties=self.ch(outs)
        return tuple(sorties)
        
