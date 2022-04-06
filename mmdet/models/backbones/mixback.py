from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple

from mmdet.models.builder import BACKBONES
from mmdet.models.backbones.swin import SwinBlockSequence
from mmdet.models.backbones.pvt import PVTEncoderLayer
from mmdet.utils import get_root_logger
from mmdet.models.utils.ckpt_convert import swin_converter
from mmdet.models.utils.transformer import PatchEmbed, PatchMerging,nlc_to_nchw

import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F




class Mix_stage(BaseModule):
    """ Implements one stage
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 block_type="swin",
                 cur=0,
                 embed_dims=96,
                 num_heads=6,
                 mlp_ratio=4,
                 depth=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 dpr=[],
                 sr_ratio=4,
                 downsample=None,
                 window_size=7,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        stage=ModuleList()
        if(block_type=="swin"):
            stage = SwinBlockSequence(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                depth=depth,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None).blocks
        elif(block_type=="pvt"):
            stage.extend([
                    PVTEncoderLayer(
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        feedforward_channels=mlp_ratio,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[cur + idx],
                        qkv_bias=qkv_bias,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        sr_ratio=sr_ratio) for idx in range(depth)
                ])
            in_channels = embed_dims
        self.blocks=stage
        self.downsample=downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape



@BACKBONES.register_module()
class Mix_back(BaseModule):

    def __init__(self,
                 stages_inf,
                 depth=None,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_sizes=[8,4,4,4],
                 strides=(8, 4, 4, 4),
                 out_indices=(0, 1, 2, 3),
                 paddings=[0,0,0,0],
                 patch_norm=True,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 norm_cfg=dict(type='LN'),
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(Mix_back, self).__init__(init_cfg=init_cfg)
        depths=[]
        for i in range(len(stages_inf)):
            depths.append(stages_inf[i]["depth"])

        num_layers = len(stages_inf)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        #assert strides[0] == patch_sizes[0], 'Use non-overlapping patch embed.'

        """
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)
        """
        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_sizes[0]
            patch_col = pretrain_img_size[1] // patch_sizes[0]
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = 3
        cr=0
        embed_dims_i=embed_dims
        self.patch_embed=ModuleList()
        for i in range(num_layers):
            print(in_channels)
            self.patch_embed.append(PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding="corner",
                bias=True,
                norm_cfg=norm_cfg))
            in_channels=embed_dims_i
            embed_dims_i = embed_dims_i * 2
            
            """          
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None
            """
            stage =Mix_stage(
            drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
            downsample=None,
            embed_dims=in_channels,
            cur=cr,
            dpr=dpr,
            **stages_inf[i]
            )
            self.stages.append(stage)
            """
            if downsample:
                in_channels = downsample.out_channels
            cr+=depths[i]
            """
        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(Mix_back, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        

        outs = []
        for i, stage in enumerate(self.stages):
            patch_embed=self.patch_embed[i]
            
            if (i==0):
              if self.use_abs_pos_embed:
                x = x + self.absolute_pos_embed
              x = self.drop_after_pos(x)

            x, hw_shape=patch_embed(x)
            print(hw_shape)
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(x)
                

        return outs