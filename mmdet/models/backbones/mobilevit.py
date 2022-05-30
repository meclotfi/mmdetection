from turtle import forward
import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Dict, Tuple
import math
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn import ConvModule,build_norm_layer
from torch import nn, Tensor
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS,ATTENTION
from mmcv.cnn.bricks.transformer import build_transformer_layer
from ..utils import InvertedResidual
from mmcv.utils import to_2tuple
from mmcv.runner import BaseModule
from ..builder import BACKBONES
import einops
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    print(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

# adding swish
if ACTIVATION_LAYERS.get("Swish") is None:
    @ACTIVATION_LAYERS.register_module()
    class Swish(nn.SiLU):
        def __init__(self, inplace: bool = False):
            super(Swish, self).__init__(inplace=inplace)

if ATTENTION.get("MultiScaleDeformableAttentionL1") is None:
    @ATTENTION.register_module()
    class MultiScaleDeformableAttentionL1(MultiScaleDeformableAttention):
        def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=16,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
            super().__init__(embed_dims=embed_dims,num_heads=num_heads,num_levels=num_levels,num_points=num_points,im2col_step=im2col_step,dropout=dropout,batch_first=batch_first,norm_cfg=norm_cfg,init_cfg=init_cfg)
        @staticmethod
        def get_reference_points(spatial_shapes, valid_ratios, device):
            """Get the reference points used in decoder.
            Args:
                spatial_shapes (Tensor): The shape of all
                    feature maps, has shape (num_level, 2).
                valid_ratios (Tensor): The radios of valid
                    points on the feature map, has shape
                    (bs, num_levels, 2)
                device (obj:`device`): The device where
                    reference_points should be.
            Returns:
                Tensor: reference points used in decoder, has \
                    shape (bs, num_keys, num_levels, 2).
            """
            reference_points_list = []
            for lvl, (H, W) in enumerate(spatial_shapes):
                #  TODO  check this 0.5
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(
                        0.5, H - 0.5, H, dtype=torch.float32, device=device),
                    torch.linspace(
                        0.5, W - 0.5, W, dtype=torch.float32, device=device))
                ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
                ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
                ref = torch.stack((ref_x, ref_y), -1)
                reference_points_list.append(ref)
            reference_points = torch.cat(reference_points_list, 1)
            reference_points = reference_points[:, :, None] * valid_ratios[:, None]
            return reference_points

        def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                **kwargs):
            num_patch=int(np.sqrt(query.shape[1]))
            #print(query.shape)
            #print(num_patch)
            dev=query.device
            Spatial_shapes=torch.tensor([[num_patch,num_patch]]).to(dev)
            level_start_index=torch.tensor([0,num_patch**2]).to(dev)
            valid_ratios = einops.repeat(torch.tensor([1.,1.]), 'n -> m k n',m=query.shape[0],k=1).to(dev)
            reference_points =self.get_reference_points(Spatial_shapes,valid_ratios,device=dev)
            return super().forward(query=query,key=key,value=value,identity=identity,query_pos=query_pos,key_padding_mask=None,level_start_index=level_start_index,reference_points=reference_points,spatial_shapes=Spatial_shapes, **kwargs)
class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x

class MobileViTBlock(BaseModule):
    """
        MobileViT block: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self, in_channels: int, num_heads: int, ffn_dim: int,
                 n_transformer_blocks: Optional[int] = 2,
                 head_dim: Optional[int] = 32, attn_dropout: Optional[float] = 0.1,
                 dropout: Optional[int] = 0.1, ffn_dropout: Optional[int] = 0.1, patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8, transformer_norm_layer: Optional[str] = "layer_norm",
                 conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1, var_ffn: Optional[bool] = False,
                 no_fusion: Optional[bool] = False,act_config=dict(type="Swish"),norm_cfg=dict(type='LN'),
                 *args, **kwargs):
        
        super().__init__()
        transformer_dim=head_dim*num_heads
        norm_cfg_v=dict(type='BN', requires_grad=True)
        self.pad=AdaptivePadding( kernel_size=conv_ksize, stride=1, dilation=1, padding='corner')
        conv_3x3_in = ConvModule(
            in_channels=in_channels, out_channels=in_channels,padding=0,
            kernel_size=conv_ksize, stride=1, act_cfg=act_config, norm_cfg=norm_cfg_v, dilation=dilation
        )
        conv_1x1_in = ConvModule(
            in_channels=in_channels, out_channels=transformer_dim,padding=0,
            kernel_size=1, stride=1, act_cfg=act_config,norm_cfg=norm_cfg_v
        )

        conv_1x1_out = ConvModule(
             in_channels=transformer_dim, out_channels=in_channels,padding=0,
            kernel_size=1, stride=1, act_cfg=act_config,norm_cfg=norm_cfg_v
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvModule(
                in_channels=2 * in_channels, out_channels=in_channels,padding=0,
                kernel_size=conv_ksize, stride=1, act_cfg=act_config, norm_cfg=norm_cfg_v
            )
        
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="pad", module=self.pad)
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)
        
        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks
        transformer_config=dict(
            type="BaseTransformerLayer",
             attn_cfgs=dict(
                 type="MultiheadAttention",
                 embed_dims=transformer_dim,
                 num_heads=num_heads,
                 attn_drop=attn_dropout,
                 proj_drop=ffn_dropout,
                 dropout_layer=dict(type='Dropout', drop_prob=dropout),
                 batch_first=True,
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=transformer_dim,
                feedforward_channels=ffn_dim,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='ReLU', inplace=True),
                 ),
            operation_order=('norm','self_attn', 'norm', 'ffn','norm'),
            norm_cfg=dict(type='LN'),
            batch_first=True,)

        global_rep = build_transformer_layer(transformer_config)
        self.global_rep = global_rep

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.ffn_max_dim = ffn_dims[0]
        self.ffn_min_dim = ffn_dims[-1]
        self.var_ffn = var_ffn
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n\tconv_in_dim={}, conv_out_dim={}, dilation={}, conv_ksize={}".format(self.cnn_in_dim, self.cnn_out_dim, self.dilation, self.conv_ksize)
        repr_str += "\n\tpatch_h={}, patch_w={}".format(self.patch_h, self.patch_w)
        repr_str += "\n\ttransformer_in_dim={}, transformer_n_heads={}, transformer_ffn_dim={}, dropout={}, " \
                    "ffn_dropout={}, attn_dropout={}, blocks={}".format(
            self.cnn_out_dim,
            self.n_heads,
            self.ffn_dim,
            self.dropout,
            self.ffn_dropout,
            self.attn_dropout,
            self.n_blocks
        )
        if self.var_ffn:
            repr_str += "\n\t var_ffn_min_mult={}, var_ffn_max_mult={}".format(
                self.ffn_min_dim, self.ffn_max_dim
            )

        repr_str += "\n)"
        return repr_str

    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x: Tensor) -> Tensor:
        res = x
        fm = self.local_rep(x)
        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(
                self.pad(torch.cat((res, fm), dim=1))
            )
        
        return fm

head_dim=32
num_heads=4
mv2_exp_mult = 2
LayersConfig={
        "layer1": {
                "type":"mobilenet2",
                "out_channels": 16,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                
            },
        "layer2": {
                "type":"mobilenet2",
                "out_channels": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                
            },
        "layer3": {  # 28x28
                "type":"mobilevit",
                "out_channels": 48,
                "head_dim": 16,
                "ffn_dim": 128,
                "n_transformer_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": num_heads,
                
            },
        "layer4": {  # 14x14
                "type":"mobilevit",
                "out_channels": 64,
                "head_dim": 20,
                "ffn_dim": 160,
                "n_transformer_blocks": 4,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": num_heads,
                
            },
        "layer5": {  # 7x7
                "type":"mobilevit",
                "out_channels": 80,
                "head_dim": 24,
                "ffn_dim": 192,
                "n_transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": num_heads,
                
            },
    }

@BACKBONES.register_module()
class MobileViT(BaseModule):
    """
        MobileViT: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self,
                 Layers_config=LayersConfig,
                 out_indices=(1, 2, 4),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Swish'),
                 ffn_droput=0.0,
                 attn_dropout=0.0,
                 dropout=0.1,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm']),
                         dict(type='TruncNormal',layer=['Linear'])],
                  *args, **kwargs) -> None:
     
        super(MobileViT, self).__init__(init_cfg)
        image_channels = 3
        out_channels = 16
        self.act_cfg=act_cfg
        self.out_indices=out_indices


        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the classification backbones
        # We allow that using `output_stride` arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        super(MobileViT, self).__init__()
        self.dilation = 1

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.pad=AdaptivePadding( kernel_size=3, stride=1, dilation=1, padding='corner')
        self.conv1 = ConvModule(
            in_channels=image_channels, out_channels=out_channels,
            kernel_size=3, stride=2, act_cfg=act_cfg, norm_cfg=norm_cfg
        )

        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': out_channels}

    
        self.layers=ModuleList()
        
        for config in Layers_config.keys():
            in_channels = out_channels
            self.layer, out_channels = self._make_layer(input_channel=in_channels, cfg=Layers_config[config],dropout=dropout,attn_dropout=attn_dropout,ffn_dropout=ffn_droput)
            self.model_conf_dict[config] = {'in': in_channels, 'out': out_channels}
            self.layers.append(self.layer)
        
        in_channels = out_channels
        exp_channels = min(2 * in_channels, 128)
        self.conv_1x1_exp = ConvModule( in_channels=in_channels, out_channels=exp_channels,kernel_size=1, stride=1, act_cfg=act_cfg, norm_cfg=norm_cfg)
        # check model
        #self.check_model()

        # weight initialization
        #self.reset_parameters(opts=opts)

    def _make_layer(self, input_channel, cfg: Dict,dropout,attn_dropout,ffn_dropout, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        copied_cfg = cfg.copy()
        block_type = copied_cfg.pop("type","mobilenet")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                input_channel=input_channel,
                cfg=copied_cfg,
                dropout=dropout,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                input_channel=input_channel,
                cfg=copied_cfg
            )

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.pop("num_blocks",1)
        stride=cfg.pop("stride",1)
        block = []

        for i in range(num_blocks):
            stride=stride if i==0 else 1
            layer = InvertedResidual(in_channels=input_channel,stride=stride,**cfg)
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, input_channel, cfg: Dict,dropout,attn_dropout,ffn_dropout, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        block.append(
            MobileViTBlock(
                in_channels=input_channel,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                attn_dropout=attn_dropout,
                **cfg
            )
        )

        return nn.Sequential(*block), input_channel
    
    def forward(self, x):
        x=self.pad(x)
        x = self.conv1(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
            
        
        #out=self.conv_1x1_exp(x)
        return outs