import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Dict, Tuple

from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn import ConvModule,build_norm_layer
from torch import nn, Tensor
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS
from mmcv.cnn.bricks.transformer import build_transformer_layer

from mmdet.models.builder import BACKBONES

from ..utils import AdaptivePadding

#@ACTIVATION_LAYERS.register_module()
class Swish(nn.SiLU):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__(inplace=inplace)

class MobileViTBlock(BaseModule):
    """
        MobileViT block: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self, in_channels: int, transformer_dim: int, ffn_dim: int,
                 n_transformer_blocks: Optional[int] = 2,
                 head_dim: Optional[int] = 32, attn_dropout: Optional[float] = 0.1,
                 dropout: Optional[int] = 0.1, ffn_dropout: Optional[int] = 0.1, patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8, transformer_norm_layer: Optional[str] = "layer_norm",
                 conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1, var_ffn: Optional[bool] = False,
                 no_fusion: Optional[bool] = False,act_config=dict(type="Swish"),norm_cfg=dict(type='LN'),
                 *args, **kwargs):
        
        super().__init__()
        norm_cfg_v=dict(type='BN', requires_grad=True)
        pad=AdaptivePadding( kernel_size=3, stride=1, dilation=1, padding='corner')
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
        self.local_rep.add_module(name="pad", module=pad)
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
        print(x.shape)

        fm = self.local_rep(x)
        print(fm.shape)
        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        print(patches.shape)
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(
                torch.cat((res, fm), dim=1)
            )
        return fm


@BACKBONES.register_module()
class MobileViT(BaseModule):
    """
        MobileViT: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self,
                 
                 widen_factor=1.,
                 out_indices=(1, 2, 4, 7),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.2)

        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        image_channels = 3
        out_channels = 16

        mobilevit_config = get_configuration(opts=opts)

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
        self.conv_1 = ConvLayer(
                opts=opts, in_channels=image_channels, out_channels=out_channels,
                kernel_size=3, stride=2, use_norm=True, use_act=True
            )

        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict['layer1'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict['layer2'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict['layer3'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer4"], dilate=dilate_l4
        )
        self.model_conf_dict['layer4'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer5"], dilate=dilate_l5
        )
        self.model_conf_dict['layer5'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
                opts=opts, in_channels=in_channels, out_channels=exp_channels,
                kernel_size=1, stride=1, use_act=True, use_norm=True
            )

        self.model_conf_dict['exp_before_cls'] = {'in': in_channels, 'out': exp_channels}

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False))
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="dropout", module=Dropout(p=classifier_dropout, inplace=True))
        self.classifier.add_module(
            name="fc",
            module=LinearLayer(in_features=exp_channels, out_features=num_classes, bias=True)
        )

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument('--model.classification.mit.mode', type=str, default=None,
                           choices=['xx_small', 'x_small', 'small'], help="MIT mode")
        group.add_argument('--model.classification.mit.attn-dropout', type=float, default=0.1,
                           help="Dropout in attention layer")
        group.add_argument('--model.classification.mit.ffn-dropout', type=float, default=0.0,
                           help="Dropout between FFN layers")
        group.add_argument('--model.classification.mit.dropout', type=float, default=0.1,
                           help="Dropout in Transformer layer")
        group.add_argument('--model.classification.mit.transformer-norm-layer', type=str, default="layer_norm",
                           help="Normalization layer in transformer")
        group.add_argument('--model.classification.mit.no-fuse-local-global-features', action="store_true",
                           help="Do not combine local and global features in MIT block")
        group.add_argument('--model.classification.mit.conv-kernel-size', type=int, default=3,
                           help="Kernel size of Conv layers in MIT block")

        group.add_argument('--model.classification.mit.head-dim', type=int, default=None,
                           help="Head dimension in transformer")
        group.add_argument('--model.classification.mit.number-heads', type=int, default=None,
                           help="No. of heads in transformer")
        return parser

    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg,
                dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            logger.error("Transformer input dimension should be divisible by head dimension. "
                         "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(
            MobileViTBlock(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(opts, "model.classification.mit.attn_dropout", 0.1),
                head_dim=head_dim,
                no_fusion=getattr(opts, "model.classification.mit.no_fuse_local_global_features", False),
                conv_ksize=getattr(opts, "model.classification.mit.conv_kernel_size", 3)
            )
        )

        return nn.Sequential(*block), input_channel