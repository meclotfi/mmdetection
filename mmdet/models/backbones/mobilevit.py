#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from cv2 import norm
from torch import nn
import argparse
from typing import Dict, Tuple, Optional
from torch import nn, Tensor

from utils import logger
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mobilenet.layers import norm_layers_tuple
from mobilenet.misc.profiler import module_profile
from mobilenet.misc.init_utils import initialize_weights
from mobilenet.layers import ConvLayer, LinearLayer, GlobalPool, Dropout, SeparableConv
from mobilenet.modules import InvertedResidual, MobileViTBlock


@PLUGIN_LAYERS.register_module("Swish")
class Swish(nn.SiLU):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__(inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple(Tensor, float, float):
        return input, 0.0, 0.0
d_opts={
    "model":{
         "classification":{
                            "name": "mobilevit",
                            "classifier_dropout": 0.1,
                            "mit":{
                                    "mode": "small",
                                    "ffn_dropout": 0.0,
                                    "attn_dropout": 0.0,
                                    "dropout": 0.1,
                                    "number_heads": 4,
                                    "no_fuse_local_global_features": False,
                                    "conv_kernel_size": 3,
                                },
                            "activation":{"name": "swish"},
                            "pretrained":"https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt"
                           },
        "normalization":{
            "name": "sync_batch_norm",
            "momentum": 0.1
            },
        "activation":{
            "name": "relu", 
            "inplace": False
        },
        "layer":{
            "global_pool": "mean",
            "conv_init": "kaiming_normal",
            "linear_init": "normal",
            "conv_weight_std": False
        }
}
}

mv2_exp_mult = 4
head_dim=32
config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2"
            },
            "layer2": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2"
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": 4,
                "num_heads": 4,
                "block_type": "mobilevit"
            },
            "layer4": {  # 14x14
                "out_channels": 128,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": 4,
                "block_type": "mobilevit"
            },
            "layer5": {  # 7x7
                "out_channels": 160,
                "transformer_channels": 240,
                "ffn_dim": 480,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": 4,
                "block_type": "mobilevit"
            },
            "last_layer_exp_factor": 4
        }
def parameter_list(named_parameters, weight_decay: float = 0.0, no_decay_bn_filter_bias: bool = False):
    with_decay = []
    without_decay = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if param.requires_grad and len(param.shape) == 1 and no_decay_bn_filter_bias:
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                elif param.requires_grad:
                    with_decay.append(param)
    else:
        for p_name, param in named_parameters():
            if param.requires_grad and len(param.shape) == 1 and no_decay_bn_filter_bias:
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
            elif param.requires_grad:
                with_decay.append(param)
    param_list = [{'params': with_decay, 'weight_decay': weight_decay}]
    if len(without_decay) > 0:
        param_list.append({'params': without_decay, 'weight_decay': 0.0})
    return param_list








@BACKBONES.register_module()
class MobileViT(BaseModule):
    """
        MobileViT: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self,
        out_indices=(1, 2, 3, 4),
        opts=d_opts,
        mobilevit_config=config,
        act_cfg=dict(type="Swish"),
        norm_cfg=dict(type='LN'),
        pretrained="https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pth",
        *args, **kwargs) -> None:
        

        self.out_indices=out_indices
        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = None
        else:
            raise TypeError('pretrained must be a str or None')

        super(MobileViT, self).__init__(init_cfg=self.init_cfg)

        #num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = classifier_dropout

        pool_type = "mean"
        image_channels = 3
        out_channels = 16

        mobilevit_config = mobilevit_config

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
        self.conv_1 = ConvModule(
                opts=opts, in_channels=image_channels, out_channels=out_channels,
                kernel_size=3, stride=2,act_cfg=act_cfg, norm_cfg=norm_cfg
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

        self.model_conf_dict['exp_before_cls'] = {'in': in_channels, 'out': exp_channels}

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
    
    def check_model(self):
        assert self.model_conf_dict, "Model configuration dictionary should not be empty"
        assert self.conv_1 is not None, 'Please implement self.conv_1'
        assert self.layer_1 is not None, 'Please implement self.layer_1'
        assert self.layer_2 is not None, 'Please implement self.layer_2'
        assert self.layer_3 is not None, 'Please implement self.layer_3'
        assert self.layer_4 is not None, 'Please implement self.layer_4'
        assert self.layer_5 is not None, 'Please implement self.layer_5'
        assert self.conv_1x1_exp is not None, 'Please implement self.conv_1x1_exp'
        assert self.classifier is not None, 'Please implement self.classifier'

    def reset_parameters(self, opts):
        initialize_weights(opts=opts, modules=self.modules())

    def extract_end_points_all(self, x: Tensor, use_l5: Optional[bool] = True, use_l5_exp: Optional[bool] = False) -> Dict:
        out_dict = {} # Use dictionary over NamedTuple so that JIT is happy
        x = self.conv_1(x)  # 112 x112
        x = self.layer_1(x)  # 112 x112
        out_dict["out_l1"] = x

        x = self.layer_2(x)  # 56 x 56
        out_dict["out_l2"] = x

        x = self.layer_3(x)  # 28 x 28
        out_dict["out_l3"] = x

        x = self.layer_4(x)  # 14 x 14
        out_dict["out_l4"] = x

        if use_l5:
            x = self.layer_5(x)  # 7 x 7
            out_dict["out_l5"] = x

            if use_l5_exp:
                x = self.conv_1x1_exp(x)
                out_dict["out_l5_exp"] = x
        return out_dict
    def extract_end_points_l4(self, x: Tensor) -> Dict:
        return self.extract_end_points_all(x, use_l5=False)

    def extract_features(self, x: Tensor):
        outs=[]
        x = self.conv_1(x)
        if 0 in self.out_indices:
            outs.append(x)
        x = self.layer_1(x)
        if 1 in self.out_indices:
            outs.append(x)
        x = self.layer_2(x)
        if 2 in self.out_indices:
            outs.append(x)
        x = self.layer_3(x)
        if 3 in self.out_indices:
            outs.append(x)
        x = self.layer_4(x)
        if 4 in self.out_indices:
            outs.append(x)
        x = self.layer_5(x)
        if 5 in self.out_indices:
            outs.append(x)
        return outs

    def forward(self, x: Tensor):
        return self.extract_features(x)

    def freeze_norm_layers(self):
        for m in self.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

    def get_trainable_parameters(self, weight_decay: float = 0.0, no_decay_bn_filter_bias: bool = False):
        param_list = parameter_list(named_parameters=self.named_parameters,
                                    weight_decay=weight_decay,
                                    no_decay_bn_filter_bias=no_decay_bn_filter_bias)
        return param_list, [1.0] * len(param_list)

    @staticmethod
    def _profile_layers(layers, input, overall_params, overall_macs):
        if not isinstance(layers, list):
            layers = [layers]

        for layer in layers:
            if layer is None:
                continue
            input, layer_param, layer_macs = module_profile(module=layer, x=input)

            overall_params += layer_param
            overall_macs += layer_macs

            if isinstance(layer, nn.Sequential):
                module_name = "\n+".join([l.__class__.__name__ for l in layer])
            else:
                module_name = layer.__class__.__name__
            print(
                '{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M'.format(module_name,
                                                                          'Params',
                                                                          round(layer_param / 1e6, 3),
                                                                          'MACs',
                                                                          round(layer_macs / 1e6, 3)
                                                                          ))
            logger.singe_dash_line()
        return input, overall_params, overall_macs

    def profile_model(self, input: Tensor, is_classification: bool = True) -> Tuple(Tensor or Dict[Tensor], float, float):
        # Note: Model profiling is for reference only and may contain errors.
        # It relies heavily on the user to implement the underlying functions accurately.
        overall_params, overall_macs = 0.0, 0.0

        if is_classification:
            logger.log('Model statistics for an input of size {}'.format(input.size()))
            logger.double_dash_line(dashes=65)
            print('{:>35} Summary'.format(self.__class__.__name__))
            logger.double_dash_line(dashes=65)

        out_dict = {}
        input, overall_params, overall_macs = self._profile_layers([self.conv_1, self.layer_1], input=input, overall_params=overall_params, overall_macs=overall_macs)
        out_dict["out_l1"] = input

        input, overall_params, overall_macs = self._profile_layers(self.layer_2, input=input,
                                                                   overall_params=overall_params,
                                                                   overall_macs=overall_macs)
        out_dict["out_l2"] = input

        input, overall_params, overall_macs = self._profile_layers(self.layer_3, input=input,
                                                                   overall_params=overall_params,
                                                                   overall_macs=overall_macs)
        out_dict["out_l3"] = input

        input, overall_params, overall_macs = self._profile_layers(self.layer_4, input=input,
                                                                   overall_params=overall_params,
                                                                   overall_macs=overall_macs)
        out_dict["out_l4"] = input

        input, overall_params, overall_macs = self._profile_layers(self.layer_5, input=input,
                                                                   overall_params=overall_params,
                                                                   overall_macs=overall_macs)
        out_dict["out_l5"] = input

        if self.conv_1x1_exp is not None:
            input, overall_params, overall_macs = self._profile_layers(self.conv_1x1_exp, input=input,
                                                                       overall_params=overall_params,
                                                                       overall_macs=overall_macs)
            out_dict["out_l5_exp"] = input

        if is_classification:
            classifier_params, classifier_macs = 0.0, 0.0
            if self.classifier is not None:
                input, classifier_params, classifier_macs = module_profile(module=self.classifier, x=input)
                print('{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M'.format('Classifier',
                                                                                'Params',
                                                                                round(classifier_params / 1e6, 3),
                                                                                'MACs',
                                                                                round(classifier_macs / 1e6, 3)))
            overall_params += classifier_params
            overall_macs += classifier_macs

            logger.double_dash_line(dashes=65)
            print('{:<20} = {:>8.3f} M'.format('Overall parameters', overall_params / 1e6))
            # Counting Addition and Multiplication as 1 operation
            print('{:<20} = {:>8.3f} M'.format('Overall MACs', overall_macs / 1e6))
            overall_params_py = sum([p.numel() for p in self.parameters()])
            print('{:<20} = {:>8.3f} M'.format('Overall parameters (sanity check)', overall_params_py / 1e6))
            logger.double_dash_line(dashes=65)

        return out_dict, overall_params, overall_macs