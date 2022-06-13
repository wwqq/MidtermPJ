import math
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.runner import _load_checkpoint
from mmseg.utils import get_root_logger

from ..builder import BACKBONES


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        ks: int,
        stride: int,
        expand_ratio: int,
        activations = None,
        norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks//2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class StackedMV2Block(nn.Module):
    def __init__(
            self,
            cfgs,
            stem,
            inp_channel=16,
            activation=nn.ReLU,
            norm_cfg=dict(type='BN', requires_grad=True),
            width_mult=1.):
        super().__init__()
        self.stem = stem
        if stem:
            self.stem_block = nn.Sequential(
                Conv2d_BN(3, inp_channel, 3, 1, 1, norm_cfg=norm_cfg),
                activation()
            )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg,
                                     activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        if self.stem:
            x = self.stem_block(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1) # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer, norm_cfg=norm_cfg)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                norm_cfg=dict(type='BN2d', requires_grad=True), 
                act_layer=None):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,
                act_layer=act_layer))

    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return 
class Transformer(nn.Module):
    def __init__(self, cfgs,
                 channels,
                 emb_dims,
                 key_dims,
                 depths=[2,2],
                 num_heads=4,
                 attn_ratios=2,
                 mlp_ratios=[2, 4],
                 drop_path_rate=0.,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6,
                 num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.depths = depths
        self.cfgs = cfgs
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg

        for i in range(len(cfgs)):
            smb = StackedMV2Block(cfgs=cfgs[i], stem=True if i == 0 else False, inp_channel=channels[i], norm_cfg=norm_cfg)
            setattr(self, f"smb{i + 1}", smb)
            
        for i in range(len(depths)):
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths[i])]  # stochastic depth decay rule
            trans = BasicLayer(
                block_num=depths[i],
                embedding_dim=emb_dims[i],
                key_dim=key_dims,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
                norm_cfg=norm_cfg,
                act_layer=act_layer)
            setattr(self, f"trans{i + 1}", trans)
        self.linear = nn.Linear(channels[-1], num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        num_smb_stage = len(self.cfgs)
        num_trans_stage = len(self.depths)
        for i in range(num_smb_stage):
            smb = getattr(self, f"smb{i + 1}")
            x = smb(x)
            if num_trans_stage + i >= num_smb_stage:
                trans = getattr(self, f"trans{i + num_trans_stage - num_smb_stage + 1}")
                x = trans(x)
        out = self.avgpool(x).view(-1, x.shape[1])
        out = self.linear(out)
        return out
      
def transformer():
    model_cfgs = dict(
        cfg1=[
            # k,  t,  c, s
            [3, 6, 64, 1],
            [3, 6, 64, 1]],
        cfg2=[
            [5, 3, 128, 2]],
        cfg3=[
            [3, 3, 256, 2]],
        cfg4=[
            [5, 3, 512, 2]],
        channels=[64, 64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        depths=[2, 2, 6, 2],
        emb_dims=[64, 128, 256, 512],
        key_dims=64,
        drop_path_rate=0.1,
        attn_ratios=1,
        mlp_ratios=[4, 4, 4, 4])
    return Transformer(
        cfgs=[model_cfgs['cfg1'], model_cfgs['cfg2'], model_cfgs['cfg3'], model_cfgs['cfg4']],
        channels=model_cfgs['channels'],
        emb_dims=model_cfgs['emb_dims'],
        key_dims=model_cfgs['key_dims'],
        depths=model_cfgs['depths'],
        attn_ratios=model_cfgs['attn_ratios'],
        mlp_ratios=model_cfgs['mlp_ratios'],
        num_heads=model_cfgs['num_heads'],
        drop_path_rate=model_cfgs['drop_path_rate'])
  
  if __name__ == '__main__':
    model = transformer()
    input = torch.rand((1, 3, 32, 32))
    print(model)
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    model.eval()
    flops = FlopCountAnalysis(model, input)
    print(flop_count_table(flops))
