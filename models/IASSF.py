import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from get_hazy_density.show_threshold_1 import get_fog_intensity, fog_intensity_to_tensor


class RLN(nn.Module):

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad
        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)
        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)
        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_positions = relative_positions.permute(1, 2, 0).contiguous()
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
    return relative_positions_log

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)

        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm
        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)
        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        if self.use_attn:
            x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn:
            x = x * rescale + rebias
        x = identity + x

        identity = x
        if self.use_attn and self.mlp_norm:
            x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm:
            x = x * rescale + rebias
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i],
                             conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        out = torch.sum(in_feats * attn, dim=1)
        return out


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)

            attn_windows = self.attn(qkv)

            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)

            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out


class RestormerEncoder(nn.Module):
    def __init__(self, in_c, out_c, network_depth=4, num_heads=1, window_size=8, shift_size=0, conv_type=None):
        super(RestormerEncoder, self).__init__()
        dim = out_c
        LayerNorm_type = 'WithBias'
        ffn_expansion_factor = 2.66
        self.norm1 = RLN(dim, detach_grad=False)

        self.attn = Attention(
            network_depth=network_depth,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            use_attn=True,
            conv_type=conv_type
        )

        self.norm2 = RLN(dim, detach_grad=False)

        self.ffn = Mlp(
            network_depth=network_depth,
            in_features=dim,
            hidden_features=int(dim * ffn_expansion_factor)
        )

        self.mlp = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.mlp(x)
        identity = x
        x, rescale1, rebias1 = self.norm1(x)
        x = self.attn(x)
        x = x * rescale1 + rebias1
        x = identity + x
        identity = x
        x, rescale2, rebias2 = self.norm2(x)
        x = self.ffn(x)
        x = x * rescale2 + rebias2
        x = identity + x
        return x


class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=6, prompt_len=5, prompt_size=256, lin_dim=6):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.adjust_channels = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, vi_img, ir_img):
        B, C, H, W = vi_img.shape
        x = ir_img - vi_img
        x = self.conv1(x)
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
                 self.prompt_param.expand(B, -1, -1, -1, -1)
        prompt = prompt.squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear", align_corners=True)
        prompt = self.conv3x3(prompt)
        prompt = self.adjust_channels(prompt)
        return prompt



class PromptInBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, network_depth):
        super(PromptInBlock, self).__init__()
        concat_dim = dim * 2
        self.attention = Attention(network_depth=network_depth, dim=concat_dim, num_heads=num_heads,
                                   window_size=window_size, shift_size=0, use_attn=True, conv_type=None)
        self.conv1x1 = nn.Conv2d(concat_dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, feature, prompt):
        x = torch.cat([feature, prompt], dim=1)
        x = self.attention(x)
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        return x


class PromptGenerationAndEmbeddingBlock(nn.Module):
    def __init__(self, prompt_dim=24, prompt_len=5, prompt_size=256, lin_dim=24, dim=24, num_heads=1, window_size=8,
                 network_depth=4):
        super(PromptGenerationAndEmbeddingBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3_gen = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.attention = Attention(network_depth=network_depth, dim=dim + prompt_dim, num_heads=num_heads,
                                   window_size=window_size, shift_size=0, use_attn=True, conv_type=None)
        self.conv1x1_embed = nn.Conv2d(dim + prompt_dim, prompt_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3_embed = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, y):
        B, C, H, W = y.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb.float()), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        prompt = prompt * self.prompt_param.expand(B, -1, -1, -1, -1)
        prompt = prompt.squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3_gen(prompt)

        z = torch.cat([y, prompt], dim=1)
        z = self.attention(z)
        z = self.conv1x1_embed(z)
        z = self.conv3x3_embed(z)

        return z


class IASSF(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[8, 8, 8, 4, 4],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(IASSF, self).__init__()

        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.w_ir = nn.Parameter(torch.ones(1))
        self.w_vi = nn.Parameter(torch.ones(1))
        self.ir_scale_conv = nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=1, bias=False)
        self.vi_scale_conv = nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=1, bias=False)
        self.restormer_encoder = RestormerEncoder(in_chans, embed_dims[0])
        self.conv24_3 = nn.Conv2d(24, 3, 1, 1, 0)
        self.PromptGenerationAndEmbeddingBlock = PromptGenerationAndEmbeddingBlock(prompt_size=256,
                                                                                   lin_dim=embed_dims[0],
                                                                                   prompt_dim=embed_dims[0],
                                                                                   prompt_len=5,
                                                                                   dim=embed_dims[0],
                                                                                   num_heads=num_heads[0],
                                                                                   window_size=window_size,
                                                                                   network_depth=sum(depths))

        self.patch_embed = PatchEmbed(patch_size=1, in_chans=embed_dims[0], embed_dim=embed_dims[0], kernel_size=3)
        self.skip0 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0], num_heads=num_heads[0],
                                 mlp_ratio=mlp_ratios[0], norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1], num_heads=num_heads[1],
                                 mlp_ratio=mlp_ratios[1], norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2], num_heads=num_heads[2],
                                 mlp_ratio=mlp_ratios[2], norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])
        self.patch_split1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])
        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3], num_heads=num_heads[3],
                                 mlp_ratio=mlp_ratios[3], norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])
        self.patch_split2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])
        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4], num_heads=num_heads[4],
                                 mlp_ratio=mlp_ratios[4], norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])
        self.fusion3 = SKFusion(embed_dims[4])
        self.patch_unembed = PatchUnEmbed(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward_features(self, x):
        x = self.patch_embed(x)
        skip0 = self.skip0(x)
        x = self.layer1(x)
        skip1 = x
        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x
        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)
        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)
        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.fusion3([x, skip0]) + x
        x = self.patch_unembed(x)
        return x

    def forward(self, vi_image, ir_image):
        H, W = vi_image.shape[2:]
        vi_image = self.check_image_size(vi_image)
        ir_image = self.check_image_size(ir_image)
        vi_feature = self.restormer_encoder(vi_image)
        ir_feature = self.restormer_encoder(ir_image)
        ir_feature_dehaze = self.conv24_3(ir_feature)
        difference_feature = vi_feature - ir_feature
        ir_feature = self.PromptGenerationAndEmbeddingBlock(difference_feature, ir_feature)
        vi_fog_intensity = fog_intensity_to_tensor(vi_feature)
        combined_feature = ir_feature * vi_fog_intensity + vi_feature * (1 - vi_fog_intensity) + vi_feature
        dehaze_feature = self.forward_features(combined_feature)

        return dehaze_feature[:, :, :H, :W], ir_feature_dehaze[:, :, :H, :W]


class RestormerUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, base_channels=6):
        super(RestormerUNet, self).__init__()

        self.prompt_gen_block = PromptGenBlock(prompt_dim=base_channels, lin_dim=in_channels)
        self.prompt_in_blocks_enc = nn.ModuleList([
            PromptInBlock(dim=base_channels * (2 ** i), num_heads=1, window_size=8, network_depth=4)
            for i in range(3)
        ])
        self.prompt_in_blocks_dec = nn.ModuleList([
            PromptInBlock(dim=base_channels * (2 ** i), num_heads=1, window_size=8, network_depth=4)
            for i in reversed(range(2))
        ])

        self.encoder1 = RestormerEncoder(in_c=in_channels, out_c=base_channels)
        self.encoder2 = RestormerEncoder(in_c=base_channels, out_c=base_channels * 2)
        self.encoder3 = RestormerEncoder(in_c=base_channels * 2, out_c=base_channels * 4)

        self.decoder3 = RestormerEncoder(in_c=base_channels * 4, out_c=base_channels * 2)
        self.decoder2 = RestormerEncoder(in_c=base_channels * 2, out_c=base_channels)

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.skip_conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1)

        self.prompt_adjust_conv_enc = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels * (2 ** i), kernel_size=1, bias=False)
            for i in range(3)
        ])
        self.prompt_adjust_conv_dec = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels * (2 ** i), kernel_size=1, bias=False)
            for i in reversed(range(2))
        ])

    def forward(self, vi_img, ir_img):
        x = torch.cat([vi_img, ir_img], dim=1)
        prompt = self.prompt_gen_block(vi_img, ir_img)
        enc1 = self.encoder1(x)
        prompt_enc1 = self.prompt_adjust_conv_enc[0](prompt)
        enc1 = self.prompt_in_blocks_enc[0](enc1, prompt_enc1) + enc1

        enc2 = self.encoder2(self.downsample(enc1))
        prompt_enc2 = F.interpolate(prompt, scale_factor=0.5, mode='bilinear', align_corners=True)
        prompt_enc2 = self.prompt_adjust_conv_enc[1](prompt_enc2)
        enc2 = self.prompt_in_blocks_enc[1](enc2, prompt_enc2) + enc2

        enc3 = self.encoder3(self.downsample(enc2))
        prompt_enc3 = F.interpolate(prompt, scale_factor=0.25, mode='bilinear', align_corners=True)
        prompt_enc3 = self.prompt_adjust_conv_enc[2](prompt_enc3)
        enc3 = self.prompt_in_blocks_enc[2](enc3, prompt_enc3) + enc3

        dec3 = self.upsample(enc3)
        dec3 = self.decoder3(dec3)
        prompt_dec3 = F.interpolate(prompt, scale_factor=0.5, mode='bilinear', align_corners=True)
        prompt_dec3 = self.prompt_adjust_conv_dec[0](prompt_dec3)
        dec3 = self.prompt_in_blocks_dec[0](dec3, prompt_dec3) + dec3
        enc2_adjusted = self.skip_conv2(enc2)
        dec3 = dec3 + enc2_adjusted

        dec2 = self.upsample(dec3)
        dec2 = self.decoder2(dec2)
        prompt_dec2 = prompt
        prompt_dec2 = self.prompt_adjust_conv_dec[1](prompt_dec2)
        dec2 = self.prompt_in_blocks_dec[1](dec2, prompt_dec2) + dec2
        enc1_adjusted = self.skip_conv1(enc1)
        dec2 = dec2 + enc1_adjusted

        output = self.final_conv(dec2)

        return output

def iassf():
    return IASSF(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])
