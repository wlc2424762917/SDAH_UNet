import sys
sys.path.append("/home/lichao/segmentation_framework")
from torch import nn
import torch
import numpy as np
from model.networks.neural_network import SegmentationNetwork
import torch.nn.functional
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import einops

# ------------------------------
# DATransformer Basic
# ------------------------------


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


def reverse_deform_point(points):
    """
    Args:
        points: (B, C, Hk, Wk, 2)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W, O)
    """
    x = torch.zeros_like(points)
    x[:, :, :, :, 1] = - points[:, :, :, :, 0]
    x[:, :, :, :, 0] = points[:, :, :, :, 1]

    return x


def window_reverse_deform_point(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size, coordinate)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W, O)
    """
    device = windows.device
    nWH = H // window_size  # number of windows in H direction
    nWW = W // window_size  # number of windows in W direction
    assert (H % window_size == 0) or (W % window_size == 0), "H and W should be the integer multiple of window size!"

    B = int(windows.shape[0] / (nWH * nWW))
    #C = windows.shape[1]  # batch number# batch number
    C = 1
    x = windows.reshape(B, C, nWH, nWW, window_size, window_size, 2)
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().reshape(B, C, H, W, 2)

    window_coodinate_H_1d = np.linspace(-1, 1, nWH, endpoint=False) + (1 / nWH)
    window_coodinate_W_1d = np.linspace(-1, 1, nWW, endpoint=False) + (1 / nWW)
    X, Y = np.meshgrid(window_coodinate_W_1d, window_coodinate_H_1d)
    window_coodinate_2d = np.stack((Y, X), axis=2)
    window_coodinate_2d = torch.from_numpy(window_coodinate_2d).to(device)

    x[:, :, :, :, 0] = (x[:, :, :, :, 0].clone() / (nWH))
    x[:, :, :, :, 1] = (x[:, :, :, :, 1].clone() / (nWW))

    xx = torch.zeros_like(x)
    for idx_nWH in range(nWH):
        for idx_nWW in range(nWW):
            xx[:, :, idx_nWH * window_size:(idx_nWH + 1) * window_size,
            idx_nWW * window_size:(idx_nWW + 1) * window_size, 1] \
                = - x[:, :, idx_nWH * window_size:(idx_nWH + 1) * window_size,
                    idx_nWW * window_size:(idx_nWW + 1) * window_size, 0] \
                  - window_coodinate_2d[idx_nWH, idx_nWW, 0]

            xx[:, :, idx_nWH * window_size:(idx_nWH + 1) * window_size,
            idx_nWW * window_size:(idx_nWW + 1) * window_size, 0] \
                = x[:, :, idx_nWH * window_size:(idx_nWH + 1) * window_size,
                  idx_nWW * window_size:(idx_nWW + 1) * window_size, 1] \
                  + window_coodinate_2d[idx_nWH, idx_nWW, 1]

    return xx


# ------------------------------
# Swin Deformable Attention Transformer Basic(attn+block)
# ------------------------------

class SwinDAttention(nn.Module):
    r""" Shift Windows Deformable Attention

    Args:
        q_size(tuple[int]): Size if query. Here is the window size.
        kv_size(tuple[int]): Size if key and value. Here is the window size.
        dim (int): Number of input channels.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        window_size (tuple[int]): Window size for self-attention.
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        stride (int): Stride in offset calculation network
        offset_range_factor (int): Offset range factor in offset calculation network
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, q_size, kv_size, dim, n_head, n_group, window_size,
                 attn_drop, proj_drop, stride, offset_range_factor,
                 use_pe, dwc_pe, no_off, fixed_pe):
        super().__init__()

        self.dim = dim  # input channel
        self.window_size = window_size  # window height Wh, Window width Ww
        self.n_head = n_head  # number of head
        self.n_head_channels = self.dim // self.n_head  # head_dim
        self.scale = self.n_head_channels ** -0.5

        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size

        self.n_group = n_group
        self.n_group_channels = self.dim // self.n_group
        self.n_group_heads = self.n_head // self.n_group

        self.dwc_pe = dwc_pe
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.kk = 3

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kk, stride, self.kk // 2,
                      groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, groups=self.dim)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.q_h * self.q_w, self.kv_h * self.kv_w))
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.kv_h * 2 - 1, self.kv_w * 2 - 1))
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device))
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_group, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x, window_size, mask=None, dw=None):
        H = window_size
        W = window_size
        B, N, C = x.size()
        dtype, device = x.dtype, x.device
        assert H * W == N, "input feature has wrong size"

        x = einops.rearrange(x, 'b (h w) c-> b c h w', h=H, w=W)
        # calculate query
        q = self.proj_q(x)  # B C H W
        # resize query
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_group, c=self.n_group_channels)

        # use query to calculate offset
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        # get the size of offset
        Hk, Wk = offset.size(2), offset.size(3)
        # sample number
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # resize offset
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # use the number of offset point and batch size to get reference point
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        # no offset
        if self.no_off:
            offset = torch.zeros_like(offset)

        # offset + ref
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_group, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # embedding query,key,valuse B,C,H,W --> B*head,head_channel,HW
        q = q.reshape(B * self.n_head, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)

        # Q&K
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        # use position encoding
        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_head, self.n_head_channels,
                                                                              H * W)
            # fix
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_head, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(
                            B * self.n_group, H * W, 2).unsqueeze(2)
                        - pos.reshape(B * self.n_group, n_sample, 2).unsqueeze(1)
                ).mul(0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_group, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_head, H * W, n_sample)
                attn = attn + attn_bias

        if mask is not None:
            attn = attn.view(-1, self.n_head, H * W, n_sample)
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.n_head, H * W, n_sample) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_head, H * W, n_sample)
            attn = attn.view(-1, H * W, n_sample)

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        """if self.use_pe and self.dwc_pe:
            out = out + residual_lepe"""
        out = out.reshape(B, C, H, W)

        # parallel conv
        if dw is not None:
            out = out + dw.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos, reference, 0

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, n_head={self.n_head}, n_group={self.n_group}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # self.proj_q(x)
        flops += N * self.dim * self.dim * 1
        if not self.no_off:
            # self.conv_offset(q_off)
            flops += self.n_group * N * self.n_group_channels * (
                    self.n_group_channels / self.n_group_channels) * self.kk * self.kk
            flops += self.n_group * N * self.n_group_channels
            flops += self.n_group * N * self.n_group_channels * 2 * 1
        # self.proj_k(x_sampled)
        flops += N * self.dim * self.dim * 1
        # self.proj_v(x_sampled)
        flops += N * self.dim * self.dim * 1
        # torch.einsum('b c m, b c n -> b m n', q, k)
        flops += self.n_group * N * self.n_group_channels * N
        # torch.einsum('b m n, b c n -> b c m', attn, v)
        flops += self.n_group * N * N * self.n_group_channels
        # self.proj_drop(self.proj_out(out))
        flops += N * self.dim * self.dim * 1
        return flops

    # TODO
    def params(self):
        pass


class SwinDATransformerBlock(nn.Module):
    r""" Swin Deformable Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LaerNorm
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, n_head, n_group, window_size, shift_size,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.n_head = n_head
        self.n_head_channels = dim // n_head
        self.n_group = n_group
        self.window_size = window_size
        self.q_h, self.q_w = to_2tuple(window_size)
        self.kv_h, self.kv_w = to_2tuple(window_size)
        self.shift_size = shift_size

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        # parallel conv and attention
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = SwinDAttention(
            q_size=to_2tuple(window_size), kv_size=to_2tuple(window_size),
            dim=dim, n_head=n_head, n_group=n_group, window_size=to_2tuple(window_size),
            attn_drop=attn_drop, proj_drop=drop, stride=1, offset_range_factor=2,
            use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp is removed, otherwise 2 mlp in 1 layer

    def forward(self, x, mask_matrix, x_size):
        H, W = x_size  # x (batch_in_each_GPU, H*W, embedding_channel)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # x (batch_in_each_GPU, H*W, embedding_channel)
        x = x.view(B, H, W, C)  # x (batch_in_each_GPU, embedding_channel, H, W)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        # padding x, in order to concatenate with conv output
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x  # shifted_x (batch_in_each_GPU, embedding_channel, H, W)
            attn_mask = None

        dw = shifted_x.permute(0, 3, 1, 2).contiguous()
        dw = self.dwconv(dw)
        dw = dw.permute(0, 2, 3, 1).contiguous()
        dw = window_partition(dw, self.window_size)  # nW*B, window_size, window_size, C
        dw = dw.view(-1, self.window_size, self.window_size,
                     C)

        # partition windows
        x_windows = window_partition(shifted_x,
                                     self.window_size)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # (nW*B, window_size*window_size, C)  nW:number of Windows



        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows, position, reference, ___ = self.attn(x_windows, self.window_size, mask=attn_mask, dw=dw)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C shifted_x  (batch_in_each_GPU, embedding_channel, H, W)

        # Running only when batch_size=1 (test) to accelerate.
        # Comment out this to calculation the time cost.
        # if shifted_x.shape[0] == 1:
             # position = window_reverse_deform_point(position.detach().clone(), self.window_size, H, W)
             # reference = window_reverse_deform_point(reference.detach().clone(), self.window_size, H, W)
             # attn_map = window_reverse_attn_map(attn_map.detach().clone(), self.window_size, H, W)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # Comment out this to calculation the time cost.
            # position = torch.roll(position, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            # reference = torch.roll(reference, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            # TODO ATTN_MAP SHIFT
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)  # x (batch_in_each_GPU, H*W, embedding_channel)

        # residual
        x = shortcut + self.drop_path(x)
        #  FFN removed
        # x = x.reshape(B, H, W, C)

        # return x, position, reference, 0  # x (batch_in_each_GPU, H*W, embedding_channel)
        return x, 0, 0, 0  # x (batch_in_each_GPU, H*W, embedding_channel)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, n_head={self.n_head}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size  # nW: number of windows
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    # TODO
    def params(self):
        pass


# ------------------------------
# Swin Transformer Basic(attn+Block)
# ------------------------------

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        n_head (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, n_head, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim  # input channel
        self.window_size = window_size  # window height Wh, Window width Ww
        self.n_head = n_head  # number of heads
        head_dim = dim // n_head  # head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_head))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, window_size=None, mask=None, dw=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        H = window_size
        W = window_size
        B_, N, C = x.shape  # B: number of Windows * Batch_size in a GPU  N: patch number in a window  C:  embedding channel
        assert H * W == N, "input feature has wrong size"

        qkv = self.qkv(x).reshape(B_, N, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q,k,v: (number of Windows * Batch_size in a GPU, number of head, number of patch in a wondow, channe/number of head)
        # q,k,v (576,6,64,30) (number of Windows * Batch_size in a GPU, number of head, patch number in a window, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.n_head, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_head, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # parallel conv
        if dw is not None:
            x = x + dw.reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, 0, 0, 0

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, n_head={self.n_head}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += N * self.dim * N
        # x = (attn @ v)
        flops += N * N * self.dim
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

    # TODO
    def params(self):
        pass


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        n_head (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LaerNorm
    """

    def __init__(self, dim, input_resolution, n_head, window_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.n_head = n_head
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # parallel conv and attention
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=to_2tuple(self.window_size), n_head=n_head,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix, x_size):
        H, W = x_size  # x (batch_in_each_GPU, H*W, embedding_channel)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # x (batch_in_each_GPU, H*W, embedding_channel)
        x = x.view(B, H, W, C)  # x (batch_in_each_GPU, embedding_channel, H, W)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x  # shifted_x (batch_in_each_GPU, embedding_channel, H, W)
            attn_mask = None

        dw = shifted_x.permute(0, 3, 1, 2).contiguous()
        dw = self.dwconv(dw)
        dw = dw.permute(0, 2, 3, 1).contiguous()
        dw = window_partition(dw, self.window_size)  # nW*B, window_size, window_size, C
        dw = dw.view(-1, self.window_size, self.window_size,
                     C)

        # partition windows
        x_windows = window_partition(shifted_x,
                                     self.window_size)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # (nW*B, window_size*window_size, C)  nW:number of Windows

        attn_windows, position, reference, ___ = self.attn(x_windows, self.window_size, mask=attn_mask, dw=dw)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,
                                         C)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        shifted_x = window_reverse(attn_windows, self.window_size, H,
                                   W)  # B H' W' C shifted_x  (batch_in_each_GPU, embedding_channel, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)  # x (batch_in_each_GPU, H*W, embedding_channel)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # FFN
        # x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = shortcut + self.drop_path(x)
        return x, 0, 0, 0  # x (batch_in_each_GPU, H*W, embedding_channel), 0 0 0 aligned with swinDABlock

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, n_head={self.n_head}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size  # nW: number of windows
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    # TODO
    def params(self):
        pass


# ------------------------------
# DATransformer Basic
# ------------------------------

class DAttention(nn.Module):
    r""" Deformable Attention

    Args:
        q_size(tuple[int]): Size if query
        kv_size(tuple[int]): Size if key and value
        dim (int): Number of input channels.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        stride (int): Stride in offset calculation network
        offset_range_factor (int): Offset range factor in offset calculation network
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, q_size, kv_size, dim, n_head, n_group,
                 attn_drop, proj_drop, stride, offset_range_factor,
                 use_pe, dwc_pe, no_off, fixed_pe):
        super().__init__()

        self.dim = dim  # input channel
        self.n_head = n_head  # number of head
        self.n_head_channels = self.dim // self.n_head  # head_dim
        self.scale = self.n_head_channels ** -0.5

        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size

        self.n_group = n_group
        self.n_group_channels = self.dim // self.n_group
        self.n_group_heads = self.n_head // self.n_group

        self.dwc_pe = dwc_pe
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        if self.q_h <= 12 or self.q_w <= 12:
            self.kk = 3
        elif 13 <= self.q_h <= 24 or 13 <= self.q_w <= 24:
            self.kk = 5
        elif 25 <= self.q_h <= 48 or 25 <= self.q_w <= 48:
            self.kk = 7
        else:
            self.kk = 9

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kk, stride, self.kk // 2,
                      groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, groups=self.dim)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.q_h * self.q_w, self.kv_h * self.kv_w))
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.kv_h * 2 - 1, self.kv_w * 2 - 1))
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device))
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_group, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x, dw=None):
        dtype, device = x.dtype, x.device
        B, C, H, W = x.size()
        q = self.proj_q(x)  # B C H W
        # resize query
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_group, c=self.n_group_channels)

        # use query to calculate offset
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        # get the size of offset
        Hk, Wk = offset.size(2), offset.size(3)
        # sample number
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # resize offset
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # use the number of offset point and batch size to get reference point
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        # no offset
        if self.no_off:
            offset = torch.zeros_like(offset)

        # offset + ref
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_group, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # embedding query,key,valuse B,C,H,W --> B*head,head_channel,HW
        q = q.reshape(B * self.n_head, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)

        # Q&K
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        # use position encoding
        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_head, self.n_head_channels,
                                                                              H * W)
            # fix
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_head, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(
                            B * self.n_group, H * W, 2).unsqueeze(2)
                        - pos.reshape(B * self.n_group, n_sample, 2).unsqueeze(1)
                ).mul(0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_group, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_head, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        """if self.use_pe and self.dwc_pe:
            out = out + residual_lepe"""
        out = out.reshape(B, C, H, W)

        # parallel conv
        if dw is not None:
            out = out + dw.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, 0, 0, 0

    def extra_repr(self) -> str:
        return f'dim={self.dim}, n_head={self.n_head}, n_group={self.n_group}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # self.proj_q(x)
        flops += N * self.dim * self.dim * 1
        if not self.no_off:
            # self.conv_offset(q_off)
            flops += self.n_group * N * self.n_group_channels * (
                    self.n_group_channels / self.n_group_channels) * self.kk * self.kk
            flops += self.n_group * N * self.n_group_channels
            flops += self.n_group * N * self.n_group_channels * 2 * 1
        # self.proj_k(x_sampled)
        flops += N * self.dim * self.dim * 1
        # self.proj_v(x_sampled)
        flops += N * self.dim * self.dim * 1
        # torch.einsum('b c m, b c n -> b m n', q, k)
        flops += self.n_group * N * self.n_group_channels * N
        # torch.einsum('b m n, b c n -> b c m', attn, v)
        flops += self.n_group * N * N * self.n_group_channels
        # self.proj_drop(self.proj_out(out))
        flops += N * self.dim * self.dim * 1
        return flops

    # TODO
    def params(self):
        pass


class DATransformerBlock(nn.Module):
    r""" Deformable Attention Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LaerNorm
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, n_head, n_group, window_size, mlp_ratio=2., drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.n_head = n_head
        self.n_head_channels = dim // n_head
        self.n_group = n_group
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe
        self.norm1 = norm_layer(dim)
        self.attn = DAttention(
            q_size=input_resolution, kv_size=input_resolution, dim=dim, n_head=n_head, n_group=n_group,
            attn_drop=attn_drop, proj_drop=drop, stride=1, offset_range_factor=2,
            use_pe=False, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # parallel conv and attention
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

    def forward(self, x, mask_matrix, x_size):
        H, W = x_size  # x (batch_in_each_GPU, H*W, embedding_channel)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # x (batch_in_each_GPU, H*W, embedding_channel)
        x = x.view(B, H, W, C)  # B,L,C --> B,H,W,C

        dw = x.permute(0, 3, 1, 2).contiguous()  # B,H,W,C --> B,C,H,W
        dw = self.dwconv(dw)
        dw = dw.permute(0, 2, 3, 1).contiguous()
        dw = window_partition(dw, self.window_size)  # nW*B, window_size, window_size, C
        dw = dw.view(-1, H, W, C).permute(0, 3, 1, 2)

        x = x.view(-1, H, W, C).permute(0, 3, 1, 2)  # (nW*B, window_size*window_size, C)  nW:number of Windows
        x, position, reference, attn_map = self.attn(x, dw=dw)  # x (B,C,H,W) _&__ (batch_size, group, 32, 32, 2)

        x = x.view(-1, self.window_size, self.window_size, C)
        # position = reverse_deform_point(position)
        # reference = reverse_deform_point(reference)
        x = x.view(B, H * W, C)  # B,H,W,C --> B,HW,C

        # FFN
        x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, 0, 0, 0  # x (batch_in_each_GPU, H*W, embedding_channel)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, n_head={self.n_head}, " \
               f"n_group={self.n_group}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # D-MSA
        flops += self.attn.flops(H * W)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    # TODO
    def params(self):
        pass


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x


class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up = nn.ConvTranspose2d(dim, dim // 2, 2, 2)

    def forward(self, x, H, W):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.up(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 type,
                 depth,
                 num_heads,
                 n_group,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim = dim
        self.type = type
        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                type=type,
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                input_resolution=input_resolution,
                num_heads=num_heads,
                n_group=n_group,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                i_block=i
            )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # positions = []
        # references = []
        # for blk in self.blocks:
        for idx, blk in enumerate(self.blocks):
            x, position, reference = blk(x, attn_mask)
            # positions.append(position)
            # references.append((reference))
            #torch.save(position, './position{}_{}.pt'.format(idx, H))
            #torch.save(reference, './reference{}_{}.pt'.format(idx, H))

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            # return x, H, W, x_down, Wh, Ww, positions, references
            return x, H, W, x_down, Wh, Ww, 0, 0
        else:
            # return x, H, W, x_down, Wh, Ww, positions, references
            return x, H, W, x, H, W, 0, 0


class BasicLayer_up(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 type,
                 num_heads,
                 n_group,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim = dim

        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                input_resolution=input_resolution,
                num_heads=num_heads,
                type=type,
                n_group=n_group,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                i_block=i)
            for i in range(depth)])

        self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer)

    def forward(self, x, skip, H, W):
        x_up = self.Upsample(x, H, W)
        x = x_up + skip
        H, W = H * 2, W * 2
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0

        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x, position, reference = blk(x, attn_mask)

        return x, H, W


class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x


class project_up(nn.Module):
    def __init__(self, in_dim, out_dim, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_block = int(np.log2(patch_size[0]))
        self.project_block = []
        self.dim = [int(embed_dim) // (2 ** i) for i in range(self.num_block)]
        self.dim.append(in_chans)
        self.dim = self.dim[::-1]  # in_ch, embed_dim/2, embed_dim or in_ch, embed_dim/4, embed_dim/2, embed_dim

        for i in range(self.num_block)[:-1]:
            self.project_block.append(project(self.dim[i], self.dim[i + 1], 2, 1, nn.GELU, nn.LayerNorm, False))
        self.project_block.append(project(self.dim[-2], self.dim[-1], 2, 1, nn.GELU, nn.LayerNorm, True))
        self.project_block = nn.ModuleList(self.project_block)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, self.patch_size[0] - W % self.patch_size[0]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        for blk in self.project_block:
            x = blk(x)

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size=[224, 224],
                 patch_size=[4, 4],
                 in_chans=1,
                 embed_dim=96,
                 types=['k', 'k', 'k', 'k'],
                 depths=[3, 3, 3, 3],
                 num_heads=[3, 6, 12, 24],
                 n_groups=[1, 3, 6, 6],
                 window_size=[7, 7, 14, 7],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        assert len(depths) == len(types), "number of types and number of depths not match!"
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1] // patch_size[1] // 2 ** i_layer),
                type=types[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                n_group=n_groups[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        down = []
        Wh, Ww = x.size(2), x.size(3)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww, positions, references = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = x_out.permute(0, 2, 3, 1)
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                down.append(out)
            # torch.save(positions, 'position_layer{}.pt'.format(i))
            # torch.save(references, 'reference_layer{}.pt'.format(i))
        return down


class decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=[4, 4],
                 depths=[3, 3, 3],
                 types=['k', 'k', 'k', 'k'],
                 num_heads=[24, 12, 6],
                 n_groups=[6, 6, 3, 1],
                 window_size=[14, 7, 7],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths) - i_layer - 1)),
                depth=depths[i_layer],
                type=types[i_layer],
                num_heads=num_heads[i_layer],
                n_group=n_groups[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:(len(depths) - i_layer - 1)]):sum(depths[:(len(depths) - i_layer)])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, skips):
        outs = []
        H, W = x.size(2), x.size(3)
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]
            x, H, W, = layer(x, skips[i], H, W)
            outs.append(x)
        return outs


class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.num_block = int(np.log2(patch_size[0])) - 2
        self.project_block = []
        self.dim_list = [int(dim) // (2 ** i) for i in range(self.num_block + 1)]
        # dim, dim/2, dim/4
        for i in range(self.num_block):
            self.project_block.append(project_up(self.dim_list[i], self.dim_list[i + 1], nn.GELU, nn.LayerNorm, False))
        self.project_block = nn.ModuleList(self.project_block)
        self.up_final = nn.ConvTranspose2d(self.dim_list[-1], num_class, 4, 4)

    def forward(self, x):
        for blk in self.project_block:
            x = blk(x)
        x = self.up_final(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, type, dim, drop_path=0., layer_scale_init_value=1e-6, input_resolution=None, num_heads=None, n_group=None,
                 window_size=None, i_block=None, qkv_bias=None, qk_scale=None, mlp_ratio=2., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if type == 'k':
            self.blocks_tr = SwinDATransformerBlock(  # DA trans
                dim=dim,
                input_resolution=input_resolution,
                n_head=num_heads,
                n_group=n_group,
                window_size=window_size,
                shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                drop=0.,
                attn_drop=0.,
                drop_path=drop_path)

        elif type == 'd':
            self.blocks_tr = DATransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                n_head=num_heads,
                n_group=n_group,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=0.,
                attn_drop=0.,
                drop_path=drop_path,
                norm_layer=norm_layer,
            )

        elif type == 's':
            self.blocks_tr = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                n_head=num_heads,
                window_size=window_size,
                shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.,
                attn_drop=0.,
                drop_path=drop_path,
                norm_layer=norm_layer
            )

        else:
            raise "Wrong Block types for Bottleneck!"

    def forward(self, x, mask):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        N, C, H, W = x.shape
        x = input + self.drop_path(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(N, H*W, C)
        x_size = [H, W]
        x, position, reference, _ = self.blocks_tr(x, mask, x_size)
        # print("position:", position)
        # print("reference", reference)
        x = x.reshape(N, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x, 0, 0
        # return x, position, reference


class unet2022(SegmentationNetwork):
    def __init__(self,
                 num_input_channels,
                 embedding_dim,
                 types,
                 num_heads,
                 n_groups,
                 num_classes,
                 depths,
                 crop_size,
                 convolution_stem_down,
                 window_size,
                 deep_supervision=False,
                 conv_op=nn.Conv2d,
                 ):
        super(unet2022, self).__init__()

        # Don't uncomment conv_op
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.do_ds = deep_supervision
        self.embed_dim = embedding_dim
        self.depths = depths
        self.num_heads = num_heads
        self.crop_size = crop_size
        self.patch_size = [convolution_stem_down, convolution_stem_down]
        self.window_size = window_size
        self.n_groups = n_groups
        self.types = types
        # if window size of the encoder is [7,7,14,7], then decoder's is [14,7,7]. In short, reverse the list and start from the index of 1
        self.model_down = encoder(
            pretrain_img_size=self.crop_size,
            window_size=self.window_size,
            embed_dim=self.embed_dim,
            types=self.types,
            patch_size=self.patch_size,
            depths=self.depths,
            num_heads=self.num_heads,
            n_groups=self.n_groups,
            in_chans=self.num_input_channels
        )

        self.decoder = decoder(
            pretrain_img_size=self.crop_size,
            window_size=self.window_size[::-1][1:],
            embed_dim=self.embed_dim,
            types=self.types[::-1][1:],
            patch_size=self.patch_size,
            depths=self.depths[::-1][1:],
            num_heads=self.num_heads[::-1][1:],
            n_groups=self.n_groups[::-1][1:]
        )

        self.final = []
        for i in range(len(self.depths) - 1):
            self.final.append(
                final_patch_expanding(self.embed_dim * 2 ** i, self.num_classes, patch_size=self.patch_size))
        self.final = nn.ModuleList(self.final)

    def forward(self, x):
        seg_outputs = []
        skips = self.model_down(x)
        neck = skips[-1]
        out = self.decoder(neck, skips)

        for i in range(len(out)):
            seg_outputs.append(self.final[-(i + 1)](out[i]))
        if self.do_ds:
            # for training
            return seg_outputs[::-1]
        else:
            # for validation and testing
            return seg_outputs[-1]
            # size [[224,224]]

    def load_pretrained_weights(self, network, fname, verbose=False):
        """
        THIS DOES NOT TRANSFER SEGMENTATION HEADS!
        """
        saved_model = torch.load(fname)
        pretrained_dict = saved_model['state_dict']

        new_state_dict = {}

        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in pretrained_dict.items():
            key = k
            # remove module. prefix from DDP models
            if key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        pretrained_dict = new_state_dict

        model_dict = network.state_dict()
        ok = True
        for key, _ in model_dict.items():
            if ('conv_blocks' in key):
                if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                    continue
                else:
                    ok = False
                    break

        # filter unnecessary keys
        if ok:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            print("################### Loading pretrained weights from file ", fname, '###################')
            if verbose:
                print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
                for key, _ in pretrained_dict.items():
                    print(key)
            print("################### Done ###################")
            network.load_state_dict(model_dict)
        else:
            raise RuntimeError("Pretrained weights are not compatible with the current network architecture")



if __name__ == '__main__':

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = 'cuda'

    model = unet2022(num_input_channels=1,
                     embedding_dim=192,
                     types=['k', 'k', 'k', 'k'],
                     num_heads=[3, 6, 12, 24],
                     n_groups=[1, 3, 6, 6],
                     num_classes=4,
                     depths=[3, 3, 3, 3],
                     crop_size=[224, 224],
                     convolution_stem_down=4,
                     window_size=[7, 7, 14, 7],
                     deep_supervision=False,
                     conv_op=nn.Conv2d).to(device)
    pretrained_model_path = "/home/lichao/UNet-2022/DATASET/nnunet_trained_models/nnUNet/2d/Task002_Synapse/nnUNetTrainerV2_unet2022_synapse_224__nnUNetPlansv2.1/fold_0/model_best.model"
    model.load_pretrained_weights(model, pretrained_model_path, True)
    x = torch.randn((1, 1, 224, 224)).to(device)

    print(f'Input shape: {x.shape}')
    # with torch.no_grad():
    x = model(x)
    print(f'Output shape: {x.shape}')
    print('-------------------------------')



