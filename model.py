import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from math import pi
# 将 RGB_HVI 类集成到 model.py
pi = 3.141592653589793



class RGB_HVI(nn.Module):
    def __init__(self):
        super().__init__()
        # k > 0：建议在 [0.1, 5] 之间
        self.density_k = nn.Parameter(torch.tensor([0.8], dtype=torch.float32))
        # αS、αI 作用于 PHVIT 的 S、V（论文一致）
        self.alpha_S = 1.0
        self.alpha_I = 1.0
        # 仅在调试时保留
        self.enable_radius_clip = True   # 论文补充材料建议的半径裁剪
        self.eps = 1e-8

    @staticmethod
    def _rgb_to_hsv(rgb, eps=1e-8):
        # rgb in [0,1], shape [B,3,H,W]
        r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]
        maxc, _ = rgb.max(dim=1)
        minc, _ = rgb.min(dim=1)
        v = maxc
        delta = maxc - minc

        # Saturation
        s = torch.where(v > eps, delta / (v + eps), torch.zeros_like(v))

        # Hue（用 argmax 更稳）
        # 0:R 1:G 2:B
        max_idx = rgb.argmax(dim=1)
        h = torch.zeros_like(v)

        # 分段公式（H ∈ [0,6)）
        # 注意：只在 delta>0 时有定义
        mask = (delta > eps)

        # R max
        m = mask & (max_idx == 0)
        h[m] = ( (g[m] - b[m]) / (delta[m] + eps) ) % 6.0
        # G max
        m = mask & (max_idx == 1)
        h[m] = 2.0 + ( (b[m] - r[m]) / (delta[m] + eps) )
        # B max
        m = mask & (max_idx == 2)
        h[m] = 4.0 + ( (r[m] - g[m]) / (delta[m] + eps) )

        # 归一化到 [0,1)
        h = (h / 6.0) % 1.0

        return h, s, v

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        # h ∈ [0,1), s,v ∈ [0,1]
        hi = torch.floor(h * 6.0)
        f  = h * 6.0 - hi
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi0 = (hi == 0)
        hi1 = (hi == 1)
        hi2 = (hi == 2)
        hi3 = (hi == 3)
        hi4 = (hi == 4)
        hi5 = (hi == 5)

        r[hi0] = v[hi0]; g[hi0] = t[hi0]; b[hi0] = p[hi0]
        r[hi1] = q[hi1]; g[hi1] = v[hi1]; b[hi1] = p[hi1]
        r[hi2] = p[hi2]; g[hi2] = v[hi2]; b[hi2] = t[hi2]
        r[hi3] = p[hi3]; g[hi3] = q[hi3]; b[hi3] = v[hi3]
        r[hi4] = t[hi4]; g[hi4] = p[hi4]; b[hi4] = v[hi4]
        r[hi5] = v[hi5]; g[hi5] = p[hi5]; b[hi5] = q[hi5]

        return torch.stack([r, g, b], dim=1)

    def HVIT(self, img):
        """
        img: [B,3,H,W], 期望在 [0,1]
        return: [B,3,H,W]，通道为 [H, V, I]
        """
        eps = self.eps
        in_dtype = img.dtype
        img = img.float().clamp(0,1)

        h, s, v = self._rgb_to_hsv(img, eps=eps)  # H∈[0,1), S,V∈[0,1]
        h = h.unsqueeze(1); s = s.unsqueeze(1); v = v.unsqueeze(1)

        # Ck(I) = (sin(π I/2)+eps)^k
        k = torch.clamp(self.density_k, 0.1, 5.0).view(1,1,1,1)  # broadcast
        angle = v * (0.5 * pi)
        Ck = (torch.sin(angle) + eps).pow(k)

        # 极化（等价写法：cos(2π h) / sin(2π h)）
        ch = torch.cos(2.0 * pi * h)
        cv = torch.sin(2.0 * pi * h)

        H = Ck * s * ch
        V = Ck * s * cv
        I = v
        xyz = torch.cat([H, V, I], dim=1)

        # 记录 k 以便 PHVIT 使用（可逆性）
        self.this_k = k.squeeze().item()
        return xyz.to(in_dtype)

    def PHVIT(self, img_hvi):
        """
        img_hvi: [B,3,H,W]，通道为 [H, V, I]
        return: [B,3,H,W] RGB in [0,1]
        """
        eps = self.eps
        in_dtype = img_hvi.dtype

        H = img_hvi[:,0].float()
        V = img_hvi[:,1].float()
        I = img_hvi[:,2].float().clamp(0,1)

        # 重建同一个 Ck(I)
        if getattr(self, "this_k", None) is not None:
            k = float(self.this_k)
        else:
            k = float(torch.clamp(self.density_k.detach(), 0.1, 5.0))
        
        Ck = (torch.sin(I * 0.5 * pi) + eps).pow(k)

        # 逆半径约束（可选但推荐）
        if self.enable_radius_clip:
            radius2 = H**2 + V**2
            max_r = (Ck**2)
            scale = torch.clamp_max(torch.sqrt(max_r / (radius2 + eps)), 1.0)
            H = H * scale
            V = V * scale

        # 去极化，得到 HSV
        h_hat = H / (Ck + eps)
        v_hat = V / (Ck + eps)

        # Hue ∈ [0,1)
        h = torch.atan2(v_hat, h_hat) / (2 * pi)
        h = h % 1.0

        # Saturation（附加 αS）
        s = torch.sqrt(h_hat**2 + v_hat**2 + eps)
        s = torch.clamp(self.alpha_S * s, 0.0, 1.0)

        # Value（附加 αI）
        v = torch.clamp(self.alpha_I * I, 0.0, 1.0)

        # HSV → RGB
        rgb = self._hsv_to_rgb(h, s, v).clamp(0,1)
        return rgb.to(in_dtype)

class DRBNBlock_FreqEnhanced(nn.Module):
    """
    FRB with ablation modes for Reviewer Comment #1.

    mode:
        full:
            Proposed FRB. Low/high frequency features are independently transformed
            before fusion.

        old_collapse:
            Implements the criticized formulation:
            Concat(x, 0.5 * (x_low + x_high)).
            Since x_low + x_high = x, this is nearly redundant.

        no_transform:
            Uses low/high frequency features directly without independent
            transformations. This tests whether low_branch/high_branch are necessary.

        low_only:
            Uses only the transformed low-frequency component.

        high_only:
            Uses only the transformed high-frequency component.

        spatial_matched:
            Replaces FFT decomposition with two spatial convolution branches.
            This controls whether the gain comes from frequency modeling or
            simply from extra convolutional capacity.

        identity:
            Removes FRB. This is w/o FRB.
    """

    def __init__(self, channels, init_mask_ratio=0.1, mode="full", export_debug=False):
        super(DRBNBlock_FreqEnhanced, self).__init__()

        valid_modes = [
            "full",
            "old_collapse",
            "no_transform",
            "low_only",
            "high_only",
            "spatial_matched",
            "identity",
        ]
        assert mode in valid_modes, f"Unknown FRB mode: {mode}"

        self.channels = channels
        self.freq_mask_ratio = float(init_mask_ratio)
        self.mode = mode
        self.export_debug = export_debug
        self.debug_maps = {}

        # Independent transformation for low-frequency component
        self.low_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        # Independent transformation for high-frequency component
        self.high_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        # For full / low_only / high_only / no_transform / spatial_matched
        self.conv1_3 = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)

        # For old collapsed formulation: Concat(x, 0.5 * (x_lf + x_hf))
        self.conv1_2 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def _fft_decompose(self, x):
        """
        Decompose x into low-frequency and high-frequency components.
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        fft = torch.fft.fft2(x, norm="ortho")
        fft_shift = torch.fft.fftshift(fft)

        radius = max(1, int(self.freq_mask_ratio * min(H, W) / 2))
        center_h, center_w = H // 2, W // 2

        low_mask = torch.zeros(
            (1, 1, H, W),
            device=x.device,
            dtype=x.dtype,
        )

        low_mask[
            :,
            :,
            center_h - radius:center_h + radius,
            center_w - radius:center_w + radius,
        ] = 1.0

        high_mask = 1.0 - low_mask

        low_freq = fft_shift * low_mask
        high_freq = fft_shift * high_mask

        low_feat = torch.fft.ifft2(
            torch.fft.ifftshift(low_freq),
            norm="ortho",
        ).real

        high_feat = torch.fft.ifft2(
            torch.fft.ifftshift(high_freq),
            norm="ortho",
        ).real

        return low_feat, high_feat

    def _save_debug_maps(self, x, low_feat=None, high_feat=None,
                         low_trans=None, high_trans=None, fusion_input=None, residual=None, output=None):
        if not self.export_debug:
            return

        maps = {"input": x.detach().float().mean(dim=1, keepdim=True).cpu()}

        if low_feat is not None:
            maps["low_feat"] = low_feat.detach().float().mean(dim=1, keepdim=True).cpu()

        if high_feat is not None:
            # high-frequency map can contain positive and negative responses;
            # abs() makes edges/textures more visible.
            maps["high_feat_abs"] = high_feat.detach().float().abs().mean(dim=1, keepdim=True).cpu()

        if low_trans is not None:
            maps["low_trans"] = low_trans.detach().float().mean(dim=1, keepdim=True).cpu()

        if high_trans is not None:
            maps["high_trans_abs"] = high_trans.detach().float().abs().mean(dim=1, keepdim=True).cpu()

        if fusion_input is not None:
            maps["fusion_input"] = fusion_input.detach().float().mean(dim=1, keepdim=True).cpu()

        if residual is not None:
            maps["residual_abs"] = residual.detach().float().mean(dim=1, keepdim=True).cpu()

        if output is not None:
            maps["frb_output"] = output.detach().float().mean(dim=1, keepdim=True).cpu()

        self.debug_maps = maps

    def forward(self, x):
        if self.mode == "identity":
            return x
        if self.mode == "spatial_matched":
            spatial_low = self.low_branch(x)
            spatial_high = self.high_branch(x)

            fusion_input = torch.cat([x, spatial_low, spatial_high], dim=1)

            res = self.relu(self.conv1_3(fusion_input))
            res = self.conv2(res)

            self._save_debug_maps(
                x=x,
                low_trans=spatial_low,
                high_trans=spatial_high,
                fusion_input=fusion_input,
            )

            return x + self.scale * res

        B, C, H, W = x.shape

        # FFT should be computed in float32 for numerical stability.
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            low_feat, high_feat = self._fft_decompose(x32)

        low_feat = low_feat.to(x.dtype)
        high_feat = high_feat.to(x.dtype)

        if self.mode == "full":
            # Proposed valid FRB:
            # low/high are independently transformed before fusion.
            low_trans = self.low_branch(low_feat)
            high_trans = self.high_branch(high_feat)

            fusion_input = torch.cat([x, low_trans, high_trans], dim=1)
            res = self.relu(self.conv1_3(fusion_input))
            res = self.conv2(res)

            out = x + self.scale * res

            self._save_debug_maps(
                x=x,
                low_feat=low_feat,
                high_feat=high_feat,
                low_trans=low_trans,
                high_trans=high_trans,
                fusion_input=fusion_input,
                residual=res,
                output=out,
            )

            return out


        elif self.mode == "old_collapse":
            # Criticized equation in the manuscript:
            # x_low + x_high = x, so this branch is nearly redundant.
            freq_aux = 0.5 * (low_feat + high_feat)
            fusion_input = torch.cat([x, freq_aux], dim=1)

            res = self.relu(self.conv1_2(fusion_input))
            res = self.conv2(res)

            self._save_debug_maps(
                x=x,
                low_feat=low_feat,
                high_feat=high_feat,
                fusion_input=fusion_input,
            )

            return x + self.scale * res

        elif self.mode == "no_transform":
            # Tests whether independent low/high transformations are necessary.
            fusion_input = torch.cat([x, low_feat, high_feat], dim=1)

            res = self.relu(self.conv1_3(fusion_input))
            res = self.conv2(res)

            self._save_debug_maps(
                x=x,
                low_feat=low_feat,
                high_feat=high_feat,
                fusion_input=fusion_input,
            )

            return x + self.scale * res

        elif self.mode == "low_only":
            # Only low-frequency transformed feature is used.
            low_trans = self.low_branch(low_feat)
            zero_high = torch.zeros_like(low_trans)

            fusion_input = torch.cat([x, low_trans, zero_high], dim=1)

            res = self.relu(self.conv1_3(fusion_input))
            res = self.conv2(res)

            self._save_debug_maps(
                x=x,
                low_feat=low_feat,
                high_feat=high_feat,
                low_trans=low_trans,
                fusion_input=fusion_input,
            )

            return x + self.scale * res

        elif self.mode == "high_only":
            # Only high-frequency transformed feature is used.
            zero_low = torch.zeros_like(high_feat)
            high_trans = self.high_branch(high_feat)

            fusion_input = torch.cat([x, zero_low, high_trans], dim=1)

            res = self.relu(self.conv1_3(fusion_input))
            res = self.conv2(res)

            self._save_debug_maps(
                x=x,
                low_feat=low_feat,
                high_feat=high_feat,
                high_trans=high_trans,
                fusion_input=fusion_input,
            )

            return x + self.scale * res



# 普通 DRBNBlock（空间残差）
class DRBNBlock(nn.Module):
    def __init__(self, channels):
        super(DRBNBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        return x + self.scale * res

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, pooling_r=4, pool_pad=0, norm_layer=nn.BatchNorm2d):
        super(SCConv, self).__init__()

        # Use a 1x1 Conv layer to adjust the number of channels in identity
        self.conv_identity = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False)
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r, padding=pool_pad),
            nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )

    def forward(self, x):
        identity = self.conv_identity(x)  # 使用 1x1 卷积来调整通道数，确保 identity 张量的通道数与 planes 一致
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)  [b, planes, h, w]
        out = self.k4(out)  # k4
        return out




class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)
#改进后通道注意力
class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=8):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # 局部池化保留小区域信号
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        # 检查输入张量的设备
        device = x.device
        # 确保模块参数与输入张量在同一设备上
        if next(self.parameters()).device != x.device:
            self.to(x.device)
        batch_size, num_channels, _, _ = x.size()
        # 局部池化（捕捉小区域有效信号）+ 全局池化（捕捉整体趋势）
        global_y = self.global_pool(x).reshape(batch_size, num_channels)
        local_y = self.local_pool(x).mean(dim=(2,3)).reshape(batch_size, num_channels)  # 局部池化后平均
        # 融合两种特征（这里改为相加而非拼接，保持通道数不变）
        y = global_y + local_y  # (B, C)
        y = F.silu(self.fc1(y))  # Swish激活更适合捕捉噪声与信号的非线性关系
        y = torch.sigmoid(self.fc2(y))  # 输出通道权重（0-1）
        # 限制权重下限（如不低于0.3），避免过度抑制弱信号通道
        y = torch.clamp(y, min=0.3)  # 弱信号通道至少保留30%权重
        y = y.reshape(batch_size, num_channels, 1, 1).to(device)
        return x * y
    
    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='sigmoid')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)
        
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)
        
        output = self.combine_heads(attention)
        
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)



# ----------------------------
# 改进的色度降噪模块：基于U-Net的生成器（单通道输入输出）
# 替换原有的Denoiser模块
# ----------------------------
class GatedFusionAttention(nn.Module):
    def __init__(self, skip_channels, decoder_channels, out_channels):
        super(GatedFusionAttention, self).__init__()
        in_channels = skip_channels + decoder_channels

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 新增映射层：将 skip 和 decoder_input 转换到 out_channels
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.decoder_proj = nn.Conv2d(decoder_channels, out_channels, kernel_size=1)

        # 输出再处理（可选）
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, skip, decoder_input):
        # 拼接后用于生成门控图
        fusion = torch.cat([skip, decoder_input], dim=1)  # [B, skip+decoder, H, W]
        gate = self.gate_conv(fusion)                     # [B, out, H, W]

        # 将 skip 和 decoder_input 统一映射为 out_channels
        skip_proj = self.skip_proj(skip)                  # [B, out, H, W]
        decoder_proj = self.decoder_proj(decoder_input)   # [B, out, H, W]

        # 加权融合
        fused = gate * skip_proj + (1 - gate) * decoder_proj

        return self.fusion_conv(fused)



class GRB(nn.Module):
    def __init__(self, channels):
        super(GRB, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x + self.gate(x) * self.conv(x)
class ChromaDenoiserSharedUNet(nn.Module):
    def __init__(self, ngf=16, frb_mode="full", freq_mask_ratio=0.1):
        super(ChromaDenoiserSharedUNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, ngf, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(ngf)
        )
        self.pool1 = nn.AvgPool2d(2)

        self.enc2 = nn.Sequential(
            DRBNBlock_FreqEnhanced(
                ngf,
                init_mask_ratio=freq_mask_ratio,
                mode=frb_mode
            ),
            nn.Conv2d(ngf, ngf * 2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.AvgPool2d(2)

        self.enc3 = nn.Sequential(
            DRBNBlock_FreqEnhanced(
                ngf * 2,
                init_mask_ratio=freq_mask_ratio,
                mode=frb_mode
            ),

            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.AvgPool2d(2)

        self.bottleneck = GRB(ngf * 4)

        # 解码器 + GFA + GRB
        self.up3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 2, stride=2)
        self.gfa3 = GatedFusionAttention(
            skip_channels=ngf * 4,      # skip通道数（enc3输出）
            decoder_channels=ngf * 2,   # decoder上采样输入通道（up3输出）
            out_channels=ngf * 2        # 输出通道
        )

        self.dec3 = GRB(ngf * 2)

        self.up2 = nn.ConvTranspose2d(ngf * 2, ngf, 2, stride=2)
        self.gfa2 = GatedFusionAttention(
            skip_channels=ngf * 2,          # skip通道数（enc2输出）
            decoder_channels=ngf,       # decoder上采样输入通道（up2输出）
            out_channels=ngf            # 输出通道
        )
        self.dec2 = GRB(ngf)

        self.up1 = nn.ConvTranspose2d(ngf, ngf, 2, stride=2)
        self.gfa1 = GatedFusionAttention(
            skip_channels=ngf,          # skip通道数（enc1输出）
            decoder_channels=ngf,       # decoder上采样输入通道（up1输出）
            out_channels=ngf            # 输出通道
        )
        self.dec1 = GRB(ngf)

        self.out = nn.Sequential(
            nn.Conv2d(ngf, 2, 3, padding=1),  # 输出两个通道：Cb + Cr
            nn.Tanh(),  # 输出[-1,1]
            
        )
        self._init_weights()

    def forward(self, x):  # 输入维度：[B, 2, H, W]
        e1 = self.enc1(x)                    # [B, ngf, H, W]
        e2 = self.enc2(self.pool1(e1))       # [B, ngf*2, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))       # [B, ngf*4, H/4, W/4]
        b = self.bottleneck(self.pool3(e3))  # [B, ngf*4, H/8, W/8]

        d3 = self.up3(b)
        d3 = self.gfa3(e3, d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.gfa2(e2, d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.gfa1(e1, d1)
        d1 = self.dec1(d1)

        return self.out(d1)  # [B, 2, H, W]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

# ----------------------------
# 判别器（辅助训练，可选）
# ----------------------------
#2) ChromaDiscriminator: 接受 4 通道（noisy_cbcr 2ch + candidate_cbcr 2ch）
class ChromaDiscriminator(nn.Module):
    """
    条件判别器（PatchGAN风格），用于色度通道（CB/CR）判别。
    输入：
        y: 条件输入（ground truth色度，或其他特征） [B, 2, H, W]
        x: 待判别输入（生成器输出色度） [B, 2, H, W]
    输出：
        判别值 [B, 1, H/16, W/16] (PatchGAN输出)
    """
    def __init__(self, in_channels=2, base_filters=32):
        super(ChromaDiscriminator, self).__init__()
        self.in_channels = in_channels

        # 接收条件和生成器输出 -> 拼接后通道数翻倍
        self.model = nn.Sequential(
            nn.Conv2d(in_channels*2, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_filters, base_filters*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_filters*4, base_filters*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_filters*8, 1, kernel_size=4, stride=1, padding=1)  # PatchGAN输出
        )

    def forward(self, y, x):
        """
        y: 条件输入 [B, 2, H, W]
        x: 待判别输入 [B, 2, H, W]
        """
        input_cat = torch.cat([y, x], dim=1)  # 通道拼接
        out = self.model(input_cat)
        return out

class LYT(nn.Module):
    def __init__(self, filters=32, frb_mode="full", cdm_frb_mode=None, freq_mask_ratio=0.1):
        super(LYT, self).__init__()
        if cdm_frb_mode is None:
            cdm_frb_mode = frb_mode
        self.hvi_converter = RGB_HVI()  # Initialize the HVI converter
        self.process_y = self._create_processing_layers(filters)
        self.process_cb = self._create_processing_layers(filters)
        self.process_cr = self._create_processing_layers(filters)
         # 使用 SCConv 替代部分卷积层
        # 使用 SCConv 在不同的阶段进行特征融合
        self.sc_conv_stage_1 = SCConv(inplanes=filters, planes=filters, pooling_r=4)
        self.sc_conv_stage_2 = SCConv(inplanes=filters, planes=filters, pooling_r=2)

        self.drbn = nn.Sequential(
            DRBNBlock_FreqEnhanced(
                filters,
                init_mask_ratio=freq_mask_ratio,
                mode=frb_mode
            ),  # 第一个加频域残差
            DRBNBlock(filters)                # 第二个用普通版本节省计算
        )# 新增亮度增强模块

        # 替换原有Denoiser为改进的2通道色度降噪器
        self.chroma_denoiser = ChromaDenoiserSharedUNet(
            ngf=filters // 2,
            frb_mode=cdm_frb_mode,
            freq_mask_ratio=freq_mask_ratio
        )   


        self.lum_pool = nn.MaxPool2d(8)
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)
        self.lum_up = nn.Upsample(scale_factor=8, mode='nearest')
        self.lum_conv = nn.Conv2d(filters, filters, kernel_size=1, padding=0)
        self.ref_conv = nn.Conv2d(filters * 2, filters, kernel_size=1, padding=0)
        self.msef = MSEFBlock(filters)
        self.recombine = nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1)
        self.final_adjustments = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self._init_weights()

    def _create_processing_layers(self, filters):
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    # def _rgb_to_ycbcr(self, image):
    #     r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    
    #     y = 0.299 * r + 0.587 * g + 0.114 * b
    #     u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
    #     v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
    #     yuv = torch.stack((y, u, v), dim=1)
    #     return yuv

    def forward(self, inputs):
        # ycbcr = self._rgb_to_ycbcr(inputs)
        # Convert RGB to HVI
        hvi = self.hvi_converter.HVIT(inputs)
        h, v, i = torch.split(hvi, 1, dim=1)
        # 替换：
        hv_cat = torch.cat([h, v], dim=1)           # [B, 2, H, W]
        hv_denoised = self.chroma_denoiser(hv_cat)  # [B, 2, H, W]
        h_denoised, v_denoised = torch.split(hv_denoised, 1, dim=1)
        h_denoised = h + h_denoised
        v_denoised = v + v_denoised

        # 亮度特征处理
        i_feat = self.process_y(i)
        #SCConv 在不同阶段进行融合
        i_processed_1 = self.sc_conv_stage_1(i_feat) + self.drbn(i_feat)
        i_processed_2 = self.sc_conv_stage_2(i_processed_1) + i_processed_1

        # 色度特征处理
        h_processed = self.process_cb(h_denoised)
        v_processed = self.process_cr(v_denoised)
        # 特征融合
        ref = torch.cat([h_processed, v_processed], dim=1)
        lum = i_processed_2
        # 全局亮度注意力
        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_1 = self.lum_up(lum_1)
        lum = lum + lum_1
        # 参考特征调整
        ref = self.ref_conv(ref)
        shortcut = ref
        ref = ref + 0.2 * self.lum_conv(lum)
        ref = self.msef(ref)
        ref = ref + shortcut
        # 最终融合与输出
        recombined = self.recombine(torch.cat([ref, lum], dim=1))
        raw_out = self.final_adjustments(recombined)
        # IMPORTANT:
        # PHVIT expects H,V in [-1,1] and I in [0,1].
        # Use tanh for H/V and sigmoid for I before PHVIT.
        h = torch.tanh(raw_out[:, 0:1, :, :])   # H -> [-1,1]
        v = torch.tanh(raw_out[:, 1:2, :, :])   # V -> [-1,1]
        i = torch.sigmoid(raw_out[:, 2:3, :, :])# I -> [0,1]

        hvi_out = torch.cat([h, v, i], dim=1)
        
        #hvi转换回sRGB空间
        rgb_out = self.hvi_converter.PHVIT(hvi_out)
        # rgb_out = torch.clamp(rgb_out, 0.0, 1.0)
        return rgb_out
    
    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)


