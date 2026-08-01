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
