import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim
import torchvision.transforms as T
# 复用你模型里的 HVI 转换
from model import RGB_HVI
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16]
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, y_true, y_pred):
        y_true = self.normalize(y_true)
        y_pred = self.normalize(y_pred)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))


def color_loss(y_true, y_pred):
    return torch.mean(torch.abs(torch.mean(y_true, dim=[1, 2, 3]) - torch.mean(y_pred, dim=[1, 2, 3])))

def psnr_loss(y_true, y_pred, eps=1e-8):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    return -psnr / 20.0  # 归一化，避免梯度爆炸

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)

    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=0)
    
    y_true_hist /= (y_true_hist.sum() + 1e-8)
    y_pred_hist /= (y_pred_hist.sum() + 1e-8)

    hist_distance = torch.mean(torch.abs(y_true_hist - y_pred_hist))
    return hist_distance
#  定义 HVI 空间的损失函数

def hvi_loss(y_true_hvi, y_pred_hvi, weights=(0.4, 0.4, 0.2), eps=1e-8):
    # y_*_hvi: [B,3,H,W]，通道顺序 [H, V, I]
    Ht, Vt, It = y_true_hvi[:, 0], y_true_hvi[:, 1], y_true_hvi[:, 2].clamp(0,1)
    Hp, Vp, Ip = y_pred_hvi[:, 0], y_pred_hvi[:, 1], y_pred_hvi[:, 2].clamp(0,1)

    # 色相角度一致：用余弦相似度
    mag_t = torch.sqrt(Ht*Ht + Vt*Vt + eps)
    mag_p = torch.sqrt(Hp*Hp + Vp*Vp + eps)
    cos_sim = (Ht*Hp + Vt*Vp) / (mag_t*mag_p + eps)
    angle_loss  = (1.0 - cos_sim).mean()  #色相角

    # 饱和度半径一致
    radius_loss = F.l1_loss(mag_p, mag_t)

    # 亮度一致（I ∈ [0,1]）
    intensity_loss = F.l1_loss(Ip, It)

    w1, w2, w3 = weights
    return w1*angle_loss + w2*radius_loss + w3*intensity_loss

# 改进的频域感知损失 - 增加频域权重
class FrequencyPerceptualLoss(nn.Module):
    def __init__(self, low_freq_weight=0.3, high_freq_weight=0.7):
        super(FrequencyPerceptualLoss, self).__init__()
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight

    def forward(self, y_true, y_pred):
        # 转换到频域
        fft_true = torch.fft.fft2(y_true, norm='ortho')
        fft_pred = torch.fft.fft2(y_pred, norm='ortho')
        
        # 频域移位
        fft_true_shifted = torch.fft.fftshift(fft_true)
        fft_pred_shifted = torch.fft.fftshift(fft_pred)

        # 计算频域幅度
        mag_true = torch.sqrt(fft_true_shifted.real**2 + fft_true_shifted.imag**2+1e-8)
        mag_pred = torch.sqrt(fft_pred_shifted.real**2 + fft_pred_shifted.imag**2+1e-8)
        
        # 相位
        phase_true = torch.atan2(fft_true_shifted.imag, fft_true_shifted.real)
        phase_pred = torch.atan2(fft_pred_shifted.imag, fft_pred_shifted.real)

        # 分离低频和高频区域
        B, C, H, W = y_true.shape
        center_h, center_w = H // 2, W // 2
        radius = min(H, W) // 4  # 低频区域半径
        
        mask_low = torch.zeros_like(mag_true)
        mask_high = torch.ones_like(mag_true)
        
        mask_low[:, :, center_h-radius:center_h+radius, center_w-radius:center_w+radius] = 1
        mask_high -= mask_low

        # 分别计算低频和高频损失
        mag_loss_low = F.mse_loss(mag_true * mask_low, mag_pred * mask_low)
        mag_loss_high = F.mse_loss(mag_true * mask_high, mag_pred * mask_high)
        phase_loss = F.mse_loss(phase_true, phase_pred)

        # 加权组合
        total_loss = (self.low_freq_weight * mag_loss_low + 
                      self.high_freq_weight * mag_loss_high + 
                      0.1 * phase_loss)  # 相位损失权重较低

        return total_loss

# 色度降噪损失 - 针对LYT模型的色度降噪模块优化
def chroma_denoise_loss(h_true, v_true, h_pred, v_pred):
    # 色度分量的MSE损失
    h_loss = F.mse_loss(h_pred, h_true)
    v_loss = F.mse_loss(v_pred, v_true)
    
    # 色度直方图损失
    hist_h_loss = histogram_loss(h_true, h_pred)
    hist_v_loss = histogram_loss(v_true, v_pred)
    
    # 组合损失
    return (h_loss + v_loss) + 0.2 * (hist_h_loss + hist_v_loss)

def rgb_to_I(rgb, rgb2hvi: RGB_HVI):
    hvi = rgb2hvi.HVIT(rgb.clamp(0,1))   # 不要 no_grad
    return hvi[:, 2:3, :, :]  # I 通道 in [0,1]

def charbonnier(x, y, eps=1e-6):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps))
class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss_model = VGGPerceptualLoss(device)
        self.freq_loss_model = FrequencyPerceptualLoss()  # 初始化FrequencyPerceptualLoss 模型
        self.rgb2hvi = RGB_HVI().to(device).eval()
        for p in self.rgb2hvi.parameters():
            p.requires_grad = False
        # self.use_gan = True  # 是否使用GAN损失
        # GAN损失 (可选)
        # if self.use_gan:
        #     self.gan_loss = nn.BCEWithLogitsLoss()
        # （下方 GAN 分支里我会略微下调感知/频域，给 RGB SSIM/MSE 腾点权重）
        self.w_hvi   = 0.10
        self.w_freq  = 0.05
        self.w_perc  = 0.05
        self.w_chrom = 0.08

        # 新增的两个权重（建议起步值，够小，不会“改变味道”，但能抬分）
        self.w_rgb_ssim_warm = 0.25   # 预热：RGB MS-SSIM
        self.w_rgb_mse_warm  = 0.35   # 预热：RGB MSE(=PSNR 代理)
        self.w_rgb_ssim_gan  = 0.30   # GAN：RGB MS-SSIM
        self.w_rgb_mse_gan   = 0.30   # GAN：RGB MSE
        
        

    def forward(self, y_true, y_pred, h_true=None, v_true=None, h_pred=None, v_pred=None,isgan=False):
        y_true_c = y_true.clamp(0, 1)
        y_pred_c = y_pred.clamp(0, 1)
        # ====== 新增：直接在 RGB 上算 MS-SSIM 与 MSE（两行）======
        ms_rgb  = 1.0 - ms_ssim(y_pred_c, y_true_c, data_range=1.0, size_average=True)
        mse_rgb = F.mse_loss(y_pred_c, y_true_c)
        # ---------- 预热阶段： ----------
        if not isgan:
            I_true = rgb_to_I(y_true_c, self.rgb2hvi)
            I_pred = rgb_to_I(y_pred_c, self.rgb2hvi)
            ms_I = 1.0 - ms_ssim(I_pred, I_true, data_range=1.0, size_average=True)
            l1_I = charbonnier(I_pred, I_true)

            

            return 0.84 * ms_I + 0.16 * l1_I + self.w_rgb_ssim_warm * ms_rgb + self.w_rgb_mse_warm * mse_rgb
        # ------- GAN 阶段 -------
        with torch.no_grad():
            I_true = rgb_to_I(y_true_c, self.rgb2hvi)   # target 可 no_grad
        I_pred = rgb_to_I(y_pred_c, self.rgb2hvi)
        ms_I = 1.0 - ms_ssim(I_pred, I_true, data_range=1.0, size_average=True)
        l1_I = F.l1_loss(I_pred, I_true)
        recon_I = 0.60 * ms_I + 0.15 * l1_I

        with torch.no_grad():
            y_true_hvi = self.rgb2hvi.HVIT(y_true_c)
        y_pred_hvi = self.rgb2hvi.HVIT(y_pred_c)
        hvi_l = hvi_loss(y_true_hvi, y_pred_hvi)

        perc_l = self.perceptual_loss_model(y_true_c, y_pred_c)
        freq_l = self.freq_loss_model(y_true_c, y_pred_c)

        if h_true is not None and v_true is not None:
            chroma_l = chroma_denoise_loss(h_true, v_true, h_pred, v_pred)
        else:
            chroma_l = 0.0

        return (
            1.00 * recon_I +
            self.w_hvi  * hvi_l   +
            (self.w_freq * 0.6) * freq_l +   # 小幅下调频域/感知，给 RGB 两项让路（可不改）
            (self.w_perc * 0.6) * perc_l +
            self.w_chrom * chroma_l +
            self.w_rgb_ssim_gan * ms_rgb +
            self.w_rgb_mse_gan  * mse_rgb
        )
        
        # # 统一假设输入在 [0,1] y_true整张图片
        # y_true_c = torch.clamp(y_true, 0.0, 1.0)
        # y_pred_c = torch.clamp(y_pred, 0.0, 1.0)

        # smooth_l1_l = smooth_l1_loss(y_true_c, y_pred_c)
        # ms_ssim_l = multiscale_ssim_loss(y_true_c, y_pred_c)
        # perc_l = self.perceptual_loss_model(y_true_c, y_pred_c)
        # hist_l = histogram_loss(y_true_c, y_pred_c)
        # psnr_l = psnr_loss(y_true_c, y_pred_c)
        # color_l = color_loss(y_true_c, y_pred_c)
        

        # with torch.no_grad():
        #     y_true_hvi = self.rgb2hvi.HVIT(y_true_c)

        # y_pred_hvi = self.rgb2hvi.HVIT(y_pred_c)
        # hvi_l      = hvi_loss(y_true_hvi, y_pred_hvi)
        # freq_l     = self.freq_loss_model(y_true_c, y_pred_c)

        # # 色度降噪损失 (如果提供了色度分量)
        # # 色度降噪可选
        # if h_true is not None and v_true is not None:
        #     chroma_l = chroma_denoise_loss(h_true, v_true, h_pred, v_pred)
        # else:
        #     chroma_l = 0.0
        # if isgan == True:
        #     total_loss = (self.alpha1 * smooth_l1_l + self.alpha2 * perc_l + 
        #                 self.alpha3 * hist_l  + self.alpha5 * psnr_l +
        #                 self.alpha6 * color_l + self.alpha4 * ms_ssim_l +
        #                 self.alpha7 * hvi_l + self.alpha8 * freq_l +
        #                 self.alpha9 * chroma_l)
        # else:
        #     total_loss = (self.alpha1 * smooth_l1_l + 
        #                  self.alpha4 * ms_ssim_l 
        #                 )
        # # total_loss = torch.clamp(total_loss, min=0.0, max=1e5)
        # return total_loss
