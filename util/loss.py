import torch
import torch.nn as nn
import torch.nn.functional as F
from util.util import disp_warp

def ssim(x, y, md=3):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    refl = nn.ReflectionPad2d(md)

    x = refl(x)
    y = refl(y)
    mu_x = F.avg_pool2d(x, patch_size, stride=1, padding=0)
    mu_y = F.avg_pool2d(y, patch_size, stride=1, padding=0)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = F.avg_pool2d(x * x, patch_size, stride=1, padding=0) - mu_x_sq
    sigma_y = F.avg_pool2d(y * y, patch_size, stride=1, padding=0) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, patch_size, stride=1, padding=0) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)

    return dist

def photometric_loss(image, warped_image, beta=0.85):
    loss = beta * ssim(image, warped_image).mean(1, True) + (1 - beta) * (image - warped_image).abs().mean(1, True)

    return loss

def zero_raft_loss(disp_preds, left_clean, right_clean, disp_gt, conf, mask_nocc, mask_inpaint, valid, gamma=0.9, mu=0.1, max_disp=700):
    zero_loss = 0.
    valid = (valid >= 0.5) & (disp_gt < max_disp)
    warped_mask_real = disp_warp(1 - mask_inpaint, disp_gt, pad='zeros', mode='nearest')

    n_predictions = len(disp_preds)
    for i in range(n_predictions):
        adjusted_gamma = gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_gamma ** (n_predictions - i - 1)
        
        disp_preds[i] = -disp_preds[i]
        i_d_loss = (disp_preds[i] - disp_gt).abs() * conf

        warped_right_clean = disp_warp(right_clean, disp_preds[i])
        i_p_loss = photometric_loss(left_clean, warped_right_clean) * (1 - conf)

        assert i_d_loss.shape == i_p_loss.shape == valid.shape
        zero_loss += i_weight * (i_d_loss[valid].mean() + mu * i_p_loss[mask_nocc.bool() & warped_mask_real.bool()].mean())

    epe = torch.abs(disp_preds[-1] - disp_gt)
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'train/EPE': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean()
    }

    return zero_loss, metrics

def zero_igev_loss(disp_init_pred, disp_preds, left_clean, right_clean, disp_gt, conf, mask_nocc, mask_inpaint, valid, gamma=0.9, mu=0.1, max_disp=700):
    zero_loss = 0.
    valid = (valid >= 0.5) & (disp_gt < max_disp)
    warped_mask_real = disp_warp(1 - mask_inpaint, disp_gt, pad='zeros', mode='nearest')

    init_d_loss = F.smooth_l1_loss(disp_init_pred, disp_gt, reduction='none') * conf
    warped_right_clean = disp_warp(right_clean, disp_init_pred)
    init_p_loss = photometric_loss(left_clean, warped_right_clean) * (1 - conf)
    zero_loss += (init_d_loss[valid.bool()].mean() + 0.1 * init_p_loss[mask_nocc.bool() & warped_mask_real.bool()].mean())

    n_predictions = len(disp_preds)
    for i in range(n_predictions):
        adjusted_gamma = gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_gamma ** (n_predictions - i - 1)

        i_d_loss = (disp_preds[i] - disp_gt).abs() * conf

        warped_right_clean = disp_warp(right_clean, disp_preds[i])
        i_p_loss = photometric_loss(left_clean, warped_right_clean) * (1 - conf)

        assert i_d_loss.shape == i_p_loss.shape == valid.shape
        zero_loss += i_weight * (i_d_loss[valid].mean() + mu * i_p_loss[mask_nocc.bool() & warped_mask_real.bool()].mean())

    epe = torch.abs(disp_preds[-1] - disp_gt)
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'train/EPE': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean()
    }

    return zero_loss, metrics

def raft_loss(disp_preds, disp_gt, valid, gamma=0.9, max_disp=700):
    loss = 0.
    valid = (valid >= 0.5) & (disp_gt < max_disp)

    n_predictions = len(disp_preds)
    for i in range(n_predictions):
        adjusted_gamma = gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_gamma ** (n_predictions - i - 1)
        
        disp_preds[i] = -disp_preds[i]
        i_loss = (disp_preds[i] - disp_gt).abs()

        assert i_loss.shape == valid.shape
        loss += i_weight * (i_loss[valid].mean())

    epe = torch.abs(disp_preds[-1] - disp_gt)
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'train/EPE': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean()
    }

    return loss, metrics