import os
import cv2
import torch
import scipy
import skimage
import numpy as np
import torch.nn.functional as F

def convert_filepath(filepath, output_type, output_ext):
    output = filepath.replace('/left/', f'/{output_type}/')
    base, _ = os.path.splitext(output)
    output = f'{base}.{output_ext}'

    os.makedirs(os.path.dirname(output), exist_ok=True)

    return output

def generate_disparity(normalized_inverse_depth, a=0.1, b=0.05):
    _, _, h, w = normalized_inverse_depth.shape
    max_disparity_range = ((a - b) * w, (a + b) * w)

    scale = np.random.rand()
    if scale < 0.1:
        max_disarpity_range = (min(a - 2 * b, 0.01) * w, (a - b) * w)
    elif scale > 0.9:
        max_disarpity_range = ((a + b) * w, (a + 2 * b) * w)

    scaling_factor = (max_disparity_range[0] + np.random.rand() * (max_disparity_range[1] - max_disparity_range[0]))
    disparity = normalized_inverse_depth * scaling_factor

    return disparity

def get_non_occlusion_mask(shifted):
    h, w = shifted.shape
    mask_up = shifted > 0
    mask_down = shifted > 0

    shifted_up = np.ceil(shifted)
    shifted_down = np.floor(shifted)

    for col in range(w - 2):
        loc = shifted[:, col:col + 1]
        loc_up = np.ceil(loc)
        loc_down = np.floor(loc)

        _mask_down = ((shifted_down[:, col + 2:] != loc_down) * ((shifted_up[:, col + 2:] != loc_down))).min(-1)
        _mask_up = ((shifted_down[:, col + 2:] != loc_up) * ((shifted_up[:, col + 2:] != loc_up))).min(-1)

        mask_up[:, col] = mask_up[:, col] * _mask_up
        mask_down[:, col] = mask_down[:, col] * _mask_down

    mask = mask_up + mask_down

    return mask

def warp_image(image, disp):
    h, w, c = image.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disp = cv2.dilate(disp, kernel, iterations=1)
    
    edge = skimage.filters.sobel(disp) > 3
    disp[edge] = 0
    disp = scipy.interpolate.griddata(np.stack([ys[~edge].ravel(), xs[~edge].ravel()], 1), disp[~edge].ravel(), np.stack([ys.ravel(), xs.ravel()], 1), method='nearest').reshape(h, w)
    '''
    warped_image = np.zeros_like(image)
    warped_image = np.stack([warped_image] * 2, 0)

    pix_locations = xs - disp
    mask_nocc = get_non_occlusion_mask(pix_locations)
    masked_pix_locations = np.zeros_like(pix_locations) - w
    masked_pix_locations[mask_nocc] = pix_locations[mask_nocc]

    weights = np.ones((2, h, w)) * 10000
    for col in range(w - 1, -1, -1):
        loc = masked_pix_locations[:, col]
        loc_up = np.ceil(loc).astype(np.int32)
        loc_down = np.floor(loc).astype(np.int32)
        weight_up = loc_up - loc
        weight_down = 1 - weight_up

        mask = loc_up >= 0
        mask[mask] = weights[0, np.arange(h)[mask], loc_up[mask]] > weight_up[mask]
        weights[0, np.arange(h)[mask], loc_up[mask]] = weight_up[mask]
        warped_image[0, np.arange(h)[mask], loc_up[mask]] = image[:, col][mask]
        mask = loc_down >= 0
        mask[mask] = weights[1, np.arange(h)[mask], loc_down[mask]] > weight_down[mask]
        weights[1, np.arange(h)[mask], loc_down[mask]] = weight_down[mask]
        warped_image[1, np.arange(h)[mask], loc_down[mask]] = image[:, col][mask]
    
    mask_inpaint = np.all(weights == 10000, axis=0)
    weights /= weights.sum(0, keepdims=True) + 1e-7
    weights = np.expand_dims(weights, -1)
    warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
    warped_image[mask_inpaint] = 0
    
    return warped_image, mask_nocc, mask_inpaint

def disp_warp(x, disp, pad='border', mode='bilinear'):
    """
    image : [B, _, H, W]
    disp : [B, 1, H, W]
    """
    B, _, H, W = x.shape
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float()

    # vgrid = Variable(grid)
    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    warped_image = F.grid_sample(x, vgrid, mode=mode, padding_mode=pad, align_corners=True)

    return warped_image