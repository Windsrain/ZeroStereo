import cv2
import hydra
import scipy
import torch
import skimage
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from hydra.utils import instantiate
from accelerate import load_checkpoint_and_dispatch
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from model import fetch_model
from dataset import fetch_dataloader
from util.util import convert_filepath, generate_disparity
from util.padder import InputPadder

def inference(model, image, scale, mean, std):
    h, w = image.shape[-2:]
    if scale > 1:
        image = F.interpolate(image, size=(h // scale, w // scale), mode='bicubic', align_corners=True).clip(0, 255)

    image = (image / 255. - mean) / std
    padder = InputPadder(image.shape, divis_by=14)
    image = padder.pad(image)[0]
    flipped_image = torch.flip(image, [-1])

    with torch.no_grad():
        inverse_depth = model.forward(image)
        torch.cuda.empty_cache()
        flipped_inverse_depth = torch.flip(model.forward(flipped_image), [-1])
        torch.cuda.empty_cache()        

    inverse_depth = padder.unpad(inverse_depth[:, None])
    flipped_inverse_depth = padder.unpad(flipped_inverse_depth[:, None])

    if scale > 1:
        inverse_depth = F.interpolate(inverse_depth, size=(h, w), mode='bilinear', align_corners=True)
        flipped_inverse_depth = F.interpolate(flipped_inverse_depth, size=(h, w), mode='bilinear', align_corners=True)        

    return inverse_depth, flipped_inverse_depth

@hydra.main(version_base=None, config_path='config', config_name='generate_mono')
def main(cfg):
    logger = get_logger(__name__)
    accelerator = instantiate(cfg.accelerator)

    dataloader = fetch_dataloader(cfg, cfg.dataset, cfg.dataloader, logger)
    model = fetch_model(cfg, logger)
    model = load_checkpoint_and_dispatch(model, cfg.checkpoint)
    model.eval()
    logger.info(f'Loading checkpoint from {cfg.checkpoint}.')

    dataloader, model = accelerator.prepare(dataloader, model)

    mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].to(accelerator.device)
    std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].to(accelerator.device)
    set_seed(cfg.seed, device_specific=True)
    for data in tqdm(dataloader, dynamic_ncols=True, disable=not accelerator.is_main_process):
        left_file, image = data
        conf_file = convert_filepath(left_file[0], 'confidence', 'npy')
        disp_file = convert_filepath(left_file[0], 'disparity', 'npy')

        scale = 1
        while True:
            try:
                inverse_depth, flipped_inverse_depth = inference(model, image, scale, mean, std)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                scale += 1

        normalized_inverse_depth = (inverse_depth - inverse_depth.min()) / (inverse_depth.max() - inverse_depth.min())
        flipped_normalized_inverse_depth = (flipped_inverse_depth - flipped_inverse_depth.min()) / (flipped_inverse_depth.max() - flipped_inverse_depth.min())
        confidence = torch.ones_like(normalized_inverse_depth) - (torch.abs(normalized_inverse_depth - flipped_normalized_inverse_depth))
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())

        disparity = generate_disparity(normalized_inverse_depth).squeeze().cpu().numpy()

        h, w = disparity.shape[-2:]
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        disparity = cv2.dilate(disparity, kernel, iterations=cfg.dilate_iteration)
        edge = skimage.filters.sobel(disparity) > 3
        disparity[edge] = 0
        disparity = scipy.interpolate.griddata(np.stack([ys[~edge].ravel(), xs[~edge].ravel()], 1), disparity[~edge].ravel(), np.stack([ys.ravel(), xs.ravel()], 1), method='nearest').reshape(h, w)

        np.save(conf_file, confidence.squeeze().cpu().numpy())
        np.save(disp_file, disparity)
        
        torch.cuda.empty_cache()

    accelerator.end_training()

if __name__ == '__main__':
    main()