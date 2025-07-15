import hydra
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from hydra.utils import instantiate
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from model import fetch_model
from dataset import fetch_dataloader
from util.util import convert_filepath
from util.padder import InputPadder

def inference(cfg, accelerator, model, warped_image, mask_nocc, mask_inpaint, scale, h, w):
    warped_image = warped_image / 127.5 - 1
    masked_image = warped_image * (mask_inpaint < 0.5)

    padder = InputPadder(masked_image.shape, divis_by=8)
    masked_image, mask_inpaint = padder.pad(masked_image, mask_inpaint)

    with accelerator.autocast():
        image_pred = model.single_infer(masked_image, mask_inpaint, cfg.num_inference_step)

        image_pred = padder.unpad(image_pred)
        mask_inpaint = padder.unpad(mask_inpaint)
        image_pred = image_pred * (mask_inpaint >= 0.5) + warped_image * (mask_inpaint < 0.5)

    if scale > 1:
        image_pred = F.interpolate(image_pred, size=(h, w), mode='bicubic', align_corners=True).clip(-1, 1)
        mask_nocc = F.interpolate(mask_nocc, size=(h, w), mode='nearest')
        mask_inpaint = F.interpolate(mask_inpaint, size=(h, w), mode='nearest')

    return image_pred, mask_nocc, mask_inpaint

@hydra.main(version_base=None, config_path='config', config_name='generate_stereo')
def main(cfg):
    logger = get_logger(__name__)
    accelerator = instantiate(cfg.accelerator)

    dataset, dataloader = fetch_dataloader(cfg, cfg.dataset, cfg.dataloader, logger, True)
    model = fetch_model(cfg, logger)
    
    dataloader = accelerator.prepare(dataloader)
    model.to(accelerator.device, torch.float16 if cfg.accelerator.mixed_precision == 'fp16' else torch.float32)
    model.unet.eval()

    set_seed(cfg.seed, device_specific=True)
    for index, data in enumerate(tqdm(dataloader, dynamic_ncols=True, disable=not accelerator.is_main_process)):
        left_file, warped_image, mask_nocc, mask_inpaint, scale, h, w = data
        right_file = convert_filepath(left_file[0], 'right', 'png')
        mask_nocc_file = convert_filepath(left_file[0], 'mask_nocc', 'png')
        mask_inpaint_file = convert_filepath(left_file[0], 'mask_inpaint', 'png')

        while True:
            try:
                image_pred, mask_nocc, mask_inpaint = inference(cfg, accelerator, model, warped_image, mask_nocc, mask_inpaint, scale, h, w)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                scale += 1
                left_file, warped_image, mask_nocc, mask_inpaint, scale, h, w = dataset._getitem(index, scale)
            
        image_pred = ((image_pred + 1.) * 127.5).squeeze().round().permute(1, 2, 0).cpu().numpy().astype('uint8')
        mask_nocc = (mask_nocc * 255.).squeeze().round().cpu().numpy().astype('uint8')
        mask_inpaint = (mask_inpaint * 255.).squeeze().round().cpu().numpy().astype('uint8')

        image_pred = Image.fromarray(image_pred)
        mask_nocc = Image.fromarray(mask_nocc)
        mask_inpaint = Image.fromarray(mask_inpaint)

        image_pred.save(right_file, lossless=True)
        mask_nocc.save(mask_nocc_file, lossless=True)
        mask_inpaint.save(mask_inpaint_file, lossless=True)

        torch.cuda.empty_cache()

    accelerator.end_training()

if __name__ == '__main__':
    main()