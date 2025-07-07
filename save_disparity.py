import os
import cv2
import hydra
import torch
from glob import glob
from tqdm import tqdm
from hydra.utils import instantiate
from accelerate import load_checkpoint_and_dispatch
from accelerate.logging import get_logger
from matplotlib import pyplot as plt
from model import fetch_model
from util.padder import InputPadder

@hydra.main(version_base=None, config_path='config', config_name='save_disparity')
def main(cfg):
    logger = get_logger(__name__)
    accelerator = instantiate(cfg.accelerator)

    model = fetch_model(cfg, logger)
    model = load_checkpoint_and_dispatch(model, cfg.checkpoint)
    logger.info(f'Loading checkpoint from {cfg.checkpoint}.')  

    model = accelerator.prepare_model(model)
    model.eval()

    left_list = sorted(glob(cfg.left_list))
    right_list = sorted(glob(cfg.right_list))
    left_list = left_list[accelerator.process_index::accelerator.num_processes]
    right_list = right_list[accelerator.process_index::accelerator.num_processes]

    for left_file, right_file in tqdm(zip(left_list, right_list), total=len(left_list), dynamic_ncols=True, disable=not accelerator.is_main_process):
        left = cv2.imread(left_file)
        right = cv2.imread(right_file)
        left = torch.from_numpy(left).permute(2, 0, 1)[None].cuda()
        right = torch.from_numpy(right).permute(2, 0, 1)[None].cuda()

        padder = InputPadder(left.shape, divis_by=32)
        left_right = padder.pad(left, right)

        with torch.no_grad():
            if cfg.model.name == 'RAFTStereo':
                _, disp_pred = model(left, right, iters=cfg.model.valid_iters, test_mode=True)
                disp_pred = -disp_pred
            elif cfg.model.name == 'IGEVStereo':
                disp_pred = model(left, right, iters=cfg.model.valid_iters, test_mode=True)
            else:
                raise Exception(f'Invalid model name: {cfg.model.name}.')            

        disp_pred = padder.unpad(disp_pred)

        os.makedirs(cfg.disp_dir, exist_ok=True)
        plt.imsave(f'{cfg.disp_dir}/{left_file.split('/')[cfg.base_index].split('.')[0]}.png', disp_pred.squeeze().cpu().numpy(), cmap='magma')

    accelerator.end_training()

if __name__ == '__main__':
    main()