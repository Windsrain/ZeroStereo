import hydra
import torch
from tqdm import tqdm
from accelerate import Accelerator, load_checkpoint_and_dispatch
from accelerate.logging import get_logger
from model import fetch_model
from dataset import fetch_dataloader
from util import InputPadder

@hydra.main(version_base=None, config_path='config', config_name='evaluate_stereo')
def main(cfg):
    accelerator = Accelerator()
    logger = get_logger(__name__)
    
    dataloader = fetch_dataloader(cfg, logger)
    model = fetch_model(cfg, logger)
    model = load_checkpoint_and_dispatch(model, cfg.checkpoint)
    logger.info(f'Loading checkpoint from {cfg.checkpoint}.')

    model = accelerator.prepare_model(model)
    for name in dataloader:
        dataloader[name] = accelerator.prepare_data_loader(dataloader[name])

    for name in dataloader:
        model.eval()
        total_elem, total_epe, total_out = 0, 0, 0
        for data in tqdm(dataloader[name], dynamic_ncols=True, disable=not accelerator.is_main_process):
            left, right, disp_gt, valid = [x for x in data]
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)

            with torch.no_grad():
                if cfg.model.name == 'RAFTStereo':
                    _, disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    disp_pred = -disp_pred
                elif cfg.model.name == 'IGEVStereo':
                    disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                else:
                    raise Exception(f'Invalid model name: {cfg.model.name}.')
                
            disp_pred = padder.unpad(disp_pred)
            assert disp_pred.shape == disp_gt.shape

            epe = torch.abs(disp_pred - disp_gt)
            out = (epe > cfg.dataset[name].outlier).float()
            epe, out = accelerator.gather_for_metrics((epe[valid >= 0.5].mean(), out[valid >= 0.5].mean()))

            total_elem += epe.shape[0]
            total_epe += epe.sum().item()
            total_out += out.sum().item()
        accelerator.print(f'{name}/EPE: {total_epe / total_elem:.2f}, {name}/Bad {cfg.dataset[name].outlier}px: {100 * total_out / total_elem:.2f}.')

    accelerator.end_training()

if __name__ == '__main__':
    main()