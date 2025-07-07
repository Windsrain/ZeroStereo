import torch
from hydra.utils import instantiate

def fetch_model(cfg, logger):
    model = None

    if cfg.model.name in ['DepthAnythingV2', 'RAFTStereo', 'IGEVStereo']:
        model = instantiate(cfg.model.instance)
    elif cfg.model.name in ['StereoGen']:
        if cfg.accelerator.mixed_precision == 'fp16':
            model = instantiate(cfg.model.instance, torch_dtype=torch.float16)
        else:
            model = instantiate(cfg.model.instance)
        model.enable_xformers_memory_efficient_attention()
        model.encode_empty_text()
    else:
        raise Exception(f'Invalid model name: {cfg.model.name}.')
    
    logger.info(f'Loading model from {cfg.model.name}.')

    return model