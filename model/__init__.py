import torch
from hydra.utils import instantiate

def fetch_model(cfg, logger):
    model = instantiate(cfg.model.instance)

    if cfg.model.name in ['StableDiffusion', 'StereoGen']:
        model.vae.eval()
        model.text_encoder.eval()
        model.enable_xformers_memory_efficient_attention()
        model.encode_empty_text()
    
    logger.info(f'Loading model from {cfg.model.name}.')

    return model