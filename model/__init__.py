from hydra.utils import instantiate

def fetch_model(cfg, logger):
    model = None

    if cfg.model.name in ['RAFTStereo', 'IGEVStereo']:
        model = instantiate(cfg.model.instance)
    
    logger.info(f'Loading model from {cfg.model.name}.')

    return model