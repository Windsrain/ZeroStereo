from hydra.utils import instantiate
from torch.utils.data import DataLoader, ConcatDataset

def fetch_dataloader(cfg, logger):
    dataset = []
    for name in cfg.dataset:
        ds = instantiate(cfg.dataset[name].instance)
        logger.info(f'Reading {len(ds)} samples from {name}.')
        dataset.append(ds)

    if cfg.dataloader.batch_size_per_gpu:
        cfg.dataloader.param.batch_size = cfg.dataloader.batch_size_per_gpu
    else:
        num_gpus = len(cfg.gpus.split(','))
        cfg.dataloader.param.batch_size = cfg.dataloader.total_batch_size // num_gpus

    dataloader = None
    if cfg.dataloader.split:
        dataloader = {}
        for name, ds in zip(cfg.dataset, dataset):
            dataloader[name] = DataLoader(ds, **cfg.dataloader.param)
    else:
        dataset = ConcatDataset(dataset)
        dataloader = DataLoader(dataset, **cfg.dataloader.param)

    return dataloader