from hydra.utils import instantiate
from torch.utils.data import DataLoader, ConcatDataset

def fetch_dataloader(cfg, dataset_cfg, dataloader_cfg, logger, return_dataset=False):
    dataset = []
    for name in dataset_cfg:
        ds = instantiate(dataset_cfg[name].instance)
        logger.info(f'Reading {len(ds)} samples from {name}.')
        dataset.append(ds)

    gpu_num = len(cfg.gpus.split(','))
    if dataloader_cfg.batch_size_per_gpu:
        dataloader_cfg.param.batch_size = dataloader_cfg.batch_size_per_gpu * gpu_num 
    else:
        dataloader_cfg.param.batch_size = dataloader_cfg.total_batch_size
    dataloader_cfg.param.num_workers = dataloader_cfg.param.batch_size // gpu_num

    dataloader = None
    if dataloader_cfg.split:
        dataloader = {}
        for name, ds in zip(dataset_cfg, dataset):
            dataloader[name] = DataLoader(ds, **dataloader_cfg.param)
    else:
        dataset = ConcatDataset(dataset)
        dataloader = DataLoader(dataset, **dataloader_cfg.param)

    if return_dataset:
        return dataset, dataloader

    return dataloader