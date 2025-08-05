import os
import hydra
import torch
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
from diffusers import DDPMScheduler
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from util.padder import InputPadder
from model import fetch_model
from dataset import fetch_dataloader

@hydra.main(version_base=None, config_path='config', config_name='train_stereogen')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    accelerator = instantiate(cfg.accelerator)
    accelerator.init_trackers(project_name=cfg.tracker.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs=cfg.tracker.init_kwargs)

    train_loader = fetch_dataloader(cfg, cfg.train_set, cfg.train_loader, logger)
    model = fetch_model(cfg, logger)
    optimizer = instantiate(cfg.optimizer, _partial_=True)(model.unet.parameters())
    scheduler = instantiate(cfg.scheduler, _partial_=True)(optimizer)
    loss_function = torch.nn.MSELoss(reduction='mean')
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.instance.pretrained_model_name_or_path + '/scheduler')

    train_loader, model.unet, optimizer, scheduler = accelerator.prepare(train_loader, model.unet, optimizer, scheduler)
    model.vae.to(accelerator.device, torch.float16 if cfg.accelerator.mixed_precision == 'fp16' else torch.float32)

    set_seed(cfg.seed, device_specific=True)
    step, acc_loss = 0, 0
    should_keep_training = True
    while should_keep_training:
        model.unet.train()
        for data in tqdm(train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model.unet):
                image, warped_image, mask = [x for x in data]
                image = image / 127.5 - 1.
                warped_image = warped_image / 127.5 - 1.
                masked_image = warped_image * (mask < 0.5)

                with accelerator.autocast():
                    with torch.no_grad():
                        masked_image_latent = model.encode_image(masked_image)
                        image_latent = model.encode_image(image)
                        mask = torch.nn.functional.interpolate(mask, size=masked_image_latent.shape[-2:])
                        text_embed = model.empty_text_embed.to(accelerator.device).repeat(image_latent.shape[0], 1, 1)            

                    timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (image.shape[0],), device=accelerator.device)
                    noise = torch.randn(image_latent.shape, device=accelerator.device)
                    noisy_image_latent = noise_scheduler.add_noise(image_latent, noise, timestep)

                    unet_input = torch.cat([noisy_image_latent, mask, masked_image_latent], dim=1)
                    noise_pred = model.unet(unet_input, timestep, encoder_hidden_states=text_embed).sample

                loss = loss_function(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            acc_loss += loss.detach()

            if accelerator.sync_gradients:
                step += 1
                acc_loss = accelerator.reduce(acc_loss.detach(), reduction='mean') / cfg.accelerator.gradient_accumulation_steps
                accelerator.log({'train/loss': acc_loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, step)
                acc_loss = 0

                if (step > 0) and (step % cfg.save_freq == 0):
                    accelerator.save_state(os.path.join(cfg.save_path, str(step)))

                if step == cfg.scheduler.total_steps:
                    should_keep_training = False
                    break

    if accelerator.is_main_process:
        model.unet = accelerator.unwrap_model(model.unet)
        model.save_pretrained(os.path.join(cfg.save_path, 'final'))

    accelerator.end_training()

if __name__ == '__main__':
    main()