import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from diffusers import DiffusionPipeline

class StereoGen(DiffusionPipeline):
    def __init__(
        self, unet, vae, scheduler, text_encoder, tokenizer):
        super().__init__()
        self.register_modules(unet=unet, vae=vae, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer)
        self.empty_text_embed = None

    def encode_empty_text(self):
        prompt = ''
        text_inputs = self.tokenizer(prompt, padding='do_not_pad', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    def encode_image(self, image):
        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        image_latent = mean * self.vae.config.scaling_factor

        return image_latent

    def decode_image(self, image_latent):
        image_latent = image_latent / self.vae.config.scaling_factor
        z = self.vae.post_quant_conv(image_latent)
        image = self.vae.decoder(z)
        
        return image

    @torch.no_grad()
    def single_infer(self, masked_image, mask, num_inference_steps):
        device = self.device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        masked_image_latent = self.encode_image(masked_image)
        mask = F.interpolate(mask, size=masked_image_latent.shape[-2:])
        image_latent = torch.randn(masked_image_latent.shape, device=device, dtype=self.dtype)

        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(masked_image_latent.shape[0], 1, 1).to(device)

        for i, t in enumerate(timesteps):
            unet_input = torch.cat([image_latent, mask, masked_image_latent], dim=1)
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample
            image_latent = self.scheduler.step(noise_pred, t, image_latent).prev_sample

        image = self.decode_image(image_latent)
        image = image.clamp(-1, 1)

        return image