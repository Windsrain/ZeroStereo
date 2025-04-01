import numpy as np
from torchvision.transforms import ColorJitter

class Augmentor:
    def __init__(self, sparse, aug_params):
        self.sparse = sparse
        self.aug_params = aug_params
        self.color_jitter = None
        self.scale_prob = 0.0

        if self.aug_params.color_jitter:
            self.color_jitter = ColorJitter(brightness=list(aug_params.color_jitter.brightness), contrast=list(aug_params.color_jitter.contrast), saturation=list(aug_params.color_jitter.saturation), hue=aug_params.color_jitter.hue / 3.14)

        if self.aug_params.random_scale:
            self.scale_prob = self.aug_params.random_scale.scale_prob

    def color_transform(self, *image):
        