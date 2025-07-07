import os
import cv2
import torch
import numpy as np
from PIL import Image
from util.util import warp_image

class WarpDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.image_list = []
        self.disp_list = []

    def __len__(self):
        return len(self.image_list)

    def _getitem(self, index, scale):
        image = Image.open(self.image_list[index]).convert('RGB')
        image = np.array(image).astype(np.float32)[..., :3]
        disp = np.load(self.disp_list[index])

        h, w = image.shape[:2]
        if scale < 0:
            scale = min(h, w) // 1000

        if scale > 1:
            image = cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC).clip(0, 255)
            disp = cv2.resize(disp, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR) / scale

        warped_image, mask_nocc, mask_inpaint = warp_image(image, disp)

        warped_image = torch.from_numpy(warped_image).permute(2, 0, 1).float()
        mask_nocc = torch.from_numpy(mask_nocc)[None].float()
        mask_inpaint = torch.from_numpy(mask_inpaint)[None].float()      

        return self.image_list[index], warped_image, mask_nocc, mask_inpaint, scale, h, w

    def __getitem__(self, index):
        return self._getitem(index, scale=-1)

class MfS35K(WarpDataset):
    def __init__(self, root='/data/datasets/mfs35k', filelist='filelist/mfs35k.txt'):
        super(MfS35K, self).__init__()
        
        with open(filelist, 'r') as f:
            line = f.readline().rstrip('\n')
            while line:
                filenames = line.split(' ')
                filenames = [os.path.join(root, fn) for fn in filenames]
                self.image_list += [filenames[0]]
                self.disp_list += [filenames[2]]
                line = f.readline().rstrip('\n')

        assert len(self.image_list) == len(self.disp_list)