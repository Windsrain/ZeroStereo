import os
from PIL import Image
import torch
import numpy as np
from util.augmentor import Augmentor

class ZeroDataset(torch.utils.data.Dataset):
    def __init__(self, sparse=False, aug_params=None):
        self.image_list = []
        self.disp_list = []
        self.conf_list = []
        self.mask_nocc_list = []
        self.mask_inpaint_list = []
        self.augmentor = None

        if aug_params:
            self.augmentor = Augmentor(sparse, aug_params)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        left = Image.open(self.image_list[index][0]).convert('RGB')
        right = Image.open(self.image_list[index][1]).convert('RGB')
        disp = np.load(self.disp_list[index])
        conf = np.load(self.conf_list[index])
        mask_nocc = Image.open(self.mask_nocc_list[index]).convert('L')
        mask_inpaint = Image.open(self.mask_inpaint_list[index]).convert('L')

        left = np.array(left).astype(np.float32)
        right = np.array(right).astype(np.float32)
        disp = np.array(disp).astype(np.float32)
        conf = np.array(conf).astype(np.float32)
        mask_nocc = (np.array(mask_nocc) == 255).astype(np.float32)
        mask_inpaint = (np.array(mask_inpaint) == 255).astype(np.float32)
        valid = (disp > 0.).astype(np.float32)

        left, right, left_clean, right_clean, disp, conf, mask_nocc, mask_inpaint, valid = self.augmentor(left=left, right=right, left_clean=left, right_clean=right, disp=disp, conf=conf, mask_nocc=mask_nocc, mask_inpaint=mask_inpaint, valid=valid)

        left = torch.from_numpy(left).permute(2, 0, 1).float()
        right = torch.from_numpy(right).permute(2, 0, 1).float()
        left_clean = torch.from_numpy(left_clean).permute(2, 0, 1).float()
        right_clean = torch.from_numpy(right_clean).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp)[None].float()
        conf = torch.from_numpy(conf)[None].float()
        mask_nocc = torch.from_numpy(mask_nocc)[None].float()
        mask_inpaint = torch.from_numpy(mask_inpaint)[None].float()
        valid = torch.from_numpy(valid)[None].float()

        return left, right, left_clean, right_clean, disp, conf, mask_nocc, mask_inpaint, valid

class MfS35K(ZeroDataset):
    def __init__(self, aug_params=None, root='/data/datasets/mfs35k', filelist='filelist/mfs35k.txt'):
        super(MfS35K, self).__init__(aug_params=aug_params)

        with open(filelist, 'r') as f:
            line = f.readline().rstrip('\n')
            while line:
                filenames = line.split(' ')
                filenames = [os.path.join(root, fn) for fn in filenames]
                self.image_list += [[filenames[0], filenames[1]]]
                self.disp_list += [filenames[2]]
                self.conf_list += [filenames[3]]
                self.mask_nocc_list += [filenames[4]]
                self.mask_inpaint_list += [filenames[5]]
                line = f.readline().rstrip('\n')

        assert len(self.image_list) == len(self.disp_list) == len(self.conf_list) == len(self.mask_nocc_list) == len(self.mask_inpaint_list)