import os
from PIL import Image
import torch
import numpy as np

class ZeroDataset(torch.utils.data.Dataset):
    def __init__(self, sparse=False, aug_params=None):
        self.image_list = []
        self.disp_list = []
        self.conf_list = []
        self.augmentor = None

        if aug_params:
            self.augmentor = Augmentor(sparse, aug_params)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        print(index)
        left = Image.open(self.image_list[index][0]).convert('RGB')
        right = Image.open(self.image_list[index][1]).convert('RGB')
        disp = np.load(self.disp_list[index])
        conf = np.load(self.conf_list[index])

        left = np.array(left).astype(np.float32)
        right = np.array(right).astype(np.float32)
        disp = np.array(disp).astype(np.float32)
        conf = np.array(conf).astype(np.float32)
        valid = (disp > 0).astype(np.float32)

        left = torch.from_numpy(left).permute(2, 0, 1).float()
        right = torch.from_numpy(right).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp)[None].float()
        conf = torch.from_numpy(conf)[None].float()
        valid = torch.from_numpy(valid)[None].float()

        return left, right, disp, conf, valid

class MfS35K(ZeroDataset):
    def __init__(self, aug_params=None, root='/data/wxq/mfs35k', filelist='filelist/mfs35k_v2.txt'):
        super(MfS35K, self).__init__(aug_params)

        with open(filelist, 'r') as f:
            line = f.readline().rstrip('\n')
            while line:
                filename = line.split(' ')
                filename = [os.path.join(root, fn) for fn in filename]
                self.image_list += [[filename[0], filename[1]]]
                self.disp_list += [filename[2]]
                self.conf_list += [filename[3]]
                line = f.readline().rstrip('\n')

        assert len(self.image_list) == len(self.disp_list) == len(self.conf_list)