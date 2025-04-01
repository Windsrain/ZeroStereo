import os
from glob import glob
import torch
from PIL import Image
from util.reader import *

class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, mask=None, reader=None):
        self.image_list = []
        self.disp_list = []
        self.mask = mask
        self.reader = reader

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        left = Image.open(self.image_list[index][0])
        right = Image.open(self.image_list[index][1])
        disp, valid = self.reader(self.disp_list[index], self.mask)

        left = np.array(left).astype(np.float32)
        right = np.array(right).astype(np.float32)
        disp = np.array(disp).astype(np.float32)
        valid = np.array(valid).astype(np.float32)

        left = torch.from_numpy(left).permute(2, 0, 1)
        right = torch.from_numpy(right).permute(2, 0, 1)
        disp = torch.from_numpy(disp)[None]
        valid = torch.from_numpy(valid)[None]

        return left, right, disp, valid

class KITTI(StereoDataset):
    def __init__(self, root='data/datasets/kitti', mask='all'):
        super(KITTI, self).__init__(mask, reader=kitti_disp_reader)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, '2015', 'training', 'image_2/*_10.png')))
        right_list = sorted(glob(os.path.join(root, '2015', 'training', 'image_3/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, '2015', 'training', 'disp_occ_0/*_10.png')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class Middlebury(StereoDataset):
    def __init__(self, root='/data/datasets/middlebury', mask='noc', resolution='H'):
        super(Middlebury, self).__init__(mask, reader=middlebury_disp_reader)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, 'MiddEval3', f'training{resolution}', '*/im0.png')))
        right_list = sorted(glob(os.path.join(root, 'MiddEval3', f'training{resolution}', '*/im1.png')))
        disp_list = sorted(glob(os.path.join(root, 'MiddEval3', f'training{resolution}', '*/disp0GT.pfm')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class ETH3D(StereoDataset):
    def __init__(self, root='/data/datasets/eth3d', mask='noc'):
        super(ETH3D, self).__init__(mask, reader=eth3d_disp_reader)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, 'two_view_training/*/im0.png')))
        right_list = sorted(glob(os.path.join(root, 'two_view_training/*/im1.png')))
        disp_list = sorted(glob(os.path.join(root, 'two_view_training_gt/*/disp0GT.pfm')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]