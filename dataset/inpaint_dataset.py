import os
import torch
from PIL import Image
from glob import glob
from util.util import warp_image
from util.reader import *
from util.augmentor import Augmentor

class InpaintDataset(torch.utils.data.Dataset):
    def __init__(self, sparse=False, aug_params=None, reader=None):
        self.sparse = sparse
        self.image_list = []
        self.disp_list = []
        self.reader = reader
        self.augmentor = None

        if aug_params:
            self.augmentor = Augmentor(sparse, aug_params)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        left = Image.open(self.image_list[index][0])
        right = Image.open(self.image_list[index][1])
        disp = self.reader(self.disp_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024

        left = np.array(left).astype(np.float32)[..., :3]
        right = np.array(right).astype(np.float32)[..., :3]
        disp = np.array(disp).astype(np.float32)
        valid = np.array(valid).astype(np.float32) 

        if self.augmentor is not None:
            left, right, disp, valid = self.augmentor(left=left, right=right, disp=disp, valid=valid)

        warped_right, mask_nocc, mask_inpaint = warp_image(left, disp)

        right = torch.from_numpy(right).permute(2, 0, 1).float()
        warped_right = torch.from_numpy(warped_right).permute(2, 0, 1).float()
        mask_inpaint = torch.from_numpy(mask_inpaint)[None].float()

        return right, warped_right, mask_inpaint   

class TartanAir(InpaintDataset):
    def __init__(self, aug_params=None, root='/data/datasets/tartanair'):
        super().__init__(aug_params=aug_params, reader=tartanair_disp_reader)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, '*/*/*/*/image_left/*.png')))
        image2_list = sorted(glob(os.path.join(root, '*/*/*/*/image_right/*.png')))
        disp_list = sorted(glob(os.path.join(root, '*/*/*/*/depth_left/*.npy')))

        for image1, image2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [image1, image2] ]
            self.disp_list += [ disp ]

class CREStereoDataset(InpaintDataset):
    def __init__(self, aug_params=None, root='/data/datasets/crestereo'):
        super(CREStereoDataset, self).__init__(aug_params=aug_params, reader=crestereo_disp_reader)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/*_left.disp.png')))

        for idx, (image1, image2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [image1, image2] ]
            self.disp_list += [ disp ]

class FallingThings(InpaintDataset):
    def __init__(self, aug_params=None, root='/data/datasets/fallingthings'):
        super().__init__(aug_params=aug_params, reader=fallingthings_disp_reader)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/*/*/*left.jpg'))
        image2_list = sorted(glob(root + '/*/*/*right.jpg'))
        disp_list = sorted(glob(root + '/*/*/*left.depth.png'))

        image1_list += sorted(glob(root + '/*/*/*/*left.jpg'))
        image2_list += sorted(glob(root + '/*/*/*/*right.jpg'))
        disp_list += sorted(glob(root + '/*/*/*/*left.depth.png'))

        for image1, image2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [image1, image2] ]
            self.disp_list += [ disp ]

class SceneFlow(InpaintDataset):
    def __init__(self, aug_params=None, root='/data/datasets/sceneflow', dstype='frames_finalpass', things_test=False):
        super(SceneFlow, self).__init__(aug_params=aug_params, reader=pfm_reader)
        assert os.path.exists(root)
        self.root = root
        self.dstype = dstype
        
        if aug_params is None:
            things_test = True

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disp_list)
        root = self.root
        left_images = sorted( glob(os.path.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        val_idxs = np.linspace(0, len(left_images) - 1, 500, dtype=int)
        for idx, (image1, image2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [image1, image2] ]
                self.disp_list += [ disp ]

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disp_list)
        root = self.root
        left_images = sorted( glob(os.path.join(root, self.dstype, split, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for image1, image2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [image1, image2] ]
            self.disp_list += [ disp ]

    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disp_list)
        root = self.root
        left_images = sorted( glob(os.path.join(root, self.dstype, split, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for image1, image2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [image1, image2] ]
            self.disp_list += [ disp ]

class VKITTI2(InpaintDataset):
    def __init__(self, aug_params=None, root='/data/datasets/vkitti2'):
        super(VKITTI2, self).__init__(aug_params=aug_params, reader=vkitti2_disp_reader)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_0/rgb*.jpg')))
        image2_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_1/rgb*.jpg')))
        disp_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/depth/Camera_0/depth*.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (image1, image2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [image1, image2] ]
            self.disp_list += [ disp ]