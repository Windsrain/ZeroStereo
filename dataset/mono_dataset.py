import os
import cv2
import torch

class MonoDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.image_list = []

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return self.image_list[index], image

class MfS35K(MonoDataset):
    def __init__(self, root='/data/datasets/mfs35k', filelist='filelist/mfs35k.txt'):
        super(MfS35K, self).__init__()
        
        with open(filelist, 'r') as f:
            line = f.readline().rstrip('\n')
            while line:
                filenames = line.split(' ')
                filenames = [os.path.join(root, fn) for fn in filenames]
                self.image_list += [filenames[0]]
                line = f.readline().rstrip('\n')