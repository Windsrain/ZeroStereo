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
        