import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image

import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class ImageNet22k(data.Dataset):
    """
    ImageNet-22K dataset adapted from: https://github.com/microsoft/Swin-Transformer/blob/main/data/imagenet22k_dataset.py
    """

    def __init__(self, root, anno_file='', transform=None, target_transform=None):
        super(ImageNet22k, self).__init__()

        self.data_path = root
        self.anno_path = os.path.join(self.data_path, anno_file)
        self.transform = transform
        self.target_transform = target_transform
        self.database = json.load(open(self.anno_path))

    def _load_image(self, path):
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))

        return im

    def __getitem__(self, index):
        idb = self.database[index]

        # get image
        image = self._load_image(os.path.join(self.data_path, idb[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # get target
        target = int(idb[1])
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.database)
