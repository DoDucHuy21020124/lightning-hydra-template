from typing import List
from numpy import ndarray
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import albumentations
import albumentations.pytorch
from src.data.components import filter_dataset
import torch
import math


class ImageTransform(Dataset):
    def __init__(
        self,
        data: filter_dataset.FilterDataset = None,
        width: float = 224,
        height: float = 224,
        transform: albumentations.Compose = None
    ):
        
        self.data = data
        self.width = width
        self.height = height

        # if transform is None:
        #     self.transform = albumentations.Compose([
        #         albumentations.Resize(height = self.height, width = self.width),
        #         albumentations.Normalize(),
        #         albumentations.pytorch.ToTensorV2()
        #     ])
        # else:
        self.transform = transform
        self.bbox_classes = ['person']
        self.keypoints_classes = [str(i) for i in range(68)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, label, box = self.data[index]
        transformed = self.transform(
            image = image,
            bboxes = [box],
            bbox_classes = self.bbox_classes,
            keypoints = label,
            keypoints_classes = self.keypoints_classes
        )
        image = transformed['image']
        keypoints = transformed['keypoints']
        label = []
        for i in range(len(keypoints)):
            label.append(keypoints[i][0])
            label.append(keypoints[i][1])
        label = torch.Tensor(label)
        label = label.reshape(-1, 2)
        print(label.shape)
        label = label / torch.Tensor([self.width, self.height]) - 0.5
        return image, label
    
    @staticmethod
    def draw_batch_image(
        images: torch.Tensor,
        labels: torch.Tensor,
        width: float = None,
        height: float = None,
        normalize: bool = False
    ) -> None:
        fig = plt.figure(figsize = (4, 4))
        plt.subplots_adjust(left=0,bottom=0,right=1,top=1, wspace=0, hspace=0)
        number_image = images.size()[0]
        row = int(math.sqrt(number_image))
        col = None
        if number_image % row == 0:
            col = int(number_image / row)
        else:
            col = int(number_image / row) + 1

        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
            # 3, H, W, B
            ten = x.clone().permute(1, 2, 3, 0)
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
            # B, 3, H, W
            return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
        
        images = denormalize(images)
        for i in range(images.size()[0]):
            keypoints = labels[i]
            image = images[i].squeeze().permute(1, 2, 0)
            fig.add_subplot(row, col, i + 1)
            filter_dataset.FilterDataset.draw_image_with_keypoints(image, keypoints, width, height, normalize)
    
if __name__ == '__main__':
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath= True)
    filter_cfg = omegaconf.OmegaConf.load(
        root/ 'configs'/ 'data'/ 'data_train'/ 'filter_train.yaml'
    )
    data_dir = root/ 'data/ibug_300W_large_face_landmark_dataset/'
    filter = hydra.utils.instantiate(filter_cfg)
    filter = filter(data_dir = data_dir)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "data" / "train_transform"/ "train_transform.yaml"
    )
    transform = hydra.utils.instantiate(cfg)
    transform = ImageTransform(data = filter, transform= transform)
    x, y = transform[3]
    filter_dataset.FilterDataset.draw_image_with_keypoints(
        x.permute(1, 2, 0),
        keypoints = y,
        width = transform.width,
        height = transform.height,
        normalize= True
    )
    plt.show()