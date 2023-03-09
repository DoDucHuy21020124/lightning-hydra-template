from typing import List
from numpy import ndarray
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import albumentations
from omegaconf import DictConfig

class ImageTransform():
    def __init__(
        self,
        phase: str = 'train',
        transform: albumentations.Compose = None
    ):
        self.transform = transform
        self.box = None
        self.phase = phase

    def __call__(
        self,
        x: ndarray,
        box: List[float] = None,
        bbox_classes: List[str] = None,
        keypoints: List[float] = None,
        keypoints_classes: List[str] = None
    ):
        self.box = box
        image_transformed, box_transformed, keypoints_transformed= None, None, None
        if self.phase == 'train':
            transformed = self.transform(
                image = x,
                bboxes = box,
                bbox_classes = bbox_classes,
                keypoints = keypoints,
                keypoints_classes = keypoints_classes
            )
            image_transformed = transformed['image']
            box_transformed = transformed['bboxes']
            keypoints_transformed = transformed['keypoints']
        elif self.phase == 'val':
            transformed = self.transform(image = x)
            image_transformed = transformed['image']
        else:
            transformed = self.transform(image = x)
            image_transformed = self.transform(image = x)
        return image_transformed, box_transformed, keypoints_transformed
    
if __name__ == '__main__':
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath= True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "data" / "transform" / "filter_transform" / "train_transform.yaml"
    )
    transform = hydra.utils.instantiate(cfg)
    image = Image.open(str(root) + '\\data\\ibug_300W_large_face_landmark_dataset\\afw\\815038_1_mirror.jpg')
    #image = transform(np.asarray(image))
    plt.imshow(image)
    plt.show()