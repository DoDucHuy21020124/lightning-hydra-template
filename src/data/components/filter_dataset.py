from typing import Dict, List
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import albumentations
import matplotlib.pyplot as plt
from src.data.components import utils_dataset
import os
import torch
import torchvision

class FilterDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str = 'data/ibug_300W_large_face_landmark_dataset/',
        #data_dir: str,
        xml_file: str = '',
        num_of_keypoints: int = 68,
    ):
        self.num_of_keypoints = num_of_keypoints
        self.data_dir = data_dir
        
        image_xml = utils_dataset.get_imagexml(os.path.join(self.data_dir, xml_file))
        data = utils_dataset.init_dataframe(self.num_of_keypoints)
        data = utils_dataset.get_data(image_xml, data)
        self.data = data

        self.x = self.data['file_path'][:128]

        self.y = self.data['keypoints']

        self.box = self.data['box']

        self.bbox_classes = ['person']
        self.keypoints_classes = [str(i) for i in range(num_of_keypoints)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        file_path, label, box = self.x[index], self.y[index], self.box[index]
        image = np.asarray(Image.open(os.path.join(self.data_dir, file_path)).convert('RGB'))
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[0] + box[2])
        y_max = int(box[1] + box[3])
        if x_min < 0:
            box[2] += float(x_min)
            x_min = 0
        if y_min < 0:
            box[3] += float(y_min)
            y_min = 0
        if x_max > image.shape[1]:
            box[2] -= float(x_max - image.shape[1])
            x_max = image.shape[1]
        if y_max > image.shape[0]:
            box[3] -= float(y_max - image.shape[0])
            y_max = image.shape[0]
        box[0], box[1] = 0, 0
        last_transform = albumentations.Crop(x_min, y_min, x_max, y_max)
        transformed = last_transform(image = image)
        image = transformed['image']

        for i in range(label.shape[0]):
            label[i][0] -= x_min
            label[i][1] -= y_min
        return image, label, box
    
    def get_raw_data(self, index: int):
        return self.x[index], self.y[index], self.box[index]

    @staticmethod
    def draw_image_with_keypoints(
        image: torch.Tensor,
        keypoints: torch.Tensor,
        width: float = None,
        height: float = None,
        normalize: bool = False
    ) -> None:
        assert width is not None and height is not None
        if normalize:
            keypoints = (keypoints + 0.5) * torch.Tensor([width, height])
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for i in range(keypoints.shape[0]):
            if keypoints[i][0] is not None and keypoints[i][1] is not None: # and keypoints[i][0] >= 0 and keypoints[i][0] <= width and keypoints[i][1] >= 0 and keypoints[i][1] <= height:
                draw.ellipse((keypoints[i][0] - 2, keypoints[i][1] - 2, keypoints[i][0] + 2, keypoints[i][1] + 2), fill = (255, 255, 0))
        return image

    @staticmethod
    def draw_image_with_keypoints_and_boundingbox(
        image: np.ndarray,
        box: List,
        keypoints: List,
        width: float = None,
        height: float = None,
        normalize: bool = False
    ) -> None:
        left_box = box[0]
        top_box = box[1]
        width_box = box[2]
        height_box = box[3]
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.rectangle((left_box, top_box, left_box + width_box, top_box + height_box), width = 2, outline = (0, 255, 0))
        print('image', type(image))
        image = FilterDataset.draw_image_with_keypoints(image, keypoints, width, height, normalize)
        return image

if __name__ == '__main__':
    from omegaconf import DictConfig
    import pyrootutils
    import hydra

    root = pyrootutils.setup_root(__file__, pythonpath= True)
    data_dir = root / 'data/ibug_300W_large_face_landmark_dataset/'
    config_path = str(root / 'configs'/ 'data'/ 'data_test')
    data_dir = root / 'data/ibug_300W_large_face_landmark_dataset/'
    
    @hydra.main(config_path=config_path, config_name='filter_test.yaml')
    def main(cfg: DictConfig):
        filter = hydra.utils.instantiate(cfg)
        filter = filter(data_dir = data_dir)
        print(filter)
        print(len(filter))
        x, y, box = filter[10]
        print(x.shape, y.shape)
        print(x.dtype)

        image = Image.fromarray(x)
        
        
        # image1 = FilterDataset.draw_image_with_keypoints(
        #     image = x,
        #     keypoints = y,
        #     width = x.shape[1],
        #     height = x.shape[0],
        #     normalize = False
        # )
        # print(image1)
        # image1 = np.array(image1)
        # print(image1.dtype)
        # plt.imshow(image1)
        # plt.show()

        # image2 = filter.x[3]
        # image2 = Image.open(os.path.join(str(filter.data_dir), image2))
        # keypoints = filter.y[3]
        # box = filter.box[3]
        # image2 = FilterDataset.draw_image_with_keypoints_and_boundingbox(
        #     image = image2,
        #     box = box,
        #     keypoints = keypoints,
        #     width = image2.size[1],
        #     height = image2.size[0],
        #     normalize = False
        # )
        # print(image2)
        # plt.imshow(image2)
        # plt.show()
    main()
    #_ = FilterDataset()