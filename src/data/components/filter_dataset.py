from typing import Dict, List
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import albumentations
import matplotlib.pyplot as plt
from src.data.components import utils_dataset
from matplotlib.patches import Rectangle
import os

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

        self.x = self.data['file_path']

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
        image: np.ndarray,
        keypoints: np.ndarray,
        width: float = None,
        height: float = None,
        normalize: bool = False
    ) -> None:
        assert width is not None and height is not None
        if normalize:
            for i in range(keypoints.shape[0]):
                keypoints[i][0] = (keypoints[i][0] + 0.5) * width
                keypoints[i][1] = (keypoints[i][1] + 0.5) * height
            #print(keypoints)
        plt.xlim(width)
        plt.ylim(height)
        plt.imshow(image)
        for i in range(keypoints.shape[0]):
            if keypoints[i][0] is not None and keypoints[i][1] is not None and keypoints[i][0] >= 0 and keypoints[i][0] <= width and keypoints[i][1] >= 0 and keypoints[i][1] <= height:
                plt.plot(float(keypoints[i][0]), float(keypoints[i][1]), marker = '.', markersize = 0.5, color = 'red')
        #plt.scatter(keypoints[:, 0], keypoints[:, 1], s = 0.5, c = 'red')
        plt.axis('off')

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

        plt.gca().add_patch(Rectangle(
            xy = (left_box, top_box),
            width = width_box,
            height = height_box,
            fill = False,
            edgecolor = 'green',
            lw = 2
        ))
        FilterDataset.draw_image_with_keypoints(image, keypoints, width, height, normalize)

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
        x, y, _ = filter[10]
        print(x.shape, y.shape)
        
        FilterDataset.draw_image_with_keypoints(
            image = x,
            keypoints = y,
            width = x.shape[1],
            height = x.shape[0],
            normalize = False
        )
        print(y.shape)
        plt.show()
    main()
    #_ = FilterDataset()