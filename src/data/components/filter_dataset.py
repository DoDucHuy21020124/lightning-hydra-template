import torch.utils.data as data
import pandas as pd
from components import utils_dataset
from PIL import Image
import numpy as np
import albumentations
from omegaconf import DictConfig
from hydra.utils import instantiate

class FilterDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str = 'data\\',
        width: float = 224,
        height: float = 224,
        train: bool = True,
        num_of_keypoints: int = 68,
        transform = None
    ):
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.train = train
        self.transform = transform
        self.num_of_keypoints = num_of_keypoints

        image_xml = None
        image_xml = utils_dataset.get_imagexml(self.data_dir)

        data = utils_dataset.init_dataframe(self.num_of_keypoints)
        data = utils_dataset.get_data(image_xml, data)

        self.df = pd.DataFrame(data)

        file = self.df['file_path']
        self.x = list()
        for i in range(256):
            image = Image.open(self.data_dir + '\\' + file[i])
            self.x.append(np.asarray(image))

        self.y = list()
        for i in range(60):
            temp = list()
            for j in range(self.num_of_keypoints):
                temp.append(self.df.iloc[i][str(i)][0])
                temp.append(self.df.iloc[i][str(i)][1])
            self.y.append(temp)

        self.box = self.df['box']
        self.bbox_classes = ['person']

        self.keypoints = list()
        for i in range(60):
            temp = list()
            for j in range(self.num_of_keypoints):
                temp.append(self.df.iloc[i][str(j)])
            self.keypoints.append(temp)
        self.keypoints_classes = [str(i) for i in range(self.num_of_keypoints)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        image, label, box, keypoints = self.x[index], self.y[index], self.box[index], self.keypoints[index]
        if self.transform is not None:
            if self.train:
                image, box, keypoints = self.transform(
                    image = image,
                    bboxes = [self.box[index]],
                    bbox_classes = self.bbox_classes,
                    keypoints = self.keypoints[index],
                    keypoints_classes = self.keypoints_classes
                )

                label = utils_dataset.change_keypoints_to_label(keypoints)
            else:
                image, _, _ = self.transform(image = image)
        
        for i in range(1, len(label), step = 2):
            label[i - 1] = label[i - 1] / self.width
            label[i] = label[i] / self.height
        
        last_transform = albumentations.Crop(box[0], box[1], box[0] + box[2], box[1] + box[3])
        return image, label
    
    def get_raw_data(self, index: int):
        return self.x[index], self.keypoints[index], self.box[index]
    
    def get_transformed_data(self, index: int):
        image, box, keypoints = self.x[index], self.box[index], self.keypoints[index]
        if self.transform is not None:
            if self.train:
                image, box, keypoints = self.transform(
                    image = image,
                    bboxes = box,
                    bbox_classes = self.bbox_classes,
                    keypoints = keypoints,
                    keypoints_classes = self.keypoints_classes)
            else:
                image, _, _ = self.transform(image = image)
        return image, box, keypoints

if __name__ == '__main__':
    import omegaconf
    import pyrootutils
    import hydra

    root = pyrootutils.setup_root(__file__, pythonpath= True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs"/ "data"/ "filter_tran_test"/"filter_train.yaml"
    )
    filter = hydra.utils.instantiate(cfg)
    print(filter)
    #_ = FilterDataset()