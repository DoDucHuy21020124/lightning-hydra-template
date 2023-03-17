import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import albumentations
import matplotlib.pyplot as plt
from components import utils_dataset
import torch

class FilterDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str = 'data',
        width: float = 224,
        height: float = 224,
        train: bool = True,
        num_of_keypoints: int = 68,
        transform = None
    ):
        self.data_dir = str(data_dir) + '\\ibug_300W_large_face_landmark_dataset'
        self.width = width
        self.height = height
        self.train = train
        self.transform = transform
        self.num_of_keypoints = num_of_keypoints

        image_xml = None
        if self.train:
            image_xml = utils_dataset.get_imagexml(self.data_dir + "\\labels_ibug_300W.xml")
        else:
            image_xml = utils_dataset.get_imagexml(self.data_dir + "\\labels_ibug_300W_test.xml")

        data = utils_dataset.init_dataframe(self.num_of_keypoints)
        data = utils_dataset.get_data(image_xml, data)

        self.df = pd.DataFrame(data)

        file = self.df['file_path']
        self.x = list()
        for i in range(128):
            image = Image.open(self.data_dir + '\\' + file[i])
            self.x.append(np.asarray(image))

        self.y = list()
        for i in range(128):
            temp = list()
            for j in range(self.num_of_keypoints):
                temp.append(self.df.iloc[i][str(j)][0])
                temp.append(self.df.iloc[i][str(j)][1])
            temp = torch.Tensor(temp)
            self.y.append(temp)

        self.box = self.df['box']
        self.bbox_classes = ['person']

        self.keypoints_classes = [str(i) for i in range(self.num_of_keypoints)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        image, label, box = self.x[index], self.y[index], self.box[index]
        if self.transform is not None:
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

            for i in range(1, len(label), 2):
                label[i - 1] = label[i - 1] - x_min
                label[i] = label[i] - y_min
            last_transform = albumentations.Crop(x_min, y_min, x_max, y_max)
            image = last_transform(image = image)['image']
            box[0], box[1] = 0, 0
            if self.train:
                keypoints = label.reshape(-1, 2)
                image, box, keypoints = self.transform(
                    x = image,
                    box = box,
                    bbox_classes = self.bbox_classes,
                    keypoints = keypoints,
                    keypoints_classes = self.keypoints_classes
                )
                label = torch.Tensor(keypoints).reshape(-1)
            else:
                image, _, _ = self.transform(x = image)
        for i in range(1, len(label), 2):
            label[i - 1] = label[i - 1] / self.width - 0.5
            label[i] = label[i] / self.height - 0.5
        label = torch.Tensor(label)
        return image, label
    
    def get_raw_data(self, index: int):
        return self.x[index], self.y[index], self.box[index]
    
    def get_transformed_data(self, index: int):
        image, box, label = self.x[index], self.box[index], self.y[index]
        if self.transform is not None:
            if self.train:
                keypoints = label.reshape(-1, 2)
                image, box, keypoints = self.transform(
                    x = image,
                    box = box,
                    bbox_classes = self.bbox_classes,
                    keypoints = keypoints,
                    keypoints_classes = self.keypoints_classes)
            else:
                image, _, _ = self.transform(x = image)
        return image, box, keypoints

if __name__ == '__main__':
    import omegaconf
    import pyrootutils
    import hydra

    root = pyrootutils.setup_root(__file__, pythonpath= True)
    print(root)
    cfg = omegaconf.OmegaConf.load(
        root / "configs"/ "data"/ "filter_train_test" / "filter_train.yaml"
    )
    cfg.data_dir = str(root / "data")
    filter = hydra.utils.instantiate(cfg)
    filter.transform = utils_dataset.instantiate_transform("transform\\filter_transform\\train_transform.yaml")
    print(filter)
    x, y = filter[10]
    print(x.shape, y.shape)
    
    utils_dataset.draw_image_with_keypoints(
        x.numpy().transpose(1, 2, 0),
        y.reshape(-1, 2),
        width = filter.width,
        height = filter.height,
        normalize= True
    )
    print(y.shape)
    plt.show()
    #_ = FilterDataset()