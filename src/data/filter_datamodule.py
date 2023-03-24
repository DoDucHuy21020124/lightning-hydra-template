from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.filter_dataset import FilterDataset
from src.data.components.transform import ImageTransform
from omegaconf import DictConfig
import pyrootutils
import hydra
import omegaconf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import albumentations

class FilterDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (128, 64, 64),
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = False,
        width: int = 224,
        height: int = 224,
        data_train: FilterDataset = None,
        data_test: FilterDataset = None,
        train_transform: albumentations.Compose = None,
        val_transform: albumentations.Compose = None,
        test_transform: albumentations.Compose = None
    ):
        super().__init__()
        # def instantiate_data(file_path: str):
        #     root = pyrootutils.setup_root(__file__, pythonpath= True)
        #     cfg = omegaconf.OmegaConf.load(root / "configs"/ "data"/ file_path)
        #     cfg.data_dir = str(root / "data")
        #     data = hydra.utils.instantiate(cfg)
        #     return data

        # def instantiate_transform(file_path: str):
        #     root = pyrootutils.setup_root(__file__, pythonpath= True)
        #     cfg = omegaconf.OmegaConf.load(root / "configs"/ "data"/ file_path)
        #     data = hydra.utils.instantiate(cfg)
        #     return data

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.train_transform = util.instantiate_transform(train_transform)
        # self.val_transform = util.instantiate_transform(val_transform)
        # self.test_transform = util.instantiate_transform(test_transform)

        # self.train_set: Optional[FilterDataset] = util.instantiate_data(data_train)

        self.data_train = None
        self.data_val = None
        #self.data_test: Optional[FilterDataset] = instantiate_data(data_test)
        self.data_test = None

    @property
    def num_classes(self):
        return 136

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        #if not self.data_train and not self.data_val and not self.data_test:
        # self.data_train, self.data_val = random_split(
        #     dataset=self.data_train,
        #     lengths=self.hparams.train_val_test_split,
        #     generator=torch.Generator().manual_seed(42),
        # )

        # train, val = train_test_split(
        #     [i for i in range(len(self.train_set))],
        #     train_size = self.hparams.train_val_test_split[0],
        #     test_size = self.hparams.train_val_test_split[1],
        #     random_state= 42,
        #     shuffle = True
        # )

        # self.train_set.train = True
        # self.train_set.transform = self.val_transform
        # self.data_val = torch.utils.data.Subset(self.train_set, val)
        # print(self.data_val.transform)

        # self.train_set.train = True
        # self.train_set.transform = self.train_transform
        # self.data_train = torch.utils.data.Subset(self.train_set, train)
        # self.data_train.transform = self.train_transform

        # for x, y in self.data_val:
        #     util.draw_image_with_keypoints(x.numpy().transpose(1, 2, 0), y.reshape(-1, 2), 224, 224, True)
        #     plt.show()
        

        #self.data_test.transform = self.test_transform

        if not self.data_train and not self.data_val and not self.data_test:
            data_train = self.hparams.data_train(data_dir = self.hparams.data_dir)
            #data_train = self.hparams.data_train
            #data_test = self.hparams.data_test
            data_train, data_val = random_split(
                dataset=data_train,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train = ImageTransform(
                data_train, self.hparams.width, self.hparams.height, self.hparams.train_transform
            )
            self.data_val = ImageTransform(
                data_val, self.hparams.width, self.hparams.height, self.hparams.val_transform
            )
            # self.data_test = ImageTransform(
            #     data_test, self.hparams.width, self.hparams.height, self.hparams.test_transform
            # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    import matplotlib.pyplot as plt
    #from components import utils_dataset

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    config_path = str(root / 'configs'/ 'data')
    data_dir = root / 'data/ibug_300W_large_face_landmark_dataset/'
    @hydra.main(config_path=config_path, config_name = 'filter.yaml')
    def main(cfg: DictConfig):
        cfg.data_dir = data_dir
        datamodule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        # x, y = datamodule.data_train[3]
        # print(x)
        # print(y)
        dataloader = datamodule.train_dataloader()
        iterbatch = iter(dataloader)
        x, y = next(iterbatch)
        ImageTransform.draw_batch_image(
            images = x,
            labels = y,
            width = x.shape[2],
            height = x.shape[3],
            normalize = True
        )
        plt.show()
    main()

    # root = pyrootutils.setup_root(__file__, pythonpath=True)
    # cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "filter.yaml")
    # cfg.data_dir = str(root / "data")
    # print(cfg.data_dir)
    # filter_data = hydra.utils.instantiate(cfg)
    # filter_data.prepare_data()
    # filter_data.setup()
    # print(filter_data.data_train.__len__())

    # x, y = None, None
    # i = 0
    # try:
    #     while i < len(filter_data.data_train):
    #         x, y = None, None
    #         x, y = filter_data.data_train[i]
    #         i += 1
    # except:
    #     print(i)

    # x, y = filter_data.data_train[3]
    # util.draw_image_with_keypoints(x.numpy().transpose(1, 2, 0), util.change_label_to_keypoints(y), 224, 224, True)
    # plt.show()
    # print(type(x), type(y))
    # print(x.shape, len(y))
    # train_loader = filter_data.train_dataloader()
    # print(train_loader)
    # batch_iterator = iter(train_loader)
    # print(batch_iterator)
    # images, labels = next(batch_iterator)
    # print(images.size(), labels.size())
    # print('labels:', labels)
    # util.draw_batch_image(images, labels, 224, 224, True)

    # val_loader = filter_data.val_dataloader()
    # print(val_loader)
    # batch_iterator = iter(val_loader)
    # print(batch_iterator)
    # images, labels = next(batch_iterator)
    # print(images.size(), labels.size())
    # print('labels:', labels)
    # util.draw_batch_image(images, labels, 224, 224, True)
    
