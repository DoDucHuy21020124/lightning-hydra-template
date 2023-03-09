from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from components.filter_dataset import FilterDataset
from components.transform import ImageTransform
from components import utils_dataset

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
        train_val_test_split: Tuple[int, int, int] = (64, 64, 50),
        batch_size: int = 5,
        num_workers: int = 8,
        pin_memory: bool = False,
        data_train: FilterDataset = None,
        data_test: FilterDataset = None,
        train_transform: ImageTransform = None,
        val_transform: ImageTransform = None,
        test_transform: ImageTransform = None

    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.data_train: Optional[Dataset] = data_train
        self.data_test: Optional[Dataset] = data_test

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
        self.data_train, self.data_val = random_split(
            dataset=self.data_train,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.data_val.train = False

        self.data_train.transform = self.train_transform
        self.data_val.transform = self.val_transform
        self.data_test.transform = self.test_transform

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

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "filter.yaml")
    cfg.data_dir = str(root / "data")
    print(cfg.data_dir)
    filter_data = hydra.utils.instantiate(cfg)
    print(filter_data)
    filter_data.prepare_data()
    filter_data.setup()
    train_loader = filter_data.train_dataloader()
    batch_iterator = iter(train_loader)
    images, labels = next(batch_iterator)
    #print(images.size())
    #print('labels:', labels)
    utils_dataset.draw_batch_image(images, labels)
    
