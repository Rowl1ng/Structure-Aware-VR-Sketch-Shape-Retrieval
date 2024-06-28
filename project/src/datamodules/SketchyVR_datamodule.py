import importlib
from torch.utils.data import DataLoader, ConcatDataset, random_split
from pytorch_lightning import LightningDataModule
from torchvision.transforms import transforms
from src.datasets.SketchyVRLoader import SketchyVR, Shapes, SketchyVR_single, SketchyVR_single_multifold
import src.datamodules.data_utils as d_utils
import numpy as np
import os

class PairDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.data_train = None
        self.data_val = None
        self.data_test = None
        the_module = importlib.import_module('src.datasets.SketchyVRLoader')
        self.dataset_loader = getattr(the_module, self.kwargs['dataset_loader'])
        self.test_dataset_loader = getattr(the_module, self.kwargs['test_dataset_loader'])

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        self.data_train = self.dataset_loader(self.kwargs, mode='train', source_type=self.kwargs['source_type'])
        self.data_val = self.dataset_loader(self.kwargs, mode='val', source_type=self.kwargs['source_type'])
        self.test_dataset_loader = self.test_dataset_loader(self.kwargs, mode='test', source_type='sketch')
    
    def train_dataloader(self):
        dataloader = DataLoader(dataset=self.data_train, batch_size=self.kwargs['batch_size'], \
                        shuffle=True, drop_last=True, collate_fn=self.data_train.collate, \
                        num_workers=self.kwargs['num_workers'], pin_memory=self.kwargs['pin_memory'], \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

        return dataloader
    def val_dataloader(self):
        dataloader = DataLoader(dataset=self.data_val, batch_size=self.kwargs['batch_size'], \
                        shuffle=False, drop_last=False, collate_fn=self.data_val.collate, \
                        num_workers=self.kwargs['num_workers'], pin_memory=self.kwargs['pin_memory'], \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(dataset=self.data_test, batch_size=self.kwargs['batch_size'], \
                        shuffle=False, drop_last=False, collate_fn=self.data_test.collate, \
                        num_workers=self.kwargs['num_workers'], pin_memory=self.kwargs['pin_memory'], \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

        return dataloader

class SketchyVRDataModule(PairDataModule):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.data_train = self.dataset_loader(self.kwargs, mode='train', cache=False)
        self.data_val = self.dataset_loader(self.kwargs, mode='val', cache=False)

class SketchyVRDataModule_multifold(SketchyVRDataModule):
    def setup(self, stage=None):
        self.kwargs['train_list'] = self.kwargs['train_list'].format(self.kwargs['fold_id'])
        self.kwargs['train_shape_list'] = self.kwargs['train_shape_list'].format(self.kwargs['fold_id'])
        self.data_train = self.dataset_loader(self.kwargs, mode='train', source_type=self.kwargs['source_type'])
        self.data_val = SketchyVR_single_multifold(self.kwargs, type='sketch', fold=self.kwargs['fold_id'])
        self.data_test = SketchyVR_single(self.kwargs, mode='test', type='sketch', cache=True, sketch_dir=self.kwargs['test_data'])

class SketchyVRDataModule_category(SketchyVRDataModule):
    def setup(self, stage=None):
        from src.datasets.SketchyVRLoader import SketchyVR_original_category, SketchyVR_original_category_single
        self.data_train = SketchyVR_original_category(self.kwargs, mode='train', source_type=self.kwargs['source_type'])
        self.data_val = SketchyVR_original_category_single(self.kwargs, mode='val', type='sketch', cache=True)
        self.data_test = SketchyVR_original_category_single(self.kwargs, mode='test', type='sketch', cache=True)

# hydra imports
from omegaconf import DictConfig
import hydra
@hydra.main(config_path="../../configs/", config_name="config.yaml")
def main(config: DictConfig):
    # dset = PairDataModule(config.datamodule)
    # Init PyTorch Lightning datamodule ⚡
    datamodule: LightningDataModule = hydra.utils.instantiate(config["datamodule"])

    datamodule.setup()

if __name__ == "__main__":
    main()