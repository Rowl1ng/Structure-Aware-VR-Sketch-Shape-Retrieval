import torch

# hydra imports
from omegaconf import DictConfig
import hydra
from pytorch_lightning.loggers import LightningLoggerBase

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
import os
# normal imports
from typing import List

# template utils imports
from src.utils import template_utils as utils


def train(config):
    # Set global PyTorch seed
    if "seeds" in config and "pytorch_seed" in config["seeds"]:
        torch.manual_seed(seed=config["seeds"]["pytorch_seed"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule: LightningDataModule = hydra.utils.instantiate(config["datamodule"])

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = [
        hydra.utils.instantiate(callback_conf)
        for callback_name, callback_conf in config["callbacks"].items()
    ] if "callbacks" in config else []

    # Init PyTorch Lightning loggers ⚡
    loggers: List[LightningLoggerBase] = [
        hydra.utils.instantiate(logger_conf)
        for logger_name, logger_conf in config["logger"].items()
        if "_target_" in logger_conf   # ignore logger conf if there's no target
    ] if "logger" in config else []

    config['model']['save_dir'] = os.path.join(config["original_work_dir"], 'logs/experiments', config["name"], 'vis')

    if config['mode'] == 'train':
        model: LightningModule = hydra.utils.instantiate(config["model"])
        checkpoint_file = os.path.join(config["original_work_dir"], 'logs/experiments', config["name"],  "checkpoints", "last.ckpt")
        if not os.path.exists(checkpoint_file):
            print("Training from scratch!")
            trainer: Trainer = hydra.utils.instantiate(config["trainer"], callbacks=callbacks, logger=loggers)

        else:
            print("Resume training!")
            trainer: Trainer = hydra.utils.instantiate(config["trainer"], callbacks=callbacks, logger=loggers,
                                                        resume_from_checkpoint=checkpoint_file)
        # Magic
        utils.extras(config, model, datamodule, callbacks, loggers, trainer)
    
        # Train
        trainer.fit(model=model, datamodule=datamodule)
        # Test
        trainer.test(ckpt_path=config["test_ckpt"]) # best

    else:
        # Evaluation on test set
        from glob import glob
        from pytorch_lightning.utilities.cloud_io import load as pl_load

        if config["test_ckpt"] == 'best':
            checkpoint_files = glob(
                os.path.join(config["original_work_dir"], 'logs/experiments', config["name"], "checkpoints", "val*.ckpt"))
            if len(checkpoint_files)>0:
                if 'val_loss' in os.path.basename(checkpoint_files[0]):
                    checkpoint_file = sorted(checkpoint_files)[0]
                else:
                    checkpoint_file = sorted(checkpoint_files)[-1]
                utils.log_string('Load best model: {}'.format(checkpoint_file))
        else:
            checkpoint_file = os.path.join(config["original_work_dir"], 'logs/experiments', config["name"], "checkpoints", "last.ckpt")
            utils.log_string('Load last model: {}'.format(checkpoint_file))
        
        model: LightningModule = hydra.utils.instantiate(config["model"], checkpoint_path=checkpoint_file)

        # if os.path.exists(checkpoint_file):
        #     ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  
        # else:
        #     print("No ckpt is saved!")
        #     return 0

        # model = model.cuda()
        # model.load_state_dict(ckpt['state_dict'])

        if config['mode'] == 'vis':
            visual(config, model)
        else:
            print("Evaluation on test set!")
            trainer: Trainer = hydra.utils.instantiate(config["trainer"], callbacks=callbacks, logger=loggers,
                                                        resume_from_checkpoint=checkpoint_file)
            trainer.test(model, datamodule=datamodule)
            # model.test(config, model)

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # import sys
    # print(sys.argv)
    # for item in sys.argv:
    #     if '+experiment' in item:
    #         config.exp_name = item.split('=')[1]
    #         config
    #         break

    utils.print_config(config)
    train(config)


if __name__ == "__main__":
    main()
