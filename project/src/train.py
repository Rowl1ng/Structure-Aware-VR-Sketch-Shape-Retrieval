from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils
import os

log = utils.get_logger(__name__)

from glob import glob
def get_best_ckpt(callback):
    # import pdb
    # pdb.set_trace()
    checkpoint_files = glob(os.path.join(callback.dirpath, "{}*.ckpt".format(callback.monitor)))
    if len(checkpoint_files)>0:
        if callback.mode == 'min':
            checkpoint_file = sorted(checkpoint_files)[0]
        else:
            checkpoint_file = sorted(checkpoint_files)[-1]
        # log.info(f'Load best model: {checkpoint_file}')
    return checkpoint_file


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    config['model']['save_dir'] = os.path.join(config["work_dir"], 'logs/multifold', config["name"])

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    if config.get("resume_training"):
        # resume_training
        # checkpoint_file = os.path.join(config["work_dir"], 'logs/experiments', config["name"],  "checkpoints", "last.ckpt")
        checkpoint_file = os.path.join(config["work_dir"], 'logs/multifold', config["name"],  "checkpoints", "last.ckpt")

        if not os.path.exists(checkpoint_file):
            log.info("Training from scratch!")
            trainer: Trainer = hydra.utils.instantiate(config["trainer"], callbacks=callbacks, logger=logger)

        else:
            log.info("Resume training!")
            trainer: Trainer = hydra.utils.instantiate(config["trainer"], callbacks=callbacks, logger=logger,
                                                        resume_from_checkpoint=checkpoint_file)
    else:
        # Init lightning trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        # trainer.test()
        # Test multiple "best" checkpoints
        for _ckpt in range(len(trainer.checkpoint_callbacks)):
            log.info("Testing: monitor metric: {}".format(
                trainer.checkpoint_callbacks[_ckpt].monitor
            ))
            # ckpt_path = trainer.checkpoint_callbacks[_ckpt].best_model_path
            ckpt_path = get_best_ckpt(trainer.checkpoint_callbacks[_ckpt])
            log.info("Best checkpoint path: {}".format(ckpt_path))
            config.model.test_ckpt = trainer.checkpoint_callbacks[_ckpt].monitor
            # test the model using current checkpoint
            trainer.test(ckpt_path=ckpt_path)
        
        # TODO: uncomment this to evluate with the last ckpt
        ckpt_path = os.path.join(trainer.checkpoint_callbacks[0].dirpath, 'last.ckpt')
        log.info("Last checkpoint path: {}".format(ckpt_path))
        config.model.test_ckpt = 'last'
        # test the model using current checkpoint
        trainer.test(ckpt_path=ckpt_path)
       

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]