from pytorch_lightning import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os

def remove_existing_file(fn):
    if os.path.exists(fn):
        os.remove(fn)

class ExampleCallback(Callback):
    def __init__(self):
        pass

    def on_init_start(self, trainer):
        print('Starting to initialize trainer!')

    def on_init_end(self, trainer):
        print('Trainer is initialized now.')

    def on_train_end(self, trainer, pl_module):
        print('Do something when training ends.')

class SelfModelCheckpoint(ModelCheckpoint):
    def _do_save(self, trainer: 'pl.Trainer', filepath: str) -> None:
        # in debugging, track when we save checkpoints
        trainer.dev_debugger.track_checkpointing_history(filepath)

        # make paths
        if trainer.is_global_zero:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

        # delegate the saving to the trainer
        remove_existing_file(filepath)
        trainer.save_checkpoint(filepath, self.save_weights_only)

from pytorch_lightning.callbacks import BaseFinetuning
class KP_FreezeUnfreeze(BaseFinetuning):

    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing ``feature_extractor``
        self.freeze(pl_module.KP_net, train_bn=False)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.make_trainable(pl_module.KP_net)

class UnfreezeModelCallback(Callback):
    """
    Unfreeze model after a few epochs.
    It currently unfreezes every possible parameter in model, probably shouldn't work that way...
    """
    def __init__(self, wait_epochs=5):
        self.wait_epochs = wait_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.wait_epochs:
            for param in pl_module.model.model.parameters():
                param.requires_grad = True
