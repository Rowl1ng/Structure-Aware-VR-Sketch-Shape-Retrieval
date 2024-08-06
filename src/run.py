# import dotenv
import hydra
from omegaconf import DictConfig
import rootutils
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# dotenv.load_dotenv(override=True)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base="1.3", config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934


    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if config.get("inference")==True:
    #     test(config)
        # config.test_ckpt = 'last'
        # test(config)
        # config.test_ckpt = 'best'
        # config.datamodule.sketch_dir = 'sketch_21_test'
        from src.test import test
        return test(config)
    # Train model
    # train(config)
    from src.train import train
    return train(config)

if __name__ == "__main__":
    # sys.argv.append('hydra.run.dir={}/logs/experiments/'.format())
    # sys.argv.append('hydra.run.dir=c_out/cached_loss')
    main()