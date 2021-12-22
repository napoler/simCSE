"""
v2
训练核心入口

"""
from pytorch_lightning.utilities.cli import LightningCLI

from tkitSimcse.Simcse import SimCSE

# os.environ['TOKENIZERS_PARALLELISM'] = "true"


if __name__ == '__main__':
    # freeze_support()
    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html

    cli = LightningCLI(SimCSE, save_config_overwrite=True)
