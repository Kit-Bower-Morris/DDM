from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from trainer import TrainerConfig, Trainer, TrainerArgs

from model.fastspeech2 import FastSpeech2
from dataset import Dataset
from config import Fastspeech2Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    train_args = TrainerArgs()
    config = Fastspeech2Config()

    # init the model from config
    model = FastSpeech2(config).to(device)


    # init the trainer and 🚀
    trainer = Trainer(
        train_args,
        config,
        config.output_path,
        model=model,
        train_samples=model.get_train_data_loader(config, None, None, None, None),
        eval_samples=model.get_eval_data_loader(config, None, None, None, None),
        parse_command_line_args=True,
    )
    trainer.fit()


if __name__ == "__main__":
    main()