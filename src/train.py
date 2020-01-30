import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import date

from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, outcome_cohort_csv
from .datasets import NicuDataset

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))


def train(cfg, writer):
    # TODO: Refactor for hyperparameters
    # TODO: Tensorboard write for loss and accuracy

    transforms = None
    trainset = NicuDataset(cfg.TRAIN_DIR + outcome_cohort_csv, cfg.VOLUME_DIR, sampling_strategy='front',
                           transform=transforms)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    model = nn.LSTM(len(MEASUREMENT_SOURCE_VALUE_USES) + 1, 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    epochs = 5
    running_loss = 0.0

    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            # TODO inputs type: DataFrame to tensor
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


def main(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))
    print("Train function runs")

    # resample_and_save_by_user(cfg, writer)
    # subdivide(cfg, writer)
    train(cfg, writer)
