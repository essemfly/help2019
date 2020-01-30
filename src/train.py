import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import date

from .config import LocalConfig, ProdConfig
from .subdivide import subdivide
from .constants import MEASUREMENT_SOURCE_VALUE_USES, outcome_cohort_csv
from .datasets import NicuDataset
from .models import LSTM

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))


def train(cfg, writer):
    # TODO: Refactor for hyperparameters
    # TODO: Tensorboard write for loss and accuracy

    batch_size = 64
    lr = 0.001
    weight_decay = 0
    hidden_size = 128
    sampling_strategy = 'front'
    num_workers = 2
    epochs = 5

    transforms = None
    trainset = NicuDataset(cfg.TRAIN_DIR + outcome_cohort_csv, cfg.VOLUME_DIR, sampling_strategy=sampling_strategy,
                           transform=transforms)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = LSTM(input_size=len(MEASUREMENT_SOURCE_VALUE_USES) + 1, hidden_size=hidden_size, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    running_loss = 0.0

    for epoch in range(epochs):
        for idx, data in enumerate(trainloader):
            x, x_len, labels = data
            optimizer.zero_grad()
            outputs = model(x, x_len)
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
