import os
import torch
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
    # TODO: Tensorboard write for accuracy

    batch_size = 64
    lr = 0.01
    weight_decay = 0
    hidden_size = 128
    sampling_strategy = 'front'
    num_workers = 2
    epochs = 3

    transforms = None
    trainset = NicuDataset(cfg.TRAIN_DIR + outcome_cohort_csv, cfg.VOLUME_DIR, sampling_strategy=sampling_strategy,
                           transform=transforms)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = LSTM(input_size=len(MEASUREMENT_SOURCE_VALUE_USES) + 1, hidden_size=hidden_size, batch_size=batch_size,
                 num_labels=1)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        for idx, data in enumerate(trainloader):
            x, x_len, labels = data
            outputs = model(x, x_len)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 9:
                writer.add_scalar('Loss', loss.item(), epoch * len(trainloader) + idx)
        torch.save(model.state_dict(), f'{cfg.VOLUME_DIR}/epoch{epoch + 1}.ckpt')


def main(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))
    print("Train function runs")

    # resample_and_save_by_user(cfg, writer)
    # subdivide(cfg, writer)
    train(cfg, writer)
