import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import date

from .config import LocalConfig, ProdConfig
from .subdivide import subdivide
from .preprocessing import preprocess
from .constants import MEASUREMENT_SOURCE_VALUE_USES, outcome_cohort_csv, person_csv
from .datasets import NicuDataset
from .models import LSTM

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))


def train(cfg, writer):
    # TODO: Refactor for hyperparameters
    # TODO: Tensorboard write for accuracy

    batch_size = 64
    lr = 0.01
    weight_decay = 0
    input_size = len(MEASUREMENT_SOURCE_VALUE_USES) + 1
    hidden_size = 128
    sampling_strategy = 'front'
    max_seq_length = 4096
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    num_workers = 4 * n_gpu
    num_labels = 2
    epochs = 5

    transforms = None
    trainset = NicuDataset(cfg.TRAIN_DIR + outcome_cohort_csv, cfg.TRAIN_DIR + person_csv, cfg.VOLUME_DIR,
                           sampling_strategy=sampling_strategy, max_seq_length=max_seq_length, transform=transforms)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    model = LSTM(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size,
                 num_labels=num_labels, device=device)
    model.to(device)
    # model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        running_loss = 0.0
        for idx, data in enumerate(trainloader):
            data = tuple(t.to(device) for t in data)
            x, x_len, labels = data
            outputs = model(x, x_len)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar('Loss', running_loss / len(trainloader.dataset), epoch)
        torch.save(model.state_dict(), f'{cfg.VOLUME_DIR}/epoch{epoch + 1}.ckpt')


def main(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))
    print("Train function runs")

    # resample_and_save_by_user(cfg, writer)
    # subdivide(cfg, writer)
    # train(cfg, writer)
    preprocess(cfg, 'front')
    preprocess(cfg, 'average')
