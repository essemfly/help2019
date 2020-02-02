import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import date
from tqdm import tqdm, trange

from .config import LocalConfig, ProdConfig
from .constants import CONDITION_SOURCE_VALUE_USES, MEASUREMENT_SOURCE_VALUE_USES, outcome_cohort_csv
from .datasets.combined import CombinedDataset
from .models import LSTM, FocalLoss

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))


def train(cfg):
    # TODO: Refactor for hyperparameters
    mode = 'train'
    batch_size = 128
    lr = 0.001
    weight_decay = 0
    input_size = len(CONDITION_SOURCE_VALUE_USES) + len(MEASUREMENT_SOURCE_VALUE_USES)
    hidden_size = 512
    sampling_strategy = 'front'
    max_seq_length = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    num_workers = 6 * n_gpu
    num_labels = 1
    num_layers = 1
    epochs = 30

    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))

    transforms = None
    trainset = CombinedDataset(cfg.get_csv_path(outcome_cohort_csv, mode), max_seq_length=max_seq_length,
                               transform=transforms)
    dfs, births = cfg.load_combined_dfs_births(mode, sampling_strategy)
    trainset.fill_dfs_and_births(dfs, births)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    model = LSTM(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size,
                 num_labels=num_labels, device=device, num_layers=num_layers)
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=5.0, alpha=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in trange(epochs, desc="Epoch"):
        running_loss = 0.0
        for idx, data in enumerate(tqdm(trainloader, desc="Iteration")):
            data = tuple(t.to(device) for t in data)
            x, x_len, labels = data
            actual_batch_size = x.size()
            if actual_batch_size[0] == batch_size:
                outputs = model(x, x_len)
                loss = criterion(outputs, labels)
            else:
                x_padding = torch.zeros(
                    (batch_size - actual_batch_size[0], actual_batch_size[1], actual_batch_size[2])).to(device)
                x_len_padding = torch.ones(batch_size - actual_batch_size[0], dtype=torch.long).to(device)
                outputs = model(torch.cat((x, x_padding)), torch.cat((x_len, x_len_padding)))
                loss = criterion(outputs[:actual_batch_size[0]], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar('Loss', running_loss / len(trainloader.dataset), epoch + 1)
        model_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(model_to_save, f'{cfg.VOLUME_DIR}/200203_combined_epoch{epoch + 1}_base.ckpt')


def main_train(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    print("Train function runs")

    from .preprocess.combined_divide import combined_preprocess, combined_preprocess_from_pkl
    # combined_preprocess(cfg, 'train', 'front')
    # combined_preprocess(cfg, 'train', 'average')
    # combined_preprocess(cfg, 'test', 'front')
    # combined_preprocess(cfg, 'test', 'average')

    combined_preprocess_from_pkl(cfg, 'train', 'front')
    combined_preprocess_from_pkl(cfg, 'train', 'average')
    combined_preprocess_from_pkl(cfg, 'test', 'front')
    combined_preprocess_from_pkl(cfg, 'test', 'average')

    # from .preprocess.measure_divide import measurement_preprocess
    # measurement_preprocess(cfg, 'train', 'front')
    # measurement_preprocess(cfg, 'train', 'average')
    # measurement_preprocess(cfg, 'test', 'front')
    # measurement_preprocess(cfg, 'test', 'average')

    from .preprocess.condition_divide import condition_preprocess
    # condition_preprocess(cfg, 'train')
    # condition_preprocess(cfg, 'test')

    train(cfg)
