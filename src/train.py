import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter
from datetime import date
from tqdm import tqdm, trange

from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, outcome_cohort_csv, hyperparams
from .datasets.measurement import MeasurementDataset
from .preprocess.measure_divide import measurement_preprocess
from .models_new import NicuModel, FocalLoss

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))


def train(cfg):
    # TODO: Refactor for hyperparameters
    mode = 'train'
    batch_size = hyperparams['batch_size']
    lr = hyperparams['lr']
    weight_decay = hyperparams['weight_decay']
    sampling_strategy = hyperparams['sampling_strategy']
    max_seq_length = hyperparams['max_seq_len']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    num_workers = 8 * n_gpu
    epochs = hyperparams['epochs']

    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))

    transforms = None
    trainset = MeasurementDataset(cfg.get_csv_path(outcome_cohort_csv, mode), max_seq_length=max_seq_length,
                                  transform=transforms)
    dfs, births = cfg.load_person_dfs_births(mode, sampling_strategy)
    trainset.fill_people_dfs_and_births(dfs, births)
    '''
    target = trainset.o_df['LABEL']
    class_count = np.unique(target, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[target]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=False)
    '''
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)
    model = NicuModel(device=device, prior_prob=hyperparams['prior_prob'])
    model.to(device=device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=hyperparams['gamma'], alpha=hyperparams['alpha'])
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
        torch.save(model_to_save, f'{cfg.VOLUME_DIR}/200204_epoch{epoch + 1}_attn.ckpt')


def main_train(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    print("Train function runs")
    '''
    measurement_preprocess(cfg, 'train', 'front')
    measurement_preprocess(cfg, 'train', 'average')
    measurement_preprocess(cfg, 'test', 'front')
    measurement_preprocess(cfg, 'test', 'average')
    '''
    train(cfg)
