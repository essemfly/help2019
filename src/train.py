import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from tensorboardX import SummaryWriter
from datetime import date
from tqdm import tqdm, trange

from .config import LocalConfig, ProdConfig
from .constants import outcome_cohort_csv, hyperparams, model_config
from .datasets.measurement import MeasurementDataset
from .datasets.hourly_sampled import HourlySampledDataset

from .models import NicuModel, FocalLoss, ConvLstmLinear, ConvConvConv
from .optimization import BertAdam
from .preprocess.sample_by_hour import measure11_dfs, convert_features_to_dataset, measurement_preprocess

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
    reverse_pad = True

    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))

    transforms = None
    trainset = HourlySampledDataset(cfg.get_csv_path(outcome_cohort_csv, mode), max_seq_length=max_seq_length,
                                    transform=transforms, reverse_pad=reverse_pad)
    dfs = measure11_dfs(cfg, mode)
    trainset.fill_dfs(dfs)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)
    del dfs
    full_x = []
    full_x_len = []
    full_labels = []
    for i, (x, x_len, labels) in enumerate(tqdm(trainloader, desc="Preparing")):   
        full_x.append(x)
        full_x_len.append(x_len)
        full_labels.append(labels)
        
    full_x = torch.cat(full_x, dim=0)
    full_x_len = torch.cat(full_x_len, dim=0)
    full_labels = torch.cat(full_labels, dim=0)

    target = trainset.o_df['LABEL']
    class_count = np.unique(target, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[target]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=False, pin_memory=True)
    
    trainset = TensorDataset(full_x, full_x_len, full_labels)
    #trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)
    
    model = ConvConvConv(prior_prob=hyperparams['prior_prob'])            
    #model = NicuModel(device=device, prior_prob=hyperparams['prior_prob'])
    #model = ConvLstmLinear(device=device, prior_prob=hyperparams['prior_prob'])
    print(hyperparams)
    print(model_config)
    print(model)
    model.to(device=device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    criterion = FocalLoss(gamma=hyperparams['gamma'], alpha=hyperparams['alpha'])

    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps = len(trainloader) * epochs
    optimizer = BertAdam(model.parameters(), lr=lr, warmup=hyperparams['warmup_proportion'], t_total=num_steps)

    for epoch in trange(epochs, desc="Epoch"):
        running_loss = 0.0
        for idx, data in enumerate(tqdm(trainloader, desc="Iteration")):
            data = tuple(t.to(device) for t in data)
            x, x_len, labels = data
            if reverse_pad:
                outputs = model(x)
                loss = criterion(outputs, labels)
            else:
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
    torch.save(model_to_save, f'{cfg.VOLUME_DIR}/convconv_epoch{hyperparams["epochs"]}.ckpt')


def main_train(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    print("Train function runs")
    '''
    measurement_preprocess(cfg, 'train')
    measurement_preprocess(cfg, 'test')
    convert_features_to_dataset(cfg, 'train')
    convert_features_to_dataset(cfg, 'test')
    '''
    train(cfg)
