import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter
from datetime import date
from tqdm import tqdm, trange


from .config import LocalConfig, ProdConfig
from .constants import CONDITION_SOURCE_VALUE_USES, MEASUREMENT_SOURCE_VALUE_USES, outcome_cohort_csv, hyperparams
from .datasets.for_attn import AttentionDataset
from .models import NicuModel, FocalLoss

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))

def train(cfg):
    # TODO: Refactor for hyperparameters
    mode = 'train'
    batch_size = hyperparams['batch_size']
    lr = hyperparams['lr']
    weight_decay = hyperparams['weight_decay']
    sampling_strategy = hyperparams['sampling_strategy']
    max_seq_length = hyperparams['max_seq_length']
    epochs = hyperparams['epochs']
    gamma = hyperparams['gamma']
    alpha = hyperparams['alpha']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    num_workers = 6 * n_gpu
    
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))

    transforms = None
    trainset = AttentionDataset(cfg.get_csv_path(outcome_cohort_csv, mode), max_seq_length=max_seq_length, transform=transforms)
    dfs, births = cfg.load_combined_dfs_births(mode, sampling_strategy)
    trainset.fill_dfs_and_births(dfs, births)
    
    target = trainset.o_df['LABEL']
    class_count = np.unique(target, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[target]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=False)
    #trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    
    model = NicuModel()
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=0.0, alpha=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in trange(epochs, desc="Epoch"):
        running_loss = 0.0
        for idx, data in enumerate(tqdm(trainloader, desc="Iteration")):
            data = tuple(t.to(device) for t in data)
            times, measures, conditions, labels = data
            outputs = model(times, measures, conditions)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        writer.add_scalar('Loss', running_loss / len(trainloader.dataset), epoch + 1)
        model_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(model_to_save, f'{cfg.VOLUME_DIR}/200203_attn_epoch{epoch + 1}_base.ckpt')


def main_train(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    print("Train function runs")

    #from .preprocess.combined_divide import combined_preprocess, combined_preprocess_from_pkl
    #combined_preprocess(cfg, 'train', 'front')
    #combined_preprocess(cfg, 'train', 'average')
    #combined_preprocess(cfg, 'test', 'front')
    #combined_preprocess(cfg, 'test', 'average')

    #combined_preprocess_from_pkl(cfg, 'train', 'front')
    #combined_preprocess_from_pkl(cfg, 'train', 'average')
    #combined_preprocess_from_pkl(cfg, 'test', 'front')
    #combined_preprocess_from_pkl(cfg, 'test', 'average')

    # from .preprocess.measure_divide import measurement_preprocess
    # measurement_preprocess(cfg, 'train', 'front')
    # measurement_preprocess(cfg, 'train', 'average')
    # measurement_preprocess(cfg, 'test', 'front')
    # measurement_preprocess(cfg, 'test', 'average')

    #from .preprocess.condition_divide import condition_preprocess
    # condition_preprocess(cfg, 'train')
    # condition_preprocess(cfg, 'test')

    train(cfg)
