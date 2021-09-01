import csv
import os
import time

import torch
import torch.nn as nn
import torch.distributed
import torch.utils.data
from tqdm import tqdm
import utills


def evaluate(model:nn.Module,
             vall_data_loader: torch.utils.data.DataLoader,
             criterion: nn.Module,
             amp_enable: bool,
             ddp_enable: bool,
             device: torch.device):
    model.eval()

    if ddp_enable:
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 0

    infernce_time = torch.zeros(1, device=device)
    val_loss = torch.zeros(4, device = device)
