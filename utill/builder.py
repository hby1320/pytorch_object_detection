import torch
import torch.nn as nn
import torchvision
import yaml
from yaml import *


def load_cfg(file:str = '../config.yaml'
             ) -> dict:
    with open(file) as f:
        cfg = yaml.safe_load(f)

    cfg['amp_enabled'] =
    cfg['amp_enabled']


class build:
    def __init__(self, cfg: dict):
        self.cfg = cfg


if __name__ == '__main__':
