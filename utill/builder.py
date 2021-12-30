import torch
import torch.nn as nn
import torchvision
import yaml
from yaml import *
from model.od.Fcos import FCOS
from model.od.proposed import HalfInvertedStageFCOS


def load_cfg(file: str = '../config.yaml') -> dict:
    with open(file) as f:
        cfg = yaml.safe_load(f)
    return cfg


class build:
    def __init__(self, cfg: dict):
        self.amp_opt = cfg['amp_enabled']
        self.ddp_opt = cfg['ddp_enabled']
        self.swa_opt = cfg['swa_enabled']
        self.dataset = cfg['dataset']
        self.model_name = cfg['model_name']
        self.backbon_ch = cfg[self.model_name]['feature_extract_ch']
        self.feature_ch = cfg[self.model_name]['feature']
        self.num_of_class = cfg[self.model_name][self.dataset]['numberofclass']
        self.epoch = cfg[self.model_name][self.dataset]['EPOCH']
        self.batch_size = cfg[self.model_name][self.dataset]['BATCH_SIZE']
        self.lr = cfg[self.model_name][self.dataset]['LR_INIT']
        self.momentum = cfg[self.model_name][self.dataset]['MOMENTUM']
        self.weight_decay = cfg[self.model_name][self.dataset]['WEIGHTDECAY']
        self.model = self.build_model(self.backbon_ch, self.feature_ch, self.num_of_class)


    def build_model(self, backbone_last, feature, number_of_class) -> torch.Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.Tensor
        if self.model_name == 'fcos':
            model = FCOS(backbone_last, number_of_class, feature).to(device)
        elif self.model_name == 'hisfcos':
            model = HalfInvertedStageFCOS(backbone_last, number_of_class, feature).to(device)
        else:
            print(f'{self.model_name} is not able!')
        return model







if __name__ == '__main__':

    build(load_cfg())

