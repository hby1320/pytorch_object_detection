import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import model.od as od
from utill.utills import load_config, model_info
import dataset


class Builder:
    def __init__(self, cfg,):
        self.config = cfg
        self.model = cfg['model']['name']  # model_name

    # def dataset_build(self):
        

    def model_build(self,) -> nn.Module:
        cfg_model = self.config[self.model]
        Cob = cfg_model['CannelofBackbone']
        NoC = self.config['dataset_setting']['class_num']
        channel = cfg_model['channel']
        if self.model == 'FCOS':
            model = od.Fcos.FCOS(Cob, NoC, channel)
        elif self.model == 'HISFCOS':
            model = od.proposed.HalfInvertedStageFCOS(Cob, NoC, channel)
        elif self.model == 'SSD300':
            model = od.ssd.SSD300(NoC)
        return model

    def opt_build(self, model: nn.Module) -> torch.optim.Optimizer:
        cfg_optimizer = self.config[self.model]['optimizer']

        if cfg_optimizer['name'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), cfg_optimizer['lr'], cfg_optimizer['momentum'],
                                        cfg_optimizer['weight_decay'])
        elif cfg_optimizer['name'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), cfg_optimizer['lr'], cfg_optimizer['weight_decay'])
        elif cfg_optimizer['name'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), cfg_optimizer['lr'], cfg_optimizer['weight_decay'])
        elif cfg_optimizer['name'] == 'RAdam':
            optimizer = torch.optim.RAdam(model.parameters(), cfg_optimizer['lr'], cfg_optimizer['weight_decay'])
        else:
            raise NotImplemented(f"Wrong opt")

        return optimizer

    # def dataset_build(self):
    #
    # def loss_build(self):


if __name__ == '__main__':
    cfg = load_config('./config/main.yaml')
    build = Builder(cfg)
    model = build.model_build()
    opt = build.opt_build(model)
    print(opt)
    print(type(model))
    print(model)
