import torch
import os
from torch.utils.data import DataLoader, ConcatDataset
from utill.utills import model_info
import torch.utils.tensorboard
from tqdm import tqdm
from utill.utills import load_config
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset.voc import VOCDataset
import model.od as od
from PIL import Image
import matplotlib
import numpy as np
import cv2
import torchvision


def draw_cam_on_image(image: torch.Tensor, mask: np.ndarray, colormap=cv2.COLORMAP_JET) -> torch.Tensor:
    assert torch.min(image) >= 0 and torch.max(image) <= 1, 'Input image should in the range [0, 1]'

    heatmap = cv2.applyColorMap(np.uint8(mask * 255), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1) / 255

    cam = image + heatmap
    cam /= torch.max(cam)
    return cam


if __name__ == '__main__':

    cfg = load_config('./config/main.yaml')
    name = cfg['model']['name']

    #  DDP setting
    if cfg['model']['ddp']:
        assert torch.distributed.is_nccl_available(), 'NCCL backend is not available.'
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        os.system('clear')
    else:
        local_rank = 0
        world_size = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    valset = VOCDataset(cfg['dataset_setting']['train_07'], cfg['dataset_setting']['input'],
                        "val", False, False)


    sampler = False
    # train_dataloader = DataLoader(voc_07_train, batch_size=cfg[name]['batch_size'], shuffle=False,
    #                               num_workers=cfg['dataset_setting']['num_workers'],
    #                               collate_fn=voc_07_train.collate_fn, worker_init_fn=torch.random.seed(),
    #                               pin_memory=cfg['dataset_setting']['pin_memory'])
    # nb = len(train_dataloader)

    if name == 'FCOS':
        model = od.Fcos.FCOS(cfg[name]['CannelofBackbone'], cfg['dataset_setting']['class_num'],
                             cfg[name]['channel'],).to(device)
        target_layer = [model.FPN.P3_c1]

    elif name == 'HISFCOS':
        model = od.proposed.HalfInvertedStageFCOS(cfg[name]['CannelofBackbone'], cfg['dataset_setting']['class_num'],
                                                  cfg[name]['channel'],).to(device)
    elif name == 'MNFCOS':
        model = od.MNFcos.MNFCOS(cfg[name]['CannelofBackbone'], cfg['dataset_setting']['class_num'],
                                 cfg[name]['channel']).to(device)
    else:
        breakpoint()

    model.eval()
    image = valset[1]
    print(image)
    cv2.imshow(image)
    # cam = GradCAM(model=model, target_layers=target_layer, use_cuda=device)
    # targets = [ClassifierOutputTarget(1)]
    #
    # graysclae_cam = cam(input_tensor=train_dataloader, targets=targets)
    # graysclae_cam = graysclae_cam[0, :]
    # visualization = show_cam_on_image(train_dataloader.)


