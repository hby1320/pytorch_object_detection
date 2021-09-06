import os

import torch.distributed
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.tensorboard
from tqdm import tqdm
import torch
# from dataset.pascalvoc import PascalVoc
from dataset.voc import VOCDataset
from model.od.Fcos import FCOS, GenTargets, Loss
from model.od.Mc_Fcos import MC_FCOS
from utill.utills import model_info, PolyLR
from torch.optim import SGD, Adam
import numpy as np
from torchvision.transforms import transforms
import random
from test import evaluate

EPOCH = 10
batch_size = 14
# LR_INIT = 0.0001  # amp not using
LR_INIT = 0.0001
MOMENTUM = 0.9
WEIGHTDECAY = 0.0001

# mode = 'FCOS'
mode = 'proposed'
if mode == 'FCOS':
    model_name = 'FCOS'
else:
    model_name = 'Test'
opt = 'Adam'
amp_enabled = True
ddp_enabled = False


if __name__ == '__main__':

    # DDP option
    if ddp_enabled:
        assert torch.distributed.is_nccl_available(), 'NCCL backend is not available.'
        torch.distributed.init_process_group(backend= 'nccl', init_method='env://')
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        os.system('clear')
    else:
        local_rank = 0
        world_size = 0

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    # transform = transforms.Compose([
    #     transforms.Resize((512,512)),
    #     transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    #     transforms.RandomRotation(0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])


    #  1 Data load
    # voc_07_train = PascalVoc(root='./data/voc', year = "2007", image_set = "train", download = False, transforms = transform)
    # voc_12_train = PascalVoc(root='./data/voc', year = "2012", image_set = "train", download = False, transforms = transform)
    # voc_07_trainval = PascalVoc(root = './data/voc', year = "2012", image_set = "trainval", download = False)
    voc_07_train = VOCDataset('./data/voc/VOCdevkit/VOC2007', [512, 512], "train", False, True, )
    voc_12_train = VOCDataset('./data/voc/VOCdevkit/VOC2012', [512, 512], "train", False, True, )
    voc_07_trainval = VOCDataset('./data/voc/VOCdevkit/VOC2007', [512, 512], "trainval", True, True)
    voc_train = ConcatDataset([voc_07_train, voc_12_train])  # 07 + 12 Dataset

    if ddp_enabled:
        sampler = torch.utils.data.DistributedSampler(voc_train)
        shuffle = False
        pin_memory = False
        train_dataloder = DataLoader(voc_train, batch_size = batch_size, shuffle = shuffle, sampler=sampler,
                                     num_workers = 4, pin_memory= pin_memory, collate_fn = voc_07_train.collate_fn)
    else:
        sampler = False
        train_dataloder = DataLoader(voc_train, batch_size = batch_size, shuffle = True, num_workers = 4,
                                     collate_fn = voc_07_train.collate_fn)
        valid_dataloder = DataLoader(voc_07_trainval, batch_size = 1, num_workers = 4,
                                     collate_fn = voc_07_trainval.collate_fn)
    if mode == 'FCOS':
        model = FCOS([2048, 1024, 512], 20, 256).to(device)
        gen_target = GenTargets(strides=[8, 16, 32, 64, 128],
                                limit_range=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]])
    elif mode =='proposed':
        model = MC_FCOS([512, 1024, 2048], 20, 256).to(device)
        gen_target = GenTargets(strides=[8, 16, 32, 64],
                                limit_range=[[-1, 64], [64, 128], [128, 256], [256, 512]])

    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f'Activated model: {model_name} (rank{local_rank})')

    if opt == 'SGD':
        optimizer = SGD(model.parameters(), lr=LR_INIT, momentum = MOMENTUM, weight_decay = WEIGHTDECAY)
    elif opt == 'Adam':
        optimizer = Adam(model.parameters(), lr=LR_INIT)
    # scheduler = LambdaLR(optimizer=optimizer,
    #                      lr_lambda=lambda EPOCH: 0.95 ** EPOCH,
    #                      last_epoch=-1,
    #                      verbose=False)
    scheduler = PolyLR(optimizer, len(train_dataloder) * EPOCH)
    scaler = torch.cuda.amp.GradScaler(enabled = ddp_enabled)
    # gen_target = GenTargets(strides=[8,16,32,64,128],
    #                         limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]])
    # gen_target = GenTargets(strides=[8,16,32,64],
    #                         limit_range=[[-1,64],[64,128],[128,256],[256,999999]])
    criterion = Loss(mode='iou')  # 'iou'


    nb = len(train_dataloder)
    start_epoch = 0
    prev_mAP = 0.0
    prev_val_loss = 2 ** 32 - 1

    if local_rank == 0:
        writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name))
    else:
        writer = None

    # 5 Train & val
    for epoch in tqdm(range(start_epoch, EPOCH), desc = 'Epoch', disable = False if local_rank == 0 else True):

        # if torch.utils.train_interupter.train_interupter():
        #     print('Train interrupt occurs.')
        #     break
        #
        # if ddp_enabled:
        #     train_dataloder.sampler.set_epoch(epoch)
        #     torch.distributed.barrier()
        model.train()
        pbar = enumerate(train_dataloder)
        # print(('\n' + '%10s' * 8) % ('Epoch', 'Gpu_mem', 'CIoU', 'Obj', 'Cls', 'Total', 'Targets', 'Img_size'))
        # print(f'{"cls_loss":12s} {"cnt_loss":12s} {"reg_loss":12s} {"total_loss":12s} {"progressbar":12s}')

        print(f'{"Gpu_mem":10s} {"cls":>10s} {"cnt":>10s} {"reg":>10s} {"total":>10s} ')
        pbar = tqdm(pbar, total = nb,desc = 'Batch', leave = False, disable = False if local_rank == 0 else True)
        for batch_idx, (imgs, targets, classes) in pbar:

            iters = len(train_dataloder) * epoch + batch_idx
            imgs, targets, classes = imgs.to(device), targets.to(device), classes.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = amp_enabled):
                outputs = model(imgs)
                targets = gen_target([outputs, targets, classes])
                cls_loss, cnt_loss, reg_loss, total_loss = criterion([outputs, targets])
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ddp_enabled:
                loss_list = [torch.zeros(1, device = device) for _ in range(world_size)]
                torch.distributed.all_gather_multigpu([loss_list], [total_loss])
                if writer is not None:
                    for i, rank_loss in enumerate(loss_list):
                        writer.add_scalar(f'loss/training (rank{i})', rank_loss.item(), iters)
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)
            else:
                writer.add_scalar(f'loss/training (rank{local_rank})', total_loss, iters)
                writer.add_scalar(f'loss/training/batch', cls_loss, epoch)
                writer.add_scalar(f'loss/training/batch', cnt_loss, epoch)
                writer.add_scalar(f'loss/training/batch', reg_loss, epoch)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            # total_losses = (total_loss[-1] * batch_idx + total_loss) / (batch_idx + 1)
            s = (f'{mem:10s} {cls_loss.item():10.4g} {cnt_loss.item():10.4g} {reg_loss.item():10.4g} {total_loss.item():10.4g}')
            pbar.set_description(s)
            scheduler.step()
        # if epoch % 5 == 0:
        evaluate(model, valid_dataloder, amp_enabled, ddp_enabled, device, voc_07_trainval)

        ##  epoch 마다 저장
        if epoch > EPOCH - 10:
            torch.save(model.state_dict(), f"./checkpoint/{model_name}_{epoch + 1}.pth")
