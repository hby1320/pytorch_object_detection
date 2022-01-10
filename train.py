import os
import torch.distributed
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.tensorboard
from tqdm import tqdm
import torch
from model.modules.head import FCOSGenTargets
from dataset.voc import VOCDataset
from model.od.Fcos import FCOS
from model.loss import FCOSLoss
from model.od.proposed import HalfInvertedStageFCOS
from torch.optim import SGD, Adam
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
from data.augment import Transforms
from utill.utills import model_info, PolyLR, voc_collect
from dataset.pascalvoc import PascalVoc
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchvision.transforms import transforms
from test import evaluate

EPOCH = 50
batch_size = 16
LR_INIT = 1e-2  # 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

# mode = 'FCOS'
mode = 'proposed'
if mode == 'FCOS':
    model_name = 'FCOS_head_fix'
else:
    model_name = 'test_head_his_fpn'
opt = 'SGD'
amp_enabled = True
ddp_enabled = False
swa_enabled = False
Transform = Transforms()
# Transform = None

if __name__ == '__main__':
    #  DDP setting
    if ddp_enabled:
        assert torch.distributed.is_nccl_available(), 'NCCL backend is not available.'
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        os.system('clear')
    else:
        local_rank = 0
        world_size = 0
    #  Device setting
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    #  1 Data loader
    # voc_07_train = PascalVoc(root = "./data/voc/", year = "2007", image_set = "trainval", download = False,
    #                          transforms = data_transform1)
    #
    # voc_12_train = PascalVoc(root = "./data/voc/", year = "2012", image_set = "trainval", download = False,
    #                          transforms = data_transform1)

    # voc_07_test = PascalVoc(root="./data/voc/", year="2007", image_set="test", download=False,
    #                          transforms=data_transform1)

    voc_07_train = VOCDataset('./data/voc/VOCdevkit/VOC2007', [512, 512], "trainval", False, True, Transform)
    voc_12_train = VOCDataset('./data/voc/VOCdevkit/VOC2012', [512, 512], "trainval", False, True, Transform)
    # voc_07_trainval = VOCDataset('./data/voc/VOCdevkit/VOC2007', [512, 512], "trainval", True, True)
    voc_train = ConcatDataset([voc_07_train, voc_12_train])  # 07 + 12 Dataset
    print(len(voc_07_train+voc_12_train))
    if ddp_enabled:
        sampler = torch.utils.data.DistributedSampler(voc_07_train+voc_12_train)
        shuffle = False
        pin_memory = False
        train_dataloader = DataLoader(voc_07_train + voc_12_train, batch_size=batch_size, shuffle=shuffle,
                                      sampler=sampler, num_workers=4, pin_memory=pin_memory,
                                      collate_fn=voc_07_train.collate_fn)
    else:
        sampler = False
        train_dataloader = DataLoader(voc_07_train + voc_12_train, batch_size=batch_size, shuffle=True, num_workers=4,
                                      collate_fn=voc_07_train.collate_fn, worker_init_fn=np.random.seed(0),
                                      pin_memory=True)
        # train_dataloader = DataLoader(voc_07_train + voc_12_train, batch_size=batch_size, shuffle=True, num_workers=4,
        #                              collate_fn = voc_collect,  pin_memory= True)
        # valid_dataloader = DataLoader(voc_07_test, batch_size = batch_size, num_workers = 4,
        #                              collate_fn = voc_collect,  pin_memory= True)
    if mode == 'FCOS':
        model = FCOS([2048, 1024, 512], 20, 256).to(device)

        # gen_target = GenTargets(strides=[8, 16, 32],
        #                         limit_range=[[-1, 64], [64, 128], [128, 9999999]])
    elif mode == 'proposed':
        model = HalfInvertedStageFCOS([512, 1024, 2048], 20, 256).to(device)
    gen_target = FCOSGenTargets(strides=[8, 16, 32, 64, 128],
                                limit_range=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]])

    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if swa_enabled:
        swa_model = AveragedModel(model)
    print(f'Activated model: {model_name} (rank{local_rank})')

    if opt == 'SGD':
        optimizer = SGD(model.parameters(),
                        lr=LR_INIT,
                        momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY)

    elif opt == 'Adam':
        optimizer = Adam(model.parameters(),
                         lr=LR_INIT)

    # scheduler = LambdaLR(optimizer=optimizer,
    #                      lr_lambda=lambda EPOCH: 0.95 ** EPOCH,
    #                      last_epoch=-1,
    #                      verbose=False)
    # scheduler = PolyLR(optimizer, len(train_dataloder) * EPOCH)
    # swa_start = 5
    # scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloder))
    # swa_scheduler = SWALR(optimizer, swa_lr = 0.05)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = FCOSLoss('giou')  # 'iou'
    nb = len(train_dataloader)
    start_epoch = 0
    prev_mAP = 0.0
    best_loss = 0
    WARMUP_STEPS = 501
    GLOBAL_STEPS = 1

    if local_rank == 0:
        writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name))
    else:
        writer = None

    # 5 Train & val
    for epoch in tqdm(range(start_epoch, EPOCH), desc='Epoch', disable=False if local_rank == 0 else True):

        # if torch.utils.train_interupter.train_interupter():
        #     print('Train interrupt occurs.')
        #     break

        if ddp_enabled:  # DDP option setting
            train_dataloader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        model.train()

        pbar = enumerate(train_dataloader)
        print(f'{"Gpu_mem":10s} {"cls":>10s} {"cnt":>10s} {"reg":>10s} {"total":>10s} ')
        pbar = tqdm(pbar, total=nb, desc='Batch', leave=True, disable=False if local_rank == 0 else True)
        for batch_idx, (imgs, targets, classes) in pbar:
            iters = len(train_dataloader) * epoch + batch_idx
            imgs, targets, classes = imgs.to(device), targets.to(device), classes.to(device)

            if GLOBAL_STEPS < WARMUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMUP_STEPS * LR_INIT)
                for param in optimizer.param_groups:
                    param['lr'] = lr

            if GLOBAL_STEPS == 20001:  # 20001
                lr = LR_INIT * 0.1
                for param in optimizer.param_groups:
                    param['lr'] = lr

            if GLOBAL_STEPS == 27001:  # 27001
                lr = LR_INIT * 0.01
                for param in optimizer.param_groups:
                    param['lr'] = lr
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(imgs)
                target = gen_target([outputs, targets, classes])
                losses = criterion([outputs, target])
                loss = losses[-1]
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()

            if ddp_enabled:
                loss_list = [torch.zeros(1, device=device) for _ in range(world_size)]
                torch.distributed.all_gather_multigpu([loss_list], [loss])
                if writer is not None:
                    for i, rank_loss in enumerate(loss_list):
                        writer.add_scalar(f'loss/training (rank{i})', rank_loss.item(), iters)
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)
            else:
                writer.add_scalar(f'loss/training (rank{local_rank})', losses[-1], iters)
                writer.add_scalar(f'loss/training/batch cls_loss', losses[0], epoch)
                writer.add_scalar(f'loss/training/batch cnt_loss', losses[1], epoch)
                writer.add_scalar(f'loss/training/batch reg_loss', losses[2], epoch)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = f'{mem:10s} {losses[0].mean():10.4g} {losses[1].mean():10.4g} {losses[2].mean():10.4g} {losses[-1].mean():10.4g}'
            pbar.set_description(s)
            GLOBAL_STEPS += 1
        # if epoch > swa_start:
        #     swa_model.update_parameters(model)
        #     swa_scheduler.step()
        #
        # else:
        #     scheduler.step()

            # scheduler.step()

            # evaluate(model, valid_dataloder, True, False, device)
        # if epoch % 5 == 0:
        # evaluate(model, valid_dataloder, amp_enabled, ddp_enabled, device, voc_07_trainval)

        # if loss > best_loss:
        #     torch.save(model.state_dict(), f"./checkpoint/{model_name}_best_loss.pth")
        #     best_loss = loss

        if epoch >= (EPOCH - 5):
            torch.save(model.state_dict(), f"./checkpoint/{model_name}_{epoch + 1}.pth")

    if writer is not None:
        writer.close()
    if ddp_enabled:
        torch.distributed.destroy_process_group()
