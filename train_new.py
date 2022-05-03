import datetime
import os
import torch.distributed
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.tensorboard
from tqdm import tqdm
import torch
from model.modules.head import FCOSGenTargets
from dataset.voc import VOCDataset
import model.od as OD
from model.loss import FCOSLoss
from torch.optim import SGD, Adam
from utill.utills import load_config, voc_collect
import numpy as np
import torchvision.transforms as tfs
from data.augment import Transforms
# from dataset.pascalvoc import PascalVoc
from dataset.pascalvoc import PascalVoc
from test import evaluate
Transform = Transforms()

if __name__ == '__main__':
    # Transform = tfs.Compose([tfs.ToTensor(),
    #                          tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #                          tfs.Resize([512, 512])])
    cfg = load_config('./config/main.yaml')
    name = cfg['model']['name']
    save = cfg['savename']
    day = datetime.date.today()
    save_name = name + '_' + save
    print(f"save_name:{save_name}")
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

    #  Device setting
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    # 2. Data loader
    voc_07_train = VOCDataset(cfg['dataset_setting']['train_07'], cfg['dataset_setting']['input'],
                              cfg['dataset_setting']['type'], False, True, Transform)
    voc_12_train = VOCDataset(cfg['dataset_setting']['train_12'], cfg['dataset_setting']['input'],
                              cfg['dataset_setting']['type'], False, True, Transform)
    # voc_07_test = VOCDataset(cfg['dataset_setting']['train_07'], cfg['dataset_setting']['input'],'test',
    #                        False, True)
    voc_train = ConcatDataset([voc_07_train, voc_12_train])  # 07 + 12 Dataset
    # voc_07_train = PascalVoc(cfg['dataset_setting']['train_07'], '2007', cfg['dataset_setting']['type'],
    #                          False, transforms=Transform)
    # voc_12_train = PascalVoc(cfg['dataset_setting']['train_12'], '2012', cfg['dataset_setting']['type'],
    #                          False, transforms=Transform)
    # voc_train = ConcatDataset([voc_07_train, voc_12_train])  # 07 + 12 Dataset

    print(len(voc_07_train+voc_12_train))

    if cfg['model']['ddp']:
        sampler = torch.utils.data.DistributedSampler(voc_07_train+voc_12_train)
        shuffle = False
        pin_memory = False
        train_dataloader = DataLoader(voc_train, batch_size=cfg[name]['batch_size'], shuffle=shuffle,
                                      sampler=sampler, num_workers=cfg['dataset_setting']['num_workers'],
                                      pin_memory=pin_memory, collate_fn=voc_07_train.collate_fn)
    else:
        sampler = False
        # train_dataloader = DataLoader(voc_train, batch_size=cfg[name]['batch_size'], shuffle=True,
        #                               num_workers=cfg['dataset_setting']['num_workers'],
        #                               collate_fn=voc_collect, worker_init_fn=np.random.seed(0),
        #                               pin_memory=cfg['dataset_setting']['pin_memory'])
        train_dataloader = DataLoader(voc_train, batch_size=cfg[name]['batch_size'], shuffle=True,
                                      num_workers=cfg['dataset_setting']['num_workers'],
                                      collate_fn=voc_07_train.collate_fn, worker_init_fn=np.random.seed(0),
                                      pin_memory=cfg['dataset_setting']['pin_memory'],
                                      prefetch_factor=cfg['model']['prefetch'],
                                      persistent_workers=cfg['model']['persistent'])
        # test_dataloader = DataLoader(voc_07_test, batch_size=1, num_workers=cfg['dataset_setting']['num_workers'],
        #                              collate_fn=voc_07_train.collate_fn, pin_memory=cfg['dataset_setting']['pin_memory'],
        #                              persistent_workers=True)
    nb = len(train_dataloader)
    # Model build
    if name == 'FCOS':
        model = OD.Fcos.FCOS(cfg[name]['CannelofBackbone'], cfg['dataset_setting']['class_num'],
                             cfg[name]['channel'],).to(device)
    elif name == 'HISFCOS':
        model = OD.HISFcos.HalfInvertedStageFCOS(cfg[name]['CannelofBackbone'], cfg['dataset_setting']['class_num'],
                                                 cfg[name]['channel'],).to(device)
    elif name == 'MNFCOS':
        model = OD.MNFcos.MNFCOS(cfg[name]['CannelofBackbone'], cfg['dataset_setting']['class_num'],
                                 cfg[name]['channel']).to(device)
        # model = MNHeadFCOS([ 2048, 1024, 512 ], 20, 128).to(device)
    else:
        breakpoint()
    gen_target = FCOSGenTargets(strides=cfg[name]['stride'], limit_range=cfg[name]['range'])

    if cfg['model']['ddp']:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Optimizer
    print(f"Optimizer={cfg[name]['optimizer']['name']}")
    if cfg[name]['optimizer']['name'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=cfg[name]['optimizer']['lr'],
                        momentum=cfg[name]['optimizer']['momentum'],
                        weight_decay=cfg[name]['optimizer']['weight_decay'])

    elif cfg[name]['optimizer']['name'] == 'Adam':
        optimizer = Adam(model.parameters(),
                         lr=cfg[name]['optimizer']['lr'])
    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['model']['amp'])
    # Loss function
    criterion = FCOSLoss(cfg[name]['criterion'])  # 'iou'

    start_epoch = 0
    prev_mAP = 0.0
    best_loss = 0
    WARMUP_STEPS = 501
    GLOBAL_STEPS = 1

    if local_rank == 0:
        writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', save_name))
    else:
        writer = None

    # Train
    for epoch in tqdm(range(start_epoch, cfg[name]['Epoch']), desc='Epoch', disable=False if local_rank == 0 else True):

        # if torch.utils.train_interupter.train_interupter():
        #     print('Train interrupt occurs.')
        #     break

        # if cfg['model']['ddp']:  # DDP option setting
        #     train_dataloader.sampler.set_epoch(epoch)
        #     torch.distributed.barrier()
        model.train()

        pbar = enumerate(train_dataloader)
        print(f'{"Gpu_mem":10s} {"cls":>10s} {"cnt":>10s} {"reg":>10s} {"total":>10s} ')
        pbar = tqdm(pbar, total=nb, desc='Batch', leave=True, disable=False if local_rank == 0 else True)
        for batch_idx, (imgs, targets, classes) in pbar:
            iters = len(train_dataloader) * epoch + batch_idx
            imgs, targets, classes = imgs.to(device), targets.to(device), classes.to(device)

            if GLOBAL_STEPS < WARMUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMUP_STEPS * cfg[name]['optimizer']['lr'])
                for param in optimizer.param_groups:
                    param['lr'] = lr

            if GLOBAL_STEPS == 20001:  # 20001
                lr = cfg[name]['optimizer']['lr'] * 0.1
                for param in optimizer.param_groups:
                    param['lr'] = lr

            if GLOBAL_STEPS == 22001:  # 27001
                lr = cfg[name]['optimizer']['lr'] * 0.01
                for param in optimizer.param_groups:
                    param['lr'] = lr

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg['model']['amp']):
                outputs = model(imgs)
                target = gen_target([outputs, targets, classes])
                losses = criterion([outputs, target])
                loss = losses[-1]
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()

            if cfg['model']['ddp']:
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

        if epoch >= (cfg[name]['Epoch'] - 3) or epoch == (cfg[name]['Epoch'] - 1)//2:
            torch.save(model.state_dict(), f"./checkpoint/{save_name}_{epoch + 1}.pth")

    # evaluate(model, voc_07_test, False, False, device)
    if writer is not None:
        writer.close()
    if cfg['model']['ddp']:
        torch.distributed.destroy_process_group()
