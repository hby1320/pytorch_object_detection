import os

import torch.distributed
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.tensorboard
import tqdm
import torch
from dataset.pascalvoc import PascalVoc, collate_fn
from model.od import Fcos, proposed
from model.od.Fcos import GenTargets, Loss
from utill.utills import model_info,PolyLR
from torch.optim import adam, adamw, SGD, Optimizer

from torchvision.transforms import transforms

EPOCH = 100
LR_INIT = 2e-3
MOMENTUM = 0.9
WEIGHTDECAY = 0.0001
model_name = 'FCOS'
amp_enabled = True
ddp_enabled = False
if __name__ == '__main__':
    torch.manual_seed(0)

    if ddp_enabled:
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

    tran = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(512),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.RandomRotation(0.5),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ])

    #  1 Data load
    voc_07_train = PascalVoc(root='./data/voc', year = "2007", image_set = "train", download = False, transforms = tran)
    voc_12_train = PascalVoc(root='./data/voc', year = "2012", image_set = "train", download = False, transforms = tran)
    voc_07_trainval = PascalVoc(root = './data/voc', year = "2012", image_set = "trainval", download = False)

    voc_train = ConcatDataset([voc_07_train, voc_12_train])
    train_dataloder = DataLoader(voc_train, batch_size = 20, num_workers =4, collate_fn= collate_fn)
    valid_dataloder = DataLoader(voc_07_trainval, batch_size = 2, num_workers = 4, collate_fn = collate_fn)

    model = Fcos.FCOS([2048, 1024, 512], 20, 256).to(device)
    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


    print(f'Activated model: {model_name} (rank{local_rank})')

    optimizer = SGD(model.parameters(), lr=LR_INIT, momentum = MOMENTUM, weight_decay = WEIGHTDECAY)
    scheduler = PolyLR(optimizer, len(train_dataloder) * EPOCH)
    scaler = torch.cuda.amp.GradScaler(enabled = ddp_enabled)
    gen_target = GenTargets(s=[8,16,32,64,128], limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]])
    loss = Loss()

    start_epoch = 0
    prev_mAP = 0.0
    prev_val_loss = 2 ** 32 - 1
    ##  TODO resume Train 추가

    if local_rank == 0:
        writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name))
    else:
        writer = None

    # 5 Train & val
    for epoch in tqdm.tqdm(range(start_epoch, EPOCH), desc = 'Epoch', disable = False if local_rank == 0 else True):

        # if torch.utils.train_interupter.train_interupter():
        #     print('Train interrupt occurs.')
        #     break
        #
        # if ddp_enabled:
        #     trainloader.sampler.set_epoch(epoch)
        #     torch.distributed.barrier()
        model.train()

        for batch_idx, (imgs, targets, classes) in enumerate(tqdm.tqdm(train_dataloder, desc ='Batch', leave=False,
                                                                       disable=False if local_rank == 0 else True)):
            iters = len(train_dataloder) * epoch + batch_idx
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none = True)
            with torch.cuda.amp.autocast(enabled = amp_enabled):
                outputs = model(imgs)
                targets = gen_target(outputs, targets, classes)
                loss = loss(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ddp_enabled:
                loss_list = [torch.zeros(1, device = device) for _ in range(world_size)]
                torch.distributed.all_gather_multigpu([loss_list], [loss])
                if writer is not None:
                    for i, rank_loss in enumerate(loss_list):
                        writer.add_scalar(f'loss/training (rank{i})', rank_loss.item(), iters)
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)
            else:
                writer.add_scalar(f'loss/training (rank{local_rank})', loss.item(), iters)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)

            scheduler.step()

