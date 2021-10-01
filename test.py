import csv
import os
import time

import torch
import torch.nn as nn
import torch.distributed
import torch.utils.data
from tqdm import tqdm

from model.modules.head import FCOSHead, ClipBoxes
import numpy as np
from model.od.Fcos import FCOS
from model.od.proposed import FRFCOS




def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
    pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores


def iou_2d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(x1,y1,x2,y2)]
    :param cubes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # expands dim
    # cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    # cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]
    cubes_a = torch.unsqueeze(cubes_a, dim = 1)  # [N,1,4]
    cubes_b = torch.unsqueeze(cubes_b, dim = 0)  # [1,M,4]

    # overlap = np.maximum(0.0,
    #                      np.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
    #                      np.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]
    overlap = torch.maximum(0.0,
                         torch.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
                         torch.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]

    # overlap
    # overlap = np.prod(overlap, axis=-1)  # [N,M]
    overlap = torch.prod(overlap, dim = -1)  # [N,M]
    # compute area
    # area_a = np.prod(cubes_a[..., 2:] - cubes_a[..., :2], axis=-1)
    # area_b = np.prod(cubes_b[..., 2:] - cubes_b[..., :2], axis=-1)
    area_a = torch.prod(cubes_a[..., 2:] - cubes_a[..., :2], dim = -1)
    area_b = torch.prod(cubes_b[..., 2:] - cubes_b[..., :2], dim = -1)
    # compute iou
    iou = overlap / (area_a + area_b - overlap)
    return iou


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    # mrec = np.concatenate(([0.], recall, [1.]))
    # mpre = np.concatenate(([0.], precision, [0.]))
    mrec = torch.cat(([0.], recall, [1.]))
    mpre = torch.cat(([0.], precision, [0.]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        # mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    # i = np.where(mrec[1:] != mrec[:-1])[0]
    i = torch.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    # ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_ap_2d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 4, total number of class including background which is equal to 0
    :return: a dict containing average precision for each cls
    """
    all_ap = {}
    for label in range(num_cls)[1:]:
        # get samples with specific label
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]
        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        # fp = np.zeros((0,))
        # tp = np.zeros((0,))
        # scores = np.zeros((0,))
        fp = torch.zeros((0,))
        tp = torch.zeros((0,))
        scores = torch.zeros((0,))
        total_gts = 0
        # loop for each sample
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            for index in range(len(sample_pred_box)):
                # scores = np.append(scores, sample_scores[index])
                scores = torch.cat([scores, sample_scores[index]],dim=1)
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    # fp = np.append(fp, 1)
                    # tp = np.append(tp, 0)
                    fp = torch.cat((fp, 0), dim = 1)
                    tp = torch.cat((tp, 1), dim = 1)
                    continue
                # pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                pred_box = torch.unsqueeze(sample_pred_box[index], dim=0)
                iou = iou_2d(sample_gts, pred_box)
                # gt_for_box = np.argmax(iou, axis=0)
                gt_for_box = torch.argmax(iou, dim = 0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    # fp = np.append(fp, 0)
                    # tp = np.append(tp, 1)
                    # assigned_gt.append(gt_for_box)
                    fp = torch.cat((fp, 0),dim=1)
                    tp = torch.cat((tp, 1),dim=1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = torch.cat((fp, 0), dim = 1)
                    tp = torch.cat((tp, 1), dim = 1)
                    # fp = np.append(fp, 1)
                    # tp = np.append(tp, 0)
        # sort by score
        # indices = np.argsort(-scores)
        indices = torch.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        # fp = np.cumsum(fp)
        # tp = np.cumsum(tp)
        fp = torch.cumsum(fp,dim=0)
        tp = torch.cumsum(tp,dim=0)
        # compute recall and precision
        recall = tp / total_gts
        # precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        precision = tp / torch.maximum(tp + fp, torch.finfo(torch.float64).eps)
        ap = _compute_ap(recall, precision)
        all_ap[label] = ap
        # print(recall, precision)
    return all_ap


def evaluate(model: nn.Module,
             vall_data_loader: torch.utils.data.DataLoader,
             # criterion: nn.Module,
             amp_enable: bool,
             ddp_enable: bool,
             device: torch.device,
             ):
    score_threshold = 0.05
    nms_iou_threshold = 0.6
    max_detection_boxes_num = 1000
    # strides = [8, 16, 32, 64, 128]
    strides = [8, 16, 32]
    gt_boxes = []
    gt_classes = []
    pred_boxes = []
    pred_classes = []
    pred_scores = []
    model.eval()

    if ddp_enable:
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 0
    head = FCOSHead(score_threshold, nms_iou_threshold, max_detection_boxes_num, strides)
    Clip = ClipBoxes()
    inference_time = torch.zeros(1, device = device)
    # val_loss = torch.zeros(4, device = device)
    pbar = enumerate(vall_data_loader)
    nb = len(vall_data_loader)
    pbar = tqdm(pbar, total=nb, desc='Batch', leave=True, disable=False if local_rank == 0 else True)
    for batch_idx, (imgs, targets, classes) in pbar:
        imgs, targets, classes = imgs.to(device), targets.to(device), classes.to(device)
        with torch.cuda.amp.autocast(enabled=amp_enable):
            # torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                out = model(imgs)
                score, cls, boxes = head(out)
                box = Clip(imgs, boxes)
                # pred_boxes.append(box[0].cpu().numpy())
                # pred_classes.append(cls[0].cpu().numpy())
                # pred_scores.append(score[0].cpu().numpy())
                pred_boxes.append(box[0])
                pred_classes.append(cls[0])
                pred_scores.append(score[0])
            # gt_boxes.append(targets[0].cpu().numpy())
            # gt_classes.append(classes[0].cpu().numpy())
            gt_boxes.append(targets[0])
            gt_classes.append(classes[0])
            # torch.cuda.synchronize()
            inference_time += time.time() - start_time

    pred_boxes, pred_classes, pred_scores = sort_by_score(pred_boxes, pred_classes, pred_scores)
    all_AP = eval_ap_2d(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5, 20)
    CLASSES_NAME = (
        "__background__ ", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",)

    print(f'all classes AP=====>\n')
    for key, value in all_AP.items():
        # print(f'ap for {vialder.id2name[int(key)]} is {value}')
        print(f'ap for {CLASSES_NAME[int(key)]} is {value}')
    mAP = 0.
    for class_id, class_mAP in all_AP.items():
        mAP += float(class_mAP)
    mAP /= (len(CLASSES_NAME) - 1)
    print(f'mAP=====>{mAP:.3f}\n {inference_time=}')


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utill.utills import voc_collect
    from dataset.pascalvoc import PascalVoc
    from torchvision.transforms import transforms
    data_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 2
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    voc_07_trainval = PascalVoc(root ="./data/voc/", year = "2007", image_set = "test", download = False, transforms=data_transform)
    #voc_07_trainval = VOCDataset('./data/voc/VOCdevkit/VOC2007', [512, 512], "test", False, False)
    #valid_dataloder = DataLoader(voc_07_trainval, batch_size=batch_size, num_workers=4,
    #                             collate_fn=voc_07_trainval.collate_fn)
    valid_dataloder = DataLoader(voc_07_trainval, batch_size=batch_size, num_workers=4,
                              collate_fn=voc_collect)

    # model = FCOS([2048, 1024, 512], 20, 256).to(device)
    model = FRFCOS([512, 1024, 2048], [128, 256, 512], 20, 256).to(device)
    # model = MC_FCOS([512, 1024, 2048], 20, 256).to(device)
    # model.load_state_dict(torch.load('./checkpoint/FCOS_512_test_30.pth'))
    model.load_state_dict(torch.load('./checkpoint/proposed-lr2_50.pth'))
    evaluate(model, valid_dataloder, False, False, device)
