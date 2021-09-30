# import torch
# from utill.utills import model_info
#
# def eeal(dataset, weight, batch_size, img_size, conf_thres, iou_th, model):
#
#     training = model is not None
#     if training:
#         device = next(model.parameters()).device
#
#     else:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model.eval()
#     # nc = 1 if single_cls else int(data['nc'])  # nu
#     iouv = torch.linspace(0.5, 0.95, 10).to(device)
#     niou = iouv.numel()
#
#
#     #  dataloader
#     if not training:
#
