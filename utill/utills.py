import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary



def model_info(model: nn.Module, batch: int, ch: int, width: int, hight: int, device: torch.device,depth=4):
    col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds")
    img = torch.rand(batch, ch, width, hight).to(device)
    summary(model, img.size(), None, None, col_names, depth = depth)



def generate_anchor(base_size=16, ratios=None, scales=None):
    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchor = len(ratios) * len(scales)
    anchor = np.zeros((num_anchor, 4))
    anchor[:, :2] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchor[:, 2] * anchor[:, 3]

    anchor[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchor[:, 3] = anchor[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchor[:, 0::2] -= np.tile(anchor[:,2] * 0.5, (2,1)).T
    anchor[:, 1::2] -= np.tile(anchor[:, 2] * 0.5, (2, 1)).T

    return anchor


def shift_xy(shape, stride, anchor):
    shift_x = (np.arange(0, shape[1] + 0.5) * stride)
    shift_y = (np.arange(0, shape[0] + 0.5) * stride)

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    ))

    A = anchor.shape[0]
    K = shifts.shape[0]
    all_anchor = (anchor.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchor = all_anchor.reshape((K*A, 4))

    return all_anchor
