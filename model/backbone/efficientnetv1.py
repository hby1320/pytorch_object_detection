from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from utill.utills import model_info
from efficientnet_pytorch.utils import get_same_padding_conv2d, round_filters, calculate_output_image_size,\
                                       round_repeats



class EfficientNetV1(nn.Module):
    def __init__(self, backbone_number:int, memory_efficient:bool):
        super(EfficientNetV1, self).__init__()
        VALID_MODELS = (
            'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
            'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
            'efficientnet-b8',

            # Support the construction of 'efficientnet-l2' without pretrained weights
            'efficientnet-l2'
        )
        self.model = EfficientNet.from_pretrained(model_name = VALID_MODELS[backbone_number])
        self.model.set_swish(memory_efficient = memory_efficient)

    def forward(self, x:torch.Tensor) -> torch.tensor:
        x = self.model.extract_endpoints(x)
        return x['reduction_1'],x['reduction_2'],x['reduction_3'],x['reduction_4'],x['reduction_5'],x['reduction_6']


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV1(0, False).to(device)
    model_info(model, 1, 3, 512, 512, device)
