import torch
import torchvision
import torch.nn as nn
from utills import model_info


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.backbone = torchvision.models.MobileNetV2()
        self.feature_extractor = torchvision.models.feature_extraction.create_feature_extractor(self.backbone,
                                                                                 return_nodes = ['features.18.2'])
        print(torchvision.models.feature_extraction.get_graph_node_names(self.backbone))
        # self.feature_extractor = torchvision.models.feature_extraction.create_feature_extractor(mobilenetv2,)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return x

#  'features.18.2'


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_tensor = torch.Tensor(1,3,512,512).to(device)
    model = MobileNetV2().to(device)
    print(model(dummy_tensor).size())
    # model_info(model, 1, 3, 512, 512, device, 3)
    # mobilenetv2 = torchvision.models.MobileNetV2()

