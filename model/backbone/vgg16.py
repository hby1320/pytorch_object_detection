import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from utill.utills import model_info


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # self.model_node, _ = get_graph_node_names(vgg16)
        # print(self.model_node)
        self.node = ['features.22', 'features.30']  # conv4-3, conv5-3 -> maxpool
        self.feature_extractor = create_feature_extractor(vgg16, return_nodes=self.node)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return x[self.node[0]], x[self.node[1]]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16().to(device)
    model_info(model, 1, 3, 300, 300, device)