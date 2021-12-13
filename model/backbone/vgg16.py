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
        self.node = ['features.22', 'features.30']
        self.feature_extractor = create_feature_extractor(vgg16, return_nodes=self.node)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return x[self.node[0]], x[self.node[1]]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16().to(device)
    node, _ = get_graph_node_names(model)
    print(node)
    # feature_extractor = create_feature_extractor(model, return_nodes=['features.28', 'features.29', 'features.30'])
    test = torch.Tensor(1, 3, 300, 300).to(device)
    out, out1 = model(test)
    print(out.size())
    print(out1.size())


    # print(out['features.28'].size())
    # print(out['features.29'].size())
    # print(out['features.30'].size())
    # print(f'{out.shape=}')
    model_info(model, 1, 3, 300, 300, device)