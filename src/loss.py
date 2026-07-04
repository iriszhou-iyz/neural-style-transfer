import torch
import torch.nn as nn

def gram_matrix(feature):
    # feature shape: (batch, channels, height, width)
    b, c, h, w = feature.size()
    # flatten spatial dimensions so each channel is a vector
    x = feature.view(c, h * w)

    # compute gram matrix: gram[i][j] = feature i correlation to feature j
    gram = torch.mm(x, x.t())
    # normalize
    return gram / (c * h * w)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        # target stores target content feature map
        # detach(): do NOT backprop through the target
        self.target = target.detach()

    def forward(self, input):
        content_diff = input - self.target
        # mse over feature maps
        return torch.mean(content_diff ** 2)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        # convert target style feature map into gram matrix
        # do not backprop through the target
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        g_diff = G - self.target
        # mse between gram matrices
        return torch.mean(g_diff ** 2)