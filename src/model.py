import torch
import torch.nn as nn
import torchvision.models as models

# wrap a pretrained vgg19 network and get intermediate feature maps
class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        
        # load pretrained vgg19 model - trained on imagenet
        # .features: only use convolutional feature extractor
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        # store all convolutional layers
        self.layers = vgg[:].eval()

        # freeze all parameters - don't train vgg
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # store feature maps of each layer as we go through the network - for content/style loss
        features = []

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features