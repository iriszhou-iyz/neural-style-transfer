from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import torch

# imagenet normalization values from ImageNet-1K training set
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# load and preprocess image
def load_image(path, size=512):
    img = Image.open(path).convert("RGB")

    # preprocessing: resize image for consistency, convert to tensor
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])
    # add batch dimensions: (1, C, H, W)
    img = transform(img).unsqueeze(0)
    return img

# convert tensor back to image
def tensor_to_image(tensor):
    # clone to not modify original, remove batch dimension
    img = tensor.clone().detach().squeeze(0)
    img = torch.clamp(img, 0, 1)
    return img

def denormalize(tensor):
    # convert from optimized imagenet for vgg to standard rgb values
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(STD).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def save_output(tensor, filename):
    img = denormalize(tensor).clamp(0, 1)
    save_image(img, filename)