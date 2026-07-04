from PIL import Image
import torchvision.transforms as T
import torch

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