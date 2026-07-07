import torch
import torch.optim as optim

from model import VGGFeatures
from losses import ContentLoss, StyleLoss
from utils import load_image, save_output

# gpu if available, else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_img = load_image("images/content.jpg").to(device)
style_img = load_image("images/style.jpg").to(device)

# pretrained VGG19 network to extract image features - network weights frozen
model = VGGFeatures().to(device).eval()

# approx conv4_2 - high-level semantic content
CONTENT_LAYER = 21
STYLE_LAYERS = [0, 5, 10, 19, 28]

# compute feature maps, never change during optimization
content_features = model(content_img)
style_features = model(style_img)

# reusable loss modules
# StyleLoss: stores target Gram matrix from style image
# ContentLoss: stores target feature map from content image
content_loss_fn = ContentLoss(content_features[CONTENT_LAYER])
style_loss_fns = [StyleLoss(style_features[layer]) for layer in STYLE_LAYERS]

# tensor that will be updated in optimization, starts from content image
generated = content_img.clone().requires_grad_(True)

# updates pixels of the generated image
optimizer = optim.Adam([generated], lr=0.03)

# hyperparameters
num_steps = 300
# larger style weight -> more artistic image
style_weight = 1e6
# larger content weight -> more structure
content_weight = 1

for step in range(num_steps):
    # compute vgg feature maps for generated image
    gen_features = model(generated)

    content_loss = content_loss_fn(gen_features[CONTENT_LAYER])

    # styleloss compares Gram matrices between generated and style feature maps
    style_loss = 0
    for loss_fn, layer in zip(style_loss_fns, STYLE_LAYERS):
        style_loss += loss_fn(gen_features[layer])

    total_loss = content_weight * content_loss + style_weight * style_loss

    # backprop: compute loss gradients wrt generated image and update its pixels
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(
            f"Step {step:3d} | "
            f"Content: {content_loss.item():.4f} | "
            f"Style: {style_loss.item():.4f} | "
            f"Total: {total_loss.item():.4f}"
        )

save_output(generated, "images/output/stylized.png")