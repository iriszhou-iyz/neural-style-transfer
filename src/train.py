import torch
import torch.optim as optim

from model import VGGFeatures
from losses import ContentLoss, StyleLoss
from utils import load_image

# gpu if available, else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_img = load_image("images/content.jpg").to(device)
style_img = load_image("images/style.jpg").to(device)

# pretrained VGG19 network to extract image features - network weights frozen
model = VGGFeatures().to(device).eval()

content_layers = [21]  # conv4_2 approx
style_layers = [0, 5, 10, 19, 28]

generated = content_img.clone().requires_grad_(True)

optimizer = optim.Adam([generated], lr=0.03)

num_steps = 300
style_weight = 1e6
content_weight = 1

for step in range(num_steps):

    gen_features = model(generated)
    content_features = model(content_img)
    style_features = model(style_img)

    content_loss = 0
    style_loss = 0

    # content loss
    content_loss += torch.mean((gen_features[21] - content_features[21]) ** 2)

    # style loss
    for i in style_layers:
        gm_gen = torch.mm(gen_features[i].view(gen_features[i].shape[1], -1),
                          gen_features[i].view(gen_features[i].shape[1], -1).t())

        gm_style = torch.mm(style_features[i].view(style_features[i].shape[1], -1),
                            style_features[i].view(style_features[i].shape[1], -1).t())

        style_loss += torch.mean((gm_gen - gm_style) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {total_loss.item()}")

# save output
torch.save(generated, "images/output/stylized.pt")