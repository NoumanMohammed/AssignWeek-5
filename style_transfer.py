# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Neural Style Transfer  (content + style loss with VGG-19)
# Commit message: "Implemented neural style transfer"
#
# Content image : first CIFAR-10 test image (scaled to 128×128)
# Style image   : second CIFAR-10 test image (different class)
# Output        : output_style_transfer.png  (before | style | after grid)
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128   # upscale from 32×32 so VGG has enough spatial resolution

# ── Helper: tensor → numpy for display ───────────────────────────────────────
def tensor_to_np(t):
    img = t.squeeze(0).detach().cpu().permute(1,2,0).numpy()
    return np.clip(img, 0, 1)

# ── 1. Load two CIFAR-10 images as content & style ───────────────────────────
raw_ds = CIFAR10(root="./data", train=False, download=True,
                 transform=transforms.ToTensor())

content_img_raw = raw_ds[0][0]   # airplane
style_img_raw   = raw_ds[1][0]   # automobile (different visual texture)

# Resize to 128×128 and add batch dim
resize_norm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def preprocess(t):
    return resize_norm(t).unsqueeze(0).to(DEVICE)

content_tensor = preprocess(content_img_raw)
style_tensor   = preprocess(style_img_raw)

# ── 2. Build VGG-19 feature extractor ────────────────────────────────────────
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
# Freeze all VGG parameters – we only optimise the generated image
for p in vgg.parameters():
    p.requires_grad_(False)

# Layers to use: content from conv4_2, style from conv1_1…conv5_1
CONTENT_LAYERS = {"21"}          # relu4_2
STYLE_LAYERS   = {"0","5","10","19","28"}  # relu1_1 … relu5_1

def extract_features(image):
    """Return a dict {layer_name: feature_map}."""
    features = {}
    x = image
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in CONTENT_LAYERS | STYLE_LAYERS:
            features[name] = x
    return features

# ── 3. Gram matrix for style loss ────────────────────────────────────────────
def gram_matrix(feat):
    _, C, H, W = feat.size()
    f = feat.view(C, H * W)
    return torch.mm(f, f.t()) / (C * H * W)

# ── 4. Optimise generated image ──────────────────────────────────────────────
generated = content_tensor.clone().requires_grad_(True)
optimizer = optim.LBFGS([generated], max_iter=200, lr=1.0)   # fast convergence

content_feats = extract_features(content_tensor)
style_feats   = extract_features(style_tensor)
style_grams   = {l: gram_matrix(style_feats[l]) for l in STYLE_LAYERS}

CONTENT_WEIGHT = 1.0
STYLE_WEIGHT   = 1e5
step = [0]

def closure():
    optimizer.zero_grad()
    gen_feats = extract_features(generated)

    # Content loss
    c_loss = torch.mean((gen_feats["21"] - content_feats["21"]) ** 2)

    # Style loss across all style layers
    s_loss = sum(
        torch.mean((gram_matrix(gen_feats[l]) - style_grams[l]) ** 2)
        for l in STYLE_LAYERS
    )

    total = CONTENT_WEIGHT * c_loss + STYLE_WEIGHT * s_loss
    total.backward()

    step[0] += 1
    if step[0] % 50 == 0:
        print(f"  step {step[0]:4d}  total_loss={total.item():.2f}")
    return total

print("Running style transfer (LBFGS, 200 steps) ...")
optimizer.step(closure)

# ── 5. Save result grid ───────────────────────────────────────────────────────
# Un-normalise for display (rough inverse of ImageNet norm)
def unnorm(t):
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    return (t.cpu() * std + mean)

resize_only = transforms.Resize((IMG_SIZE, IMG_SIZE))
content_display  = unnorm(resize_only(content_img_raw).unsqueeze(0))
style_display    = unnorm(resize_only(style_img_raw).unsqueeze(0))
generated_display = unnorm(generated.detach())

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
titles   = ["Content (CIFAR-10 img 0)", "Style (CIFAR-10 img 1)", "Stylised Result"]
displays = [content_display, style_display, generated_display]

for ax, title, img in zip(axes, titles, displays):
    ax.imshow(tensor_to_np(img))
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.suptitle("Neural Style Transfer – VGG-19 Content & Style Loss", fontsize=10)
plt.tight_layout()
plt.savefig("output_style_transfer.png", dpi=150)
plt.close()
print("Saved: output_style_transfer.png")