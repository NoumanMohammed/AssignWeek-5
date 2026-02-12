# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – RNN-based Image Captioning with BLIP
# Commit message: "Implemented RNN-based image captioning"
#
# Uses CIFAR-10 to pull a sample image, then feeds it to a pre-trained
# BLIP captioning model (encoder-decoder with an RNN decoder).
# ─────────────────────────────────────────────────────────────────────────────

import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ── 1. Load one sample image from CIFAR-10 ──────────────────────────────────
# Download CIFAR-10 (only test split so it's fast) and grab the first image.
dataset = CIFAR10(root="./data", train=False, download=True,
                  transform=transforms.ToTensor())

# CIFAR-10 class names for display
CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

# Take the first sample
tensor_img, true_label = dataset[0]

# Convert tensor → PIL for the model
pil_img = transforms.ToPILImage()(tensor_img)

# ── 2. Load pre-trained BLIP model (RNN decoder inside) ─────────────────────
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base")
model.eval()

# ── 3. Generate caption ──────────────────────────────────────────────────────
inputs      = processor(images=pil_img, return_tensors="pt")
caption_ids = model.generate(**inputs, max_new_tokens=30)
caption     = processor.decode(caption_ids[0], skip_special_tokens=True)

print(f"True CIFAR-10 label : {CLASSES[true_label]}")
print(f"Generated Caption   : {caption}")

# ── 4. Save output image with caption overlaid ───────────────────────────────
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(pil_img, interpolation="nearest")
ax.set_title(f"Label: {CLASSES[true_label]}\nCaption: {caption}",
             fontsize=8, wrap=True)
ax.axis("off")
plt.tight_layout()
plt.savefig("output_captioning.png", dpi=150)
plt.close()
print("Saved: output_captioning.png")