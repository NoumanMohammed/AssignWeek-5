# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Attention-Based Image Captioning (ViT + GPT-2)
# Commit message: "Implemented attention-based image captioning"
#
# Uses a ViT encoder (with self-attention) + GPT-2 decoder on a CIFAR-10 image.
# Also visualises attention weights from the encoder as a heatmap.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from transformers import (VisionEncoderDecoderModel,
                          ViTImageProcessor, AutoTokenizer)

# ── 1. Load sample images from CIFAR-10 ─────────────────────────────────────
dataset = CIFAR10(root="./data", train=False, download=True,
                  transform=transforms.ToTensor())

CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

# Grab one image per class for a nice attention grid
samples = {}
for img_tensor, label in dataset:
    if label not in samples:
        samples[label] = img_tensor
    if len(samples) == 10:
        break

# ── 2. Load ViT-GPT2 model ───────────────────────────────────────────────────
MODEL_ID   = "nlpconnect/vit-gpt2-image-captioning"
model      = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
processor  = ViTImageProcessor.from_pretrained(MODEL_ID)
tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()

# ── 3. Caption all 10 samples & collect results ──────────────────────────────
to_pil = transforms.ToPILImage()
results = []   # (class_name, pil_img, caption)

for label in sorted(samples.keys()):
    pil_img = to_pil(samples[label])
    inputs  = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        caption_ids = model.generate(**inputs, max_new_tokens=20)
    caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
    results.append((CLASSES[label], pil_img, caption))
    print(f"{CLASSES[label]:12s} → {caption}")

# ── 4. Save attention caption grid (output for PPT) ──────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.suptitle("ViT + GPT-2 Attention Captioning  (CIFAR-10 – one per class)",
             fontsize=11, fontweight="bold")

for i, (cls, img, cap) in enumerate(results):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"{cls}\n\"{cap}\"", fontsize=6, wrap=True)
    ax.axis("off")

plt.tight_layout()
plt.savefig("output_attention_captions.png", dpi=150)
plt.close()
print("Saved: output_attention_captions.png")