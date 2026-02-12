# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Image Gradients: Saliency Maps & Confusion Matrix
# Commit message: "Generated saliency maps for image visualization"
#
# Trains a small CNN on CIFAR-10 (5 epochs), then:
#   • Plots saliency maps for one image per class → output_saliency.png
#   • Plots a confusion matrix for the test set   → output_confusion_matrix.png
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
CLASSES   = ["airplane","automobile","bird","cat","deer",
             "dog","frog","horse","ship","truck"]
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS    = 5
BATCH     = 64

# ── 1. Data loaders ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_set = CIFAR10(root="./data", train=True,  download=True, transform=transform)
test_set  = CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=2)

# ── 2. Simple CNN ─────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── 3. Train ──────────────────────────────────────────────────────────────────
print("Training CNN on CIFAR-10 ...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}")

# ── 4. Evaluate & build confusion matrix ─────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        preds = model(imgs.to(DEVICE)).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

# Build confusion matrix manually
N = len(CLASSES)
cm = np.zeros((N, N), dtype=int)
for t, p in zip(all_labels, all_preds):
    cm[t][p] += 1

# --- Plot confusion matrix and save ---
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(N)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_yticks(range(N)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix – SimpleCNN on CIFAR-10 (5 epochs)", fontsize=11)
# Annotate cells
for i in range(N):
    for j in range(N):
        ax.text(j, i, cm[i,j], ha="center", va="center",
                color="white" if cm[i,j] > cm.max()*0.5 else "black", fontsize=7)
plt.tight_layout()
plt.savefig("output_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: output_confusion_matrix.png")

acc = 100 * sum(p==t for p,t in zip(all_preds,all_labels)) / len(all_labels)
print(f"Test accuracy: {acc:.1f}%")

# ── 5. Saliency maps ──────────────────────────────────────────────────────────
# One sample per class from the test set
raw_test = CIFAR10(root="./data", train=False, download=False,
                   transform=transforms.ToTensor())

sample_imgs, sample_labels = [], []
seen = set()
for img, lbl in raw_test:
    if lbl not in seen:
        sample_imgs.append(img); sample_labels.append(lbl); seen.add(lbl)
    if len(seen) == 10:
        break

# Compute saliency for each sample
fig, axes = plt.subplots(2, 10, figsize=(16, 4))
fig.suptitle("Top row: original  |  Bottom row: saliency map", fontsize=9)

norm_transform = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

for i, (img_t, lbl) in enumerate(zip(sample_imgs, sample_labels)):
    # Show original
    axes[0, i].imshow(img_t.permute(1,2,0).numpy())
    axes[0, i].set_title(CLASSES[lbl], fontsize=7)
    axes[0, i].axis("off")

    # Compute gradient w.r.t. input
    inp = norm_transform(img_t).unsqueeze(0).to(DEVICE)
    inp.requires_grad_()
    out = model(inp)
    model.zero_grad()
    out[0, lbl].backward()

    saliency = inp.grad.data.abs().squeeze().cpu()
    saliency = saliency.max(dim=0)[0]  # max over channels

    axes[1, i].imshow(saliency.numpy(), cmap="hot")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("output_saliency.png", dpi=150)
plt.close()
print("Saved: output_saliency.png")