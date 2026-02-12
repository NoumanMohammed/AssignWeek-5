# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Image Gradients: Saliency Maps + 3 Visualisation Graphs
# Commit message: "Generated saliency maps for image visualization"
#
# Trains a small CNN on CIFAR-10 (5 epochs), then produces:
#   • output_saliency.png          – saliency heatmaps (one per class)
#   • output_training_curves.png   – training loss & accuracy over epochs
#   • output_tsne.png              – t-SNE of CNN feature embeddings
#   • output_perclass_accuracy.png – per-class accuracy bar chart
#
# NOTE: The entire script runs inside  if __name__ == '__main__':
#       This is REQUIRED on Windows to avoid multiprocessing errors.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# ── Constants (module level so the CNN class is always importable) ────────────
CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS  = 5
BATCH   = 64

# ── CNN defined at module level (required for Windows multiprocessing) ────────
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

    def get_embeddings(self, x):
        """Return 256-d feature vector (layer before final output)."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.classifier[1](self.classifier[0](x)))
        return x


# ── REQUIRED on Windows: all execution must be inside this guard ──────────────
if __name__ == '__main__':

    # ── 1. Data loaders (num_workers=0 avoids Windows spawn errors) ───────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    train_set    = CIFAR10(root="./data", train=True,  download=True, transform=transform)
    test_set     = CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=0)

    # ── 2. Model, loss, optimiser ─────────────────────────────────────────────
    model     = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ── 3. Train and record loss / accuracy per epoch ─────────────────────────
    print("Training CNN on CIFAR-10 ...")
    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(EPOCHS):
        # Training pass
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        # Validation pass
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                out = model(imgs.to(DEVICE))
                val_correct += (out.argmax(1) == labels.to(DEVICE)).sum().item()
                val_total   += labels.size(0)
        val_accs.append(100 * val_correct / val_total)

        print(f"  Epoch {epoch+1}/{EPOCHS}  "
              f"loss={train_losses[-1]:.4f}  "
              f"train_acc={train_accs[-1]:.1f}%  "
              f"val_acc={val_accs[-1]:.1f}%")

    # ── 4. Graph 1 – Training Loss & Accuracy Curves ──────────────────────────
    epochs_x = range(1, EPOCHS + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Training Curves – SimpleCNN on CIFAR-10",
                 fontsize=11, fontweight="bold")

    ax1.plot(epochs_x, train_losses, "o-", color="#e63946", linewidth=2, label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss per Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs_x, train_accs, "o-",  color="#2a9d8f", linewidth=2, label="Train Acc")
    ax2.plot(epochs_x, val_accs,   "s--", color="#e76f51", linewidth=2, label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy per Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("output_training_curves.png", dpi=150)
    plt.close()
    print("Saved: output_training_curves.png")

    # ── 5. Collect test predictions, labels, embeddings ───────────────────────
    model.eval()
    all_preds, all_labels, all_embeds = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            all_preds.extend(model(imgs).argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
            all_embeds.append(model.get_embeddings(imgs).cpu())

    all_embeds = torch.cat(all_embeds, dim=0).numpy()

    # ── 6. Graph 2 – Per-Class Accuracy Bar Chart ─────────────────────────────
    class_correct = np.zeros(10)
    class_total   = np.zeros(10)
    for t, p in zip(all_labels, all_preds):
        class_total[t]   += 1
        class_correct[t] += (t == p)

    class_acc = 100 * class_correct / class_total
    colors    = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASSES, class_acc, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Class Accuracy – SimpleCNN on CIFAR-10 (5 epochs)",
                 fontsize=11, fontweight="bold")
    ax.axhline(class_acc.mean(), color="black", linestyle="--",
               linewidth=1.2, label=f"Mean: {class_acc.mean():.1f}%")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("output_perclass_accuracy.png", dpi=150)
    plt.close()
    print("Saved: output_perclass_accuracy.png")

    # ── 7. Graph 3 – t-SNE Feature Embedding Plot ─────────────────────────────
    # Sub-sample 2000 points so t-SNE runs in ~30 seconds
    np.random.seed(42)
    idx      = np.random.choice(len(all_embeds), size=2000, replace=False)
    embeds_s = all_embeds[idx]
    labels_s = np.array(all_labels)[idx]

    print("Running t-SNE (this takes ~30 s) ...")
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeds_s)

    fig, ax = plt.subplots(figsize=(9, 8))
    for c in range(10):
        mask = labels_s == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=12, alpha=0.7, label=CLASSES[c], color=colors[c])
    ax.legend(markerscale=2, fontsize=9, loc="best")
    ax.set_title("t-SNE of CNN Feature Embeddings (CIFAR-10, 2 000 samples)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("output_tsne.png", dpi=150)
    plt.close()
    print("Saved: output_tsne.png")

    # ── 8. Saliency Maps – one image per class ────────────────────────────────
    raw_test = CIFAR10(root="./data", train=False, download=False,
                       transform=transforms.ToTensor())

    sample_imgs, sample_labels = [], []
    seen = set()
    for img, lbl in raw_test:
        if lbl not in seen:
            sample_imgs.append(img)
            sample_labels.append(lbl)
            seen.add(lbl)
        if len(seen) == 10:
            break

    norm_transform = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    fig, axes = plt.subplots(2, 10, figsize=(16, 4))
    fig.suptitle("Top row: original  |  Bottom row: saliency map", fontsize=9)

    for i, (img_t, lbl) in enumerate(zip(sample_imgs, sample_labels)):
        # Original image
        axes[0, i].imshow(img_t.permute(1,2,0).numpy())
        axes[0, i].set_title(CLASSES[lbl], fontsize=7)
        axes[0, i].axis("off")

        # Gradient w.r.t. input → saliency
        inp = norm_transform(img_t).unsqueeze(0).to(DEVICE)
        inp.requires_grad_()
        out = model(inp)
        model.zero_grad()
        out[0, lbl].backward()

        saliency = inp.grad.data.abs().squeeze().cpu()
        saliency = saliency.max(dim=0)[0]   # collapse channels

        axes[1, i].imshow(saliency.numpy(), cmap="hot")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("output_saliency.png", dpi=150)
    plt.close()
    print("Saved: output_saliency.png")