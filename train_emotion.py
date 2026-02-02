import argparse
import os
from pathlib import Path

from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms


class FolderEmotionDataset(Dataset):
    def __init__(self, root_dir, emotions, transform=None):
        self.root_dir = Path(root_dir)
        self.emotions = emotions
        self.transform = transform
        self.samples = self._gather_samples()

    def _gather_samples(self):
        exts = {".jpg", ".jpeg", ".png"}
        samples = []
        for idx, emotion in enumerate(self.emotions):
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                raise FileNotFoundError(f"Emotion folder not found: {emotion_dir}")
            for path in emotion_dir.rglob("*"):
                if path.suffix.lower() in exts:
                    samples.append((path, idx))
        if not samples:
            raise ValueError(f"No images found under: {self.root_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training_data")
    parser.add_argument(
        "--emotions",
        default="",
        help="Optional comma-separated folder names for emotions. "
        "If omitted, all subfolders under data-dir are used.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if args.emotions.strip():
        emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    else:
        emotions = sorted(
            [
                p.name
                for p in Path(args.data_dir).iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
        )
        if not emotions:
            raise ValueError(f"No emotion subfolders found in: {args.data_dir}")
    dataset = FolderEmotionDataset(args.data_dir, emotions, transform=transform)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(emotions)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / "emotion_multi.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": emotions,
        },
        output_path,
    )
    print(f"Saved model to {output_path}")
    print(f"Trained on {len(emotions)} emotions: {', '.join(emotions)}")


if __name__ == "__main__":
    main()
