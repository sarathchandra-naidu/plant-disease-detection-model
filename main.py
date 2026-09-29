import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    "data/color",
    transform=transform
)

print("Number of images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("Classes:", dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

class PlantDiseaseCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, X):
        out = self.features(X)
        out = self.classifier(out)
        return out


num_classes = len(dataset.classes)

model = PlantDiseaseCNN(num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

best_val_loss = float("inf")
patience = 3
counter = 0

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for Batch_X, Batch_Y in train_loader:

        y_predict = model(Batch_X)

        loss = criterion(y_predict, Batch_Y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = y_predict.argmax(dim=1)

        correct += (predictions == Batch_Y).sum().item()

        total += Batch_Y.size(0)

    train_loss = running_loss / len(train_loader)

    train_accuracy = correct / total

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for Batch_X, Batch_Y in val_loader:

            y_predict = model(Batch_X)

            loss = criterion(y_predict, Batch_Y)

            val_loss += loss.item()

            predictions = y_predict.argmax(dim=1)

            correct += (predictions == Batch_Y).sum().item()

            total += Batch_Y.size(0)

    val_loss = val_loss / len(val_loader)

    val_accuracy = correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy*100:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy*100:.2f}%"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

    else:

        counter += 1

        if counter >= patience:

            print("Early stopping")
            break


model.load_state_dict(
    torch.load("best_model.pth")
)

model.eval()

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():

    for Batch_X, Batch_Y in val_loader:

        y_predict = model(Batch_X)

        loss = criterion(y_predict, Batch_Y)

        test_loss += loss.item()

        predictions = y_predict.argmax(dim=1)

        correct += (predictions == Batch_Y).sum().item()

        total += Batch_Y.size(0)

test_loss = test_loss / len(val_loader)

test_accuracy = correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")