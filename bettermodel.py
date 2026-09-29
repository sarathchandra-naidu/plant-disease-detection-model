import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

full_dataset = datasets.ImageFolder(
    "Data/color",
    transform=train_transform
)

num_classes = len(full_dataset.classes)

print("Number of images:", len(full_dataset))
print("Number of classes:", num_classes)
print("Classes:", full_dataset.classes)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2
)

epochs = 15

best_val_loss = float("inf")
patience = 4
counter = 0

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for Batch_X, Batch_Y in train_loader:

        Batch_X = Batch_X.to(device)
        Batch_Y = Batch_Y.to(device)

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

            Batch_X = Batch_X.to(device)
            Batch_Y = Batch_Y.to(device)

            y_predict = model(Batch_X)

            loss = criterion(y_predict, Batch_Y)

            val_loss += loss.item()

            predictions = y_predict.argmax(dim=1)

            correct += (predictions == Batch_Y).sum().item()

            total += Batch_Y.size(0)

    val_loss = val_loss / len(val_loader)

    val_accuracy = correct / total

    scheduler.step(val_loss)

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
            "best_plant_disease_model.pth"
        )

    else:

        counter += 1

        if counter >= patience:

            print("Early stopping")
            break

model.load_state_dict(
    torch.load(
        "best_plant_disease_model.pth",
        map_location=device
    )
)

model.eval()

test_loss = 0.0
correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for Batch_X, Batch_Y in test_loader:

        Batch_X = Batch_X.to(device)
        Batch_Y = Batch_Y.to(device)

        y_predict = model(Batch_X)

        loss = criterion(y_predict, Batch_Y)

        test_loss += loss.item()

        predictions = y_predict.argmax(dim=1)

        correct += (predictions == Batch_Y).sum().item()

        total += Batch_Y.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            Batch_Y.cpu().numpy()
        )

test_loss = test_loss / len(test_loader)

test_accuracy = correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")