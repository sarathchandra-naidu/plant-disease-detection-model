import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DATA_DIR = "data/PlantVillage-Dataset/raw/color"


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
    DATA_DIR
)

classes = full_dataset.classes

num_classes = len(classes)

print("Number of images:", len(full_dataset))
print("Number of classes:", num_classes)

print(classes)


class PlantDiseaseDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


all_samples = full_dataset.samples


random.shuffle(all_samples)


total = len(all_samples)

train_end = int(0.70 * total)

val_end = int(0.85 * total)


train_samples = all_samples[:train_end]

val_samples = all_samples[
    train_end:val_end
]

test_samples = all_samples[
    val_end:
]


train_dataset = PlantDiseaseDataset(
    train_samples,
    transform=train_transform
)

val_dataset = PlantDiseaseDataset(
    val_samples,
    transform=val_transform
)

test_dataset = PlantDiseaseDataset(
    test_samples,
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
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


model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(
        model.fc.in_features,
        num_classes
    )
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


epochs = 20

best_val_loss = float("inf")

early_stopping_patience = 4

early_stopping_counter = 0


for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for Batch_X, Batch_Y in train_loader:

        Batch_X = Batch_X.to(device)

        Batch_Y = Batch_Y.to(device)


        y_predict = model(Batch_X)


        loss = criterion(
            y_predict,
            Batch_Y
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        predictions = y_predict.argmax(
            dim=1
        )


        correct += (
            predictions == Batch_Y
        ).sum().item()


        total += Batch_Y.size(0)


    train_loss = (
        running_loss /
        len(train_loader)
    )


    train_accuracy = (
        correct /
        total
    )


    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for Batch_X, Batch_Y in val_loader:

            Batch_X = Batch_X.to(device)

            Batch_Y = Batch_Y.to(device)


            y_predict = model(Batch_X)


            loss = criterion(
                y_predict,
                Batch_Y
            )


            val_running_loss += loss.item()


            predictions = y_predict.argmax(
                dim=1
            )


            val_correct += (
                predictions == Batch_Y
            ).sum().item()


            val_total += Batch_Y.size(0)


    val_loss = (
        val_running_loss /
        len(val_loader)
    )


    val_accuracy = (
        val_correct /
        val_total
    )


    scheduler.step(val_loss)


    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy * 100:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy * 100:.2f}% "
        f"LR: {current_lr:.6f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss

        early_stopping_counter = 0

        torch.save(
            model.state_dict(),
            "best_plant_disease_model.pth"
        )

        print("Best model saved.")


    else:

        early_stopping_counter += 1

        print(
            "Early stopping counter:",
            early_stopping_counter
        )


        if (
            early_stopping_counter
            >= early_stopping_patience
        ):

            print("Early stopping.")

            break


model.load_state_dict(
    torch.load(
        "best_plant_disease_model.pth",
        map_location=device
    )
)


model.eval()


all_predictions = []

all_labels = []

test_running_loss = 0.0


with torch.no_grad():

    for Batch_X, Batch_Y in test_loader:

        Batch_X = Batch_X.to(device)

        Batch_Y = Batch_Y.to(device)


        y_predict = model(Batch_X)


        loss = criterion(
            y_predict,
            Batch_Y
        )


        test_running_loss += loss.item()


        predictions = y_predict.argmax(
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_labels.extend(
            Batch_Y.cpu().numpy()
        )


test_loss = (
    test_running_loss /
    len(test_loader)
)


accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


print("\nFinal Test Results")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall: {recall * 100:.2f}%"
)

print(
    f"F1 Score: {f1 * 100:.2f}%"
)


print("\nClassification Report")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        zero_division=0
    )
)


cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix")

print(cm)