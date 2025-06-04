import os
import cv2
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# ================================
# 1️⃣ BÖLÜM: Dataset klasör yolları
# ================================
dataset_dir = r"C:\Users\furka\PycharmProjects\PythonProjectweedD\veriseti_weed.v6i.yolov8"
output_dir = r"C:\Users\furka\PycharmProjects\PythonProjectweedD\classification_data"
class_names = {0: "crop", 1: "weed"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder(os.path.join(output_dir, "train"), transform=transform)
val_set = datasets.ImageFolder(os.path.join(output_dir, "valid"), transform=transform)
test_set = datasets.ImageFolder(os.path.join(output_dir, "test"), transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# ================================
# 2️⃣ BÖLÜM: VGGNet Modeli (VGG16)
# ================================
print("\n🚀 VGG16 eğitimi başlatılıyor...")

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.classifier[6] = nn.Linear(4096, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 3
all_preds = []
all_labels = []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct_train = 0
    total_train = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_acc = 100 * correct_train / total_train
    print(f"[{epoch+1}] Train Accuracy: {train_acc:.2f}%")

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    preds_epoch = []
    labels_epoch = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

            preds_epoch.extend(predicted.cpu().numpy())
            labels_epoch.extend(labels.cpu().numpy())

    val_acc = 100 * correct_val / total_val
    print(f"[{epoch+1}] Validation Accuracy: {val_acc:.2f}%")

    all_preds.extend(preds_epoch)
    all_labels.extend(labels_epoch)

# ================================
# 3️⃣ BÖLÜM: Ortalama Skorlar
# ================================
print("\n📊 Ortalama Skorlar (Validation seti):")
print(f"Precision: {precision_score(all_labels, all_preds, average='macro'):.4f}")
print(f"Recall:    {recall_score(all_labels, all_preds, average='macro'):.4f}")
print(f"F1 Score:  {f1_score(all_labels, all_preds, average='macro'):.4f}")

# ================================
# 4️⃣ BÖLÜM: Test Seti Accuracy
# ================================
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

test_acc = 100 * correct_test / total_test
print(f"\n🧪 Test Set Accuracy: {test_acc:.2f}%")
print("✅ VGG16 eğitimi tamamlandı.")
