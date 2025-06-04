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
# 1️⃣ BÖLÜM: Bounding Box Kırpma
# ================================
dataset_dir = r"C:\Users\furka\PycharmProjects\PythonProjectweedD\veriseti_weed.v6i.yolov8"
output_dir = r"C:\Users\furka\PycharmProjects\PythonProjectweedD\classification_data"
class_names = {0: "crop", 1: "weed"}
splits = ["train", "valid", "test"]

for split in splits:
    img_dir = os.path.join(dataset_dir, split, "images")
    lbl_dir = os.path.join(dataset_dir, split, "labels")

    for label_file in tqdm(os.listdir(lbl_dir), desc=f"{split}"):
        if not label_file.endswith(".txt"):
            continue

        image_file = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(img_dir, image_file)
        label_path = os.path.join(lbl_dir, label_file)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            class_name = class_names[class_id]
            dest_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            output_filename = f"{os.path.splitext(image_file)[0]}_{idx}.jpg"
            output_path = os.path.join(dest_dir, output_filename)
            cv2.imwrite(output_path, cropped)

print("✅ Sınıflandırma verisi hazırlandı (crop vs weed).")

# ================================
# 2️⃣ BÖLÜM: AlexNet Eğitimi
# ================================
print("\n🚀 AlexNet eğitimi başlatılıyor...")

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

model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
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

    # Validation Accuracy
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

    # Skorları sakla
    all_preds.extend(preds_epoch)
    all_labels.extend(labels_epoch)

# ================================
# 3️⃣ BÖLÜM: Sonuç Raporu
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
print("✅ Eğitim ve değerlendirme tamamlandı.")
