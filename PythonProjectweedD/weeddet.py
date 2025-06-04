import os
import cv2
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import yaml  # Güvenilir YAML ayrıştırma için

# Veriseti dizini ve data.yaml dosyası
data_dir = r"C:\Users\furka\PycharmProjects\PythonProjectweedD\veriseti_weed.v6i.yolov8"
data_yaml_path = os.path.join(data_dir, "data.yaml")


def load_class_names(data_yaml_path):
    """data.yaml dosyasından sınıf isimlerini yükler."""
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"{data_yaml_path} bulunamadı.")

    with open(data_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    if 'names' not in yaml_data:
        raise ValueError("data.yaml içinde 'names' anahtarı bulunamadı.")

    return yaml_data['names']


def analyze_data(data_dir):
    labels_dir = os.path.join(data_dir, "train", "labels")
    images_dir = os.path.join(data_dir, "train", "images")

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    print(f"Toplam Görüntü Sayısı: {len(image_files)}")
    print(f"Toplam Etiket Dosyası: {len(label_files)}")

    class_counts = {}
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

    print("Sınıf Dağılımı:")
    for cls, count in class_counts.items():
        print(f"Sınıf {cls}: {count} örnek")

    return class_counts


def prepare_data(data_dir):
    labels_dir = os.path.join(data_dir, "train", "labels")
    images_dir = os.path.join(data_dir, "train", "images")

    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".jpg")]
    train_images, val_images = train_test_split(image_paths, test_size=0.2, random_state=42)

    with open(os.path.join(data_dir, "train.txt"), "w") as f:
        for path in train_images:
            f.write(path + "\n")

    with open(os.path.join(data_dir, "val.txt"), "w") as f:
        for path in val_images:
            f.write(path + "\n")

    print("Veri bölme tamamlandı!")
    print(f"Eğitim seti: {len(train_images)} görüntü")
    print(f"Doğrulama seti: {len(val_images)} görüntü")


def train_model(data_yaml_path):
    model = YOLO('yolov8n.pt')
    model.train(
        data=data_yaml_path,
        epochs=12,
        imgsz=640,
        batch=16,
        name="weed_detection",
        save_dir="D:/yolo_outputs"
    )
    print("Eğitim tamamlandı!")
    return model


def detect_on_image(image_path, model, class_names, confidence_threshold=0.58):
    if not os.path.exists(image_path):
        print(f"Hata: Resim bulunamadı: {image_path}")
        return

    image = cv2.imread(image_path)
    results = model.predict(source=image, save=False, conf=confidence_threshold)

    for result in results[0].boxes:
        cls_id = int(result.cls[0])
        confidence = result.conf[0]

        # Eşik değerinin altında olanları atla
        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Çerçeve ve metin rengi
        if class_names[cls_id].lower() == "crop":
            color = (0, 255, 0)  # Yeşil
        elif class_names[cls_id].lower() == "weed":
            color = (0, 0, 255)  # Kırmızı
        else:
            color = (255, 255, 255)  # Beyaz (diğer durumlar için)

        label = f"{class_names[cls_id]}: {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Merkezi nokta
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)

    output_image_path = os.path.join(os.path.dirname(image_path), "detected_image.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Sonuç görüntüsü kaydedildi: {output_image_path}")
    cv2.imshow("Detected Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_on_video(video_path, model, class_names, confidence_threshold=0.58):
    if not os.path.exists(video_path):
        print(f"Hata: Video bulunamadı: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video açılamadı!")
        return

    output_video_path = os.path.join(os.path.dirname(video_path), "detected_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, conf=confidence_threshold)

        for result in results[0].boxes:
            cls_id = int(result.cls[0])
            confidence = result.conf[0]

            # Eşik değerinin altında olanları atla
            if confidence < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, result.xyxy[0])

            # Çerçeve ve metin rengi
            if class_names[cls_id].lower() == "crop":
                color = (0, 255, 0)  # Yeşil
            elif class_names[cls_id].lower() == "weed":
                color = (0, 0, 255)  # Kırmızı
            else:
                color = (255, 255, 255)  # Beyaz (diğer durumlar için)

            label = f"{class_names[cls_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Merkezi nokta
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

        out.write(frame)
        cv2.imshow("Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Sonuç videosu kaydedildi: {output_video_path}")


if __name__ == "__main__":
    print("1. Veri Analizi ve Hazırlık")
    analyze_data(data_dir)
    prepare_data(data_dir)

    print("\n2. YOLO Model Eğitimi")
    trained_model = train_model(data_yaml_path)

    print("\n3. Tespit Yapma")
    class_names = load_class_names(data_yaml_path)

    while True:
        user_input = input("Resim dosya yolunu girin (çıkmak için 'q' yazın): ").strip()

        if user_input.lower() == 'q':
            print("Program sonlandırılıyor.")
            break
        elif user_input.lower().endswith(('.jpg', '.png', '.jpeg')):
            detect_on_image(user_input, trained_model, class_names)
        else:
            print("Geçersiz dosya türü! Lütfen bir resim (.jpg, .png) dosya yolu girin.")
