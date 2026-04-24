# 🎥 Video Anomaly Detection using CNN-LSTM Architecture

Dự án này xây dựng một hệ thống **Deep Learning** để tự động phát hiện các hành vi bất thường (*anomaly detection*) trong video giám sát. Mô hình kết hợp giữa trích xuất đặc trưng hình ảnh và học chuỗi thời gian để hiểu được diễn biến hành động theo thời gian.

---

## 📌 Method Overview

Mô hình sử dụng kiến trúc **Hybrid CNN + RNN** gồm 2 giai đoạn chính:

### 🧠 Feature Extraction (CNN)

* Sử dụng **EfficientNetB0** *(hoặc ResNet50)* pre-trained trên ImageNet
* Chuyển mỗi frame video → vector đặc trưng (embedding ~512 chiều)
* Giúp giảm chiều dữ liệu nhưng vẫn giữ thông tin quan trọng

### 🔄 Sequence Learning (RNN)

* Sử dụng **Bidirectional LSTM**
* Học mối quan hệ thời gian giữa các frame
* Phát hiện hành vi bất thường dựa trên chuỗi hành động

---

## 🛠 Tech Stack

* **Ngôn ngữ:** Python
* **Framework:** TensorFlow / Keras
* **Thư viện:**

  * OpenCV – xử lý video
  * NumPy – tính toán ma trận
  * Scikit-learn – chia dataset
  * tqdm – hiển thị progress
* **Môi trường:** Google Colab / Jupyter Notebook

---

## 📂 Project Structure

```bash id="v9f2kd"
├── list_videos/        # Quét & gán nhãn video từ dataset
├── extract_frames/     # Trích xuất frame từ video
├── embed_video/        # Convert frame → embedding vector
├── make_seq/           # Tạo sequence (32 frames / sample)
├── build_lstm/         # Xây dựng mô hình BiLSTM
├── train.ipynb         # Notebook huấn luyện
└── model.keras         # Model sau khi train
```

---

## ⚙️ Data Pipeline

1. **Load video dataset**
2. **Extract frames** (có thể dùng step để giảm số lượng frame)
3. **Feature embedding** bằng CNN
4. **Chunk thành sequence** (32 frames / sequence)
5. **Train LSTM model**

---

## 🚀 Training Results

* Accuracy trên validation: **~99–100% sau ~20 epochs**
* Loss giảm ổn định → mô hình hội tụ tốt
* Phù hợp với dataset thử nghiệm (có thể cần kiểm chứng thêm với dữ liệu thực tế lớn hơn)

---

## 🔍 Usage

### ▶️ Training

1. Mount Google Drive (nếu dùng Colab)
2. Cập nhật đường dẫn dataset trong phần `Config`
3. Chạy notebook để huấn luyện
4. Model sẽ được lưu dưới dạng `.keras`

---

### 🎯 Inference

Sử dụng hàm `predict_video()` để dự đoán:

```python id="y7k2pz"
video_path = "path/to/your/video.mp4"
predict_video(video_path)
```

---

### 📊 Output

* **Score:** Xác suất bất thường
* **Label:**

  * `Normal`
  * `Anomaly`

### ⚠️ Threshold

* `score > 0.61` → **Anomaly**
* `score ≤ 0.61` → **Normal**

---

## 📌 Notes

* Dataset nhỏ có thể gây overfitting (accuracy cao bất thường)
* Nên bổ sung:

  * Data augmentation
  * Cross-validation
  * Test trên dữ liệu thực tế

---

## 🚀 Future Improvements

* Sử dụng **3D CNN (C3D / I3D)** thay cho CNN + LSTM
* Áp dụng **Transformer (Video Swin / TimeSformer)**
* Triển khai realtime (RTSP camera)
* Xây dựng dashboard giám sát

---


## Result
<img width="1569" height="780" alt="image" src="https://github.com/user-attachments/assets/debaa3de-9ecb-4005-9b8e-c1f6fd59532f" />

<img width="1567" height="788" alt="image" src="https://github.com/user-attachments/assets/391c1936-a76c-4367-9c24-1b848cf3c94f" />

