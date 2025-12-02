# BÁO CÁO NGHIÊN CỨU: TỔNG QUAN VỀ BÀI TOÁN TEXT-TO-SPEECH (TTS)

---

## 1. Tổng quan & Tình hình nghiên cứu hiện tại

### 1.1. Bối cảnh
Bài toán Tổng hợp tiếng nói (Text-To-Speech - TTS) không còn dừng lại ở việc chuyển đổi văn bản thành âm thanh đơn thuần. Với sự bùng nổ của Deep Learning và Generative AI, mục tiêu của TTS hiện đại là đạt được sự tự nhiên (naturalness) ngang bằng con người, khả năng biểu đạt cảm xúc (expressiveness) và khả năng sao chép giọng nói (voice cloning) chỉ từ một mẫu dữ liệu cực ngắn.

### 1.2. Xu hướng phát triển
Nghiên cứu TTS đang dịch chuyển qua các giai đoạn:
* **Giai đoạn Truyền thống:** Dựa trên xử lý tín hiệu số và ghép nối từ điển âm vị.
* **Giai đoạn Deep Learning (End-to-End):** Sử dụng Neural Networks để học ánh xạ từ văn bản sang biểu đồ phổ (Mel-spectrogram).
* **Giai đoạn Generative AI (Hiện tại):** Coi TTS là bài toán mô hình ngôn ngữ (Language Modeling) hoặc sử dụng Diffusion Models để sinh âm thanh đa dạng, zero-shot (không cần huấn luyện lại).

---

## 2. Phân tích các hướng triển khai (Levels)

Dưới đây là phân tích chi tiết 3 cấp độ công nghệ TTS dựa trên cơ chế, ưu nhược điểm và ứng dụng thực tế.

### Level 1: Công nghệ ghép nối & Luật (Concatenative / Rule-based)
*Đại diện: Các hệ thống đọc màn hình cũ, giọng "chị Google" đời đầu.*

* **Cơ chế:** Hệ thống lưu trữ một cơ sở dữ liệu khổng lồ các đoạn âm thanh nhỏ (từ, âm tiết, âm vị) được thu âm sẵn. Khi có văn bản, hệ thống dùng thuật toán (Unit Selection) để tìm và ghép các mảnh nhỏ này lại theo luật ngôn ngữ.
* **Đánh giá:**

| Tiêu chí | Nội dung chi tiết |
| :--- | :--- |
| **Ưu điểm** | - **Tốc độ cực nhanh:** Độ trễ thấp, phù hợp real-time.<br>- **Nhẹ:** Chạy tốt trên CPU yếu, thiết bị nhúng.<br>- **Chính xác:** Phát âm chuẩn xác theo từ điển đã nạp. |
| **Nhược điểm** | - **Thiếu tự nhiên:** Giọng nói bị "giật cục", không có ngữ điệu liền mạch.<br>- **Khó sửa đổi:** Muốn đổi giọng phải thu âm lại toàn bộ từ điển. |
| **Trường hợp sử dụng** | Các hệ thống thông báo công cộng, thiết bị IoT cấu hình thấp, công cụ hỗ trợ người khiếm thị. |

### Level 2: Deep Learning Cá nhân hóa (Personalized / Single-Speaker Model)
*Đại diện: Tacotron 2, FastSpeech kết hợp Vocoder (HiFi-GAN).*

* **Cơ chế:** Mô hình được huấn luyện để chuyển đổi văn bản thành phổ âm thanh (Mel-spectrogram). Để có giọng của một người cụ thể, cần một bộ dữ liệu sạch của người đó để thực hiện **Fine-tuning** (tinh chỉnh).
* **Đánh giá:**

| Tiêu chí | Nội dung chi tiết |
| :--- | :--- |
| **Ưu điểm** | - **Độ tự nhiên cao:** Giọng mượt mà, ngữ điệu tốt hơn hẳn Level 1.<br>- **Hiệu suất ổn định:** Mô hình nhỏ gọn hơn so với các siêu mô hình (Level 3). |
| **Nhược điểm** | - **Phụ thuộc dữ liệu:** Cần thu âm lượng lớn dữ liệu chất lượng cao cho *từng* giọng muốn tạo.<br>- **Kém linh hoạt:** Khó tạo ra giọng mới ngay lập tức (cần thời gian train). |
| **Trường hợp sử dụng** | Trợ lý ảo (Siri, Alexa), Sách nói độc quyền, NPC trong Game có lời thoại cố định. |

### Level 3: Few-shot / Zero-shot Generative AI
*Đại diện: VALL-E, XTTS, Voicebox.*

* **Cơ chế:** Sử dụng kiến trúc Transformer hoặc Diffusion quy mô lớn, học trên hàng chục nghìn giờ âm thanh đa ngôn ngữ. Mô hình có thể trích xuất đặc trưng giọng nói (Speaker embedding) từ **3-5 giây** âm thanh mẫu và áp dụng nó để đọc văn bản mới mà không cần huấn luyện lại.
* **Đánh giá:**

| Tiêu chí | Nội dung chi tiết |
| :--- | :--- |
| **Ưu điểm** | - **Sao chép giọng tức thì:** Chỉ cần vài giây mẫu (Few-shot).<br>- **Đa cảm xúc:** Có thể mô phỏng tiếng cười, ngập ngừng, thì thầm.<br>- **Đa ngôn ngữ:** Một giọng mẫu có thể nói được nhiều thứ tiếng. |
| **Nhược điểm** | - **Tốn tài nguyên:** Yêu cầu GPU mạnh để suy luận (Inference).<br>- **Độ trễ cao:** Khó ứng dụng cho hội thoại thời gian thực nếu chưa tối ưu.<br>- **Rủi ro:** Dễ bị lạm dụng cho Deepfake. |
| **Trường hợp sử dụng** | Sáng tạo nội dung (Youtube/Tiktok), Lồng tiếng phim tự động, Cá nhân hóa trải nghiệm người dùng quy mô lớn. |

---

## 3. Tối ưu hóa Pipeline: Giải pháp khắc phục nhược điểm

Để tối đa hóa ưu điểm và giảm thiểu hạn chế của các Level trên, các nghiên cứu hiện tại tập trung vào các pipeline sau:

### 3.1. Tăng tốc độ & Giảm độ trễ (Cho Level 2 & 3)
* **Vấn đề:** Các mô hình Deep Learning thường chậm do cơ chế sinh tuần tự (Autoregressive - sinh từng phần nhỏ nối tiếp nhau).
* **Giải pháp:** Chuyển sang kiến trúc **Non-autoregressive** (như FastSpeech 2).
    * *Cách làm:* Mô hình dự đoán thời lượng (duration) của từng từ và sinh ra toàn bộ câu nói cùng một lúc (song song hóa).
    * *Kết quả:* Tốc độ nhanh hơn gấp hàng chục lần, tiệm cận tốc độ Level 1 nhưng chất lượng Level 2.
* **Streaming Inference:** Kỹ thuật sinh âm thanh ngay khi văn bản đang được nạp vào (theo dạng dòng chảy) thay vì chờ nạp hết câu.

### 3.2. Kiểm soát cảm xúc & Ngữ điệu (Cho Level 2)
* **Vấn đề:** Model Level 2 thường đọc đều đều, thiếu cảm xúc.
* **Giải pháp:** Tích hợp **Variance Adaptor**.
    * *Cách làm:* Thêm các module dự đoán và điều chỉnh Cao độ (Pitch), Năng lượng (Energy/Volume) vào giữa pipeline.
    * *Kết quả:* Người dùng có thể can thiệp thủ công để làm câu nói to hơn, trầm bổng hơn hoặc buồn/vui tùy ý.

### 3.3. Tối ưu dữ liệu & Đa ngôn ngữ (Cho Level 3)
* **Vấn đề:** Level 3 cần dữ liệu khổng lồ để học được đặc trưng giọng nói tổng quát.
* **Giải pháp:** Sử dụng **Self-supervised Learning** (Học tự giám sát).
    * *Cách làm:* Pre-train mô hình trên lượng lớn dữ liệu âm thanh không cần nhãn (như Wav2Vec) để model "hiểu" cấu trúc âm thanh trước khi học TTS.
    * *Kết quả:* Giúp model học được giọng mới nhanh hơn và hỗ trợ các ngôn ngữ hiếm dữ liệu (Low-resource languages).

---

## 4. Đạo đức nghiên cứu: Watermarking

Với sự phát triển của Level 3 (Deepfake Voice), vấn đề đạo đức trở nên cấp thiết.
* **Giải pháp:** Nhúng Watermark (Thủy vân số) vào âm thanh đầu ra.
* **Cơ chế:** Pipeline sẽ chèn một tín hiệu nhiễu cực nhỏ, không thể nghe thấy bởi tai người (imperceptible) nhưng máy có thể phát hiện.
* **Mục đích:** Để phân biệt đâu là giọng người thật, đâu là giọng AI tạo ra. Tín hiệu này phải bền vững (Robust) ngay cả khi file âm thanh bị nén, cắt ghép hoặc phát qua loa ngoài.
* **Ví dụ:** Công nghệ AudioSynthID của Google DeepMind.

---

## 5. Kết luận
Việc lựa chọn phương pháp triển khai phụ thuộc hoàn toàn vào bài toán kinh tế và kỹ thuật:
* Nếu cần **giá rẻ, chạy trên chip nhỏ**: Chọn **Level 1**.
* Nếu cần **thương hiệu riêng, ổn định, chất lượng cao**: Chọn **Level 2** (Fine-tune).
* Nếu cần **tính năng sáng tạo, linh hoạt, sao chép giọng nhanh**: Chọn **Level 3** (Generative AI).

Tương lai của TTS sẽ là sự kết hợp: Chất lượng của Generative AI (Level 3) nhưng được tối ưu hóa để chạy nhẹ và nhanh như Level 1 (Distillation/Quantization).
