
# BÁO CÁO LAB 5: LÀM QUEN VỚI PYTORCH VÀ CÁC MÔ HÌNH CHUỖI (RNN/LSTM)

---

## MỤC LỤC
1. [Phần 1: Cơ bản về PyTorch](#phần-1-cơ-bản-về-pytorch)
2. [Phần 2: Phân loại văn bản (Text Classification)](#phần-2-phân-loại-văn-bản-text-classification)
3. [Phần 3: Gán nhãn từ loại (POS Tagging)](#phần-3-gán-nhãn-từ-loại-pos-tagging)
4. [Phần 4: Nhận diện thực thể tên riêng (NER)](#phần-4-nhận-diện-thực-thể-tên-riêng-ner)
5. [Hướng dẫn chạy code](#hướng-dẫn-chạy-code)

---

## PHẦN 1: CƠ BẢN VỀ PYTORCH

### 1. Giải thích Code & Thực hiện
Mục tiêu là làm quen với các khái niệm cốt lõi của PyTorch: Tensor, Autograd và `nn.Module`.

* **Thao tác với Tensor:**
    * Khởi tạo tensor từ List và NumPy array.
    * Sử dụng các hàm tạo sẵn: `torch.ones_like`, `torch.rand_like`.
    * Thực hiện các phép toán đại số: cộng, nhân element-wise, nhân ma trận (`@`).
    * Thay đổi kích thước tensor bằng `view` (tương tự reshape).
* **Autograd (Tự động tính đạo hàm):**
    * Sử dụng `requires_grad=True` để theo dõi lịch sử tính toán.
    * Ví dụ hàm $z = 3(x+2)^2$. Với $x=1$, kết quả forward là 27, đạo hàm $dz/dx = 6(x+2) = 18$.
* **Xây dựng Neural Network (`torch.nn`):**
    * Tạo một lớp `MyFirstModel` kế thừa từ `nn.Module`.
    * Kết hợp các lớp: `nn.Embedding` -> `nn.Linear` -> `nn.ReLU` -> Output Linear.

### 2. Khó khăn và Giải pháp
* **Vấn đề:** Lỗi `RuntimeError: Trying to backward through the graph a second time`.
* **Nguyên nhân:** PyTorch giải phóng đồ thị tính toán (computational graph) ngay sau khi gọi `.backward()` để tiết kiệm bộ nhớ. Việc gọi lại lần 2 trên đồ thị đã bị hủy sẽ gây lỗi.
* **Giải pháp:**
    1.  Dùng `backward(retain_graph=True)` nếu thực sự cần giữ đồ thị (không khuyến khích vì tốn bộ nhớ).
    2.  **Khuyến nghị:** Thực hiện lại quá trình forward (recompute) để tạo đồ thị mới cho lần backward tiếp theo.

---

## PHẦN 2: PHÂN LOẠI VĂN BẢN (TEXT CLASSIFICATION)

### 1. Giải thích Code
Bài toán phân loại ý định (Intent Classification) trên tập dữ liệu HWU64. Đã thực hiện so sánh 4 mô hình:

1.  **TF-IDF + Logistic Regression:** Sử dụng `TfidfVectorizer` (max features=5000) kết hợp `LogisticRegression`.
2.  **Word2Vec (Average) + Dense:** Dùng `gensim` train Word2Vec, lấy trung bình vector các từ trong câu làm input cho mạng Neural đơn giản (Dense layer).
3.  **Embedding (Pre-trained) + LSTM:** Sử dụng trọng số từ Word2Vec đưa vào lớp `nn.Embedding` (đóng băng - `trainable=False`) kết hợp với LSTM.
4.  **Embedding (Scratch) + LSTM:** Lớp Embedding được khởi tạo ngẫu nhiên và huấn luyện cùng với mạng LSTM (`trainable=True`).

*Xử lý dữ liệu chuỗi:* Sử dụng `Tokenizer` để chuyển từ sang số và `pad_sequences` để đồng bộ độ dài câu (max_len=50).

### 2. Phân tích kết quả
**Bảng tổng hợp F1-score (Macro Avg) trên tập Test:**

| Mô hình | F1-score | Test Loss | Nhận xét |
| :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | **0.8200** | N/A | Hiệu suất tốt nhất bất ngờ. |
| **Embedding (Scratch) + LSTM** | 0.6192 | 1.8378 | Tốt nhất trong nhóm Deep Learning. |
| **Embedding (Pre-trained) + LSTM** | 0.3893 | 2.0520 | Hiệu suất thấp, có thể do embedding không phù hợp domain. |
| **Word2Vec (Avg) + Dense** | 0.3514 | 2.2898 | Kém nhất do mất thông tin thứ tự từ. |

**Phân tích sâu:**
* **Nghịch lý:** Mô hình đơn giản (TF-IDF) lại chiến thắng áp đảo LSTM.
* **Nguyên nhân:**
    * **Dữ liệu:** Tập dữ liệu có thể chứa các từ khóa đặc trưng mạnh (keywords) giúp phân loại dễ dàng mà không cần hiểu sâu ngữ cảnh phức tạp. TF-IDF bắt keyword rất tốt.
    * **Khả năng của LSTM:** LSTM gặp khó khăn với các câu chứa phủ định hoặc cấu trúc phức tạp (Ví dụ: *"I don't want to set an alarm..."* -> Model vẫn đoán là `alarm_set` thay vì `alarm_remove`).
    * **Word Averaging:** Việc lấy trung bình cộng vector làm mất hoàn toàn thông tin về thứ tự từ, dẫn đến kết quả kém nhất.

---

## PHẦN 3: GÁN NHÃN TỪ LOẠI (POS TAGGING)

### 1. Giải thích Code
Xây dựng mô hình gán nhãn từ loại (Part-of-Speech) cho từng từ trong câu sử dụng dữ liệu Universal Dependencies.

* **Xử lý dữ liệu:**
    * Đọc file `.conllu`, xây dựng từ điển `word_to_ix` và `tag_to_ix`.
    * **Padding:** Sử dụng hàm `collate_fn` tùy chỉnh với `pad_sequence` để đưa các câu trong batch về cùng độ dài.
* **Mô hình (`SimpleRNNForTokenClassification`):**
    * Kiến trúc: Embedding -> RNN (batch_first=True) -> Linear (Fully Connected).
* **Huấn luyện:**
    * Loss Function: `CrossEntropyLoss` với tham số `ignore_index=0` (để mô hình không tính lỗi tại các vị trí padding).
    * Optimizer: Adam.

### 2. Phân tích kết quả
Quá trình huấn luyện qua 20 epochs cho thấy sự hội tụ ổn định:

* **Loss:** Giảm mạnh từ 1.24 (epoch 1) xuống 0.10 (epoch 20).
* **Accuracy (Dev set):** Tăng dần và đạt đỉnh ổn định.
    * Epoch 1: 74.75%
    * Epoch 10: 88.54%
    * **Epoch 20 (Final): 88.58%**

**Ví dụ thực tế:**
* Input: `"I love NLP"`
* Prediction: `[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'PROPN')]`
-> Mô hình nhận diện chính xác Đại từ, Động từ và Danh từ riêng.

---

## PHẦN 4: NHẬN DIỆN THỰC THỂ TÊN RIÊNG (NER)

### 1. Giải thích Code
Bài toán nhận diện thực thể (Person, Organization, Location...) trên tập dữ liệu chuẩn CoNLL-2003.

* **Mô hình (`NERModel`):** Sử dụng kiến trúc **LSTM** thay vì RNN thường để xử lý sự phụ thuộc xa tốt hơn.
    * Embedding -> LSTM -> Linear.
* **Xử lý dữ liệu:**
    * Tải dữ liệu từ Hugging Face.
    * Chuyển đổi nhãn số sang chuỗi (B-PER, I-ORG...).
    * Sử dụng `IGNORE_INDEX = -1` cho phần padding của nhãn.

### 2. Phân tích kết quả
* **Độ chính xác (Validation Accuracy):** Đạt **94.07%** sau 5 epochs.
* Đây là một kết quả rất cao, cho thấy LSTM học rất tốt các quy luật chuỗi của bài toán NER (ví dụ: sau B-PER thường là I-PER).

### 3. Khó khăn và Giải pháp (Quan trọng)
* **Thách thức:** Khi sử dụng thư viện `datasets` phiên bản mới nhất để tải CoNLL-2003, gặp lỗi `RuntimeError: Dataset scripts are no longer supported`.
* **Giải pháp:** Đây là lỗi do chính sách bảo mật mới của Hugging Face. Cách khắc phục hiệu quả nhất là **hạ cấp thư viện datasets**.
    ```python
    !pip install "datasets==2.21.0"
    ```
    Sau đó khởi động lại Runtime (Restart Session) để áp dụng thay đổi.

---

## HƯỚNG DẪN CHẠY CODE

Tải các file .ipynb trên github về máy và thực hiện theo trình tự trên Google Colab hoặc môi trường Jupyter Notebook có GPU:

1.  **Cài đặt thư viện:**
    ```bash
    pip install torch numpy scikit-learn gensim
    pip install "datasets==2.21.0"  # Quan trọng cho phần NER
    ```
    *(Lưu ý: Sau khi cài datasets, cần Restart Runtime)*

2.  **Thực thi phần Text Classification:**
    * Tải file `hwu.tar.gz` và giải nén.
    * Chạy tuần tự các cell để train 4 mô hình và xem bảng so sánh.

3.  **Thực thi phần POS Tagging:**
    * Tải file `UD_English-EWT.tar.gz`.
    * Chạy code định nghĩa Dataset, Model và vòng lặp Training.

4.  **Thực thi phần NER:**
    * Đảm bảo đã cài `datasets==2.21.0`.
    * Chạy code load dataset CoNLL-2003 và train mô hình LSTM.
