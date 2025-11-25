# BÁO CÁO LAB 5: XÂY DỰNG MÔ HÌNH RNN CHO BÀI TOÁN GÁN NHÃN TỪ LOẠI (POS TAGGING)

**Họ và tên:** [Điền tên của bạn vào đây]
**MSSV:** [Điền MSSV của bạn vào đây]

---

## 1. THỐNG KÊ DỮ LIỆU VÀ MÔ HÌNH

### Thông tin dữ liệu
* **Số lượng câu huấn luyện (Train):** 12,544 câu.
* **Số lượng câu phát triển (Dev):** 2,001 câu.
* **Kích thước từ điển (Vocabulary):** 9,875 từ.
* **Số lượng nhãn (Tagset):** 18 nhãn.

### Thông tin mô hình
* **Kiến trúc:** SimpleRNNForTokenClassification.
* **Thiết bị sử dụng:** cuda (GPU).
* **Tổng số tham số có thể huấn luyện:** 999,294 tham số.

---

## 2. QUÁ TRÌNH HUẤN LUYỆN

Mô hình được huấn luyện qua 20 epochs. Dưới đây là tóm tắt quá trình hội tụ:

* **Epoch 1:** Train Loss: 1.2471 | Dev Acc: 74.75%
* **Epoch 10:** Train Loss: 0.1863 | Dev Acc: 88.54%
* **Epoch 20:** Train Loss: 0.1036 | Dev Acc: 87.75%

Biểu đồ Loss và Accuracy cho thấy mô hình đã học tốt, với Loss giảm dần và Accuracy tăng dần, đạt đỉnh ổn định quanh mức 88%.

---

## 3. KẾT QUẢ THỰC HIỆN CUỐI CÙNG

### Độ chính xác (Accuracy)
* **Độ chính xác cuối cùng trên tập Dev:** **88.58%**

### Ví dụ Dự đoán thực tế
Mô hình thực hiện dự đoán trên câu mới: *"I love NLP"*

* **Đầu vào:** `'I love NLP'`
* **Kết quả dự đoán:**
    ```python
    [('I', 'PRON'), ('love', 'VERB'), ('NLP', 'PROPN')]
    ```
---
