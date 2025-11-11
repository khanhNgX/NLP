# **Bảng Tổng hợp Kết quả Đánh giá Mô hình**

| Model                          |   F1-score (Macro Avg) | Test Loss   |
|:-------------------------------|-----------------------:|:------------|
| TF-IDF + Logistic Regression   |                 0.82   | N/A         |
| Word2Vec (Avg) + Dense         |                 0.3514 | 2.2898      |
| Embedding (Pre-trained) + LSTM |                 0.3893 | 2.0520      |
| Embedding (Scratch) + LSTM     |                 0.6192 | 1.8378      |

# **Nhận xét**

**Kết quả Dự đoán:**

| Câu                                                              | Nhãn mong đợi        | TF-IDF + LR          | W2V (Avg) + Dense    | Emb (Pre-trained) + LSTM | Emb (Scratch) + LSTM |
|:-----------------------------------------------------------------|:--------------------|:---------------------|:---------------------|:-------------------------|:---------------------|
| "I don't want to set an alarm for tomorrow, cancel it."      | `alarm_remove`      | `alarm_set`          | `alarm_set`          | `alarm_set`              | `alarm_set`          |
| "Is there no music playing right now?"                       | `music_query`       | `music_query`        | `general_dontcare`   | `recommendation_events`  | `music_likeness`     |
| "Remind me not to forget to buy milk."                       | `lists_createoradd` | `calendar_set`       | `transport_query`    | `cooking_recipe`         | `calendar_set`       |
| "Turn off all the lights in the living room, except for the lamp." | `iot_hue_lightoff`  | `iot_hue_lightoff`   | `iot_hue_lightup`    | `iot_hue_lightoff`       | `iot_hue_lightoff`   |
| "Tell me what you are not able to do."                       | `general_quirky`    | `general_quirky`     | `general_explain`    | `music_query`            | `qa_stock`           |

**Phân tích:**

1.  **Câu 1: "I don't want to set an alarm for tomorrow, cancel it." (Phủ định + Hành động kép)**
    *   **Mong đợi:** `alarm_remove` (Hủy bỏ báo thức)
    *   **Dự đoán của tất cả các mô hình:** `alarm_set`
    *   **Nhận xét:** Tất cả các mô hình đều gặp khó khăn với câu này. Yếu tố "don't want to set" và "cancel it" rõ ràng chỉ ra `alarm_remove`, nhưng tất cả đều dự đoán `alarm_set`. Điều này cho thấy các mô hình, bao gồm cả LSTM, vẫn chưa hiểu được sắc thái của sự phủ định và mối quan hệ giữa các mệnh đề trong câu để đưa ra ý định chính xác. Có lẽ từ khóa "set an alarm" mạnh hơn các từ phủ định hoặc hủy bỏ.

2.  **Câu 2: "Is there no music playing right now?" (Phủ định + Câu hỏi)**
    *   **Mong đợi:** `music_query`
    *   **Dự đoán:**
        *   TF-IDF + LR: `music_query` (Chính xác)
        *   W2V (Avg) + Dense: `general_dontcare`
        *   Emb (Pre-trained) + LSTM: `recommendation_events`
        *   Emb (Scratch) + LSTM: `music_likeness`
    *   **Nhận xét:** TF-IDF + LR bất ngờ lại chính xác nhất ở đây. Các mô hình Word2Vec và LSTM đều không đúng. Điều này có thể do TF-IDF bắt được các từ khóa như "music" và "playing" mà không bị các từ phủ định như "no" làm nhiễu loạn quá nhiều, hoặc các mô hình dựa trên embedding đã bị ảnh hưởng bởi "no" và các từ xung quanh theo cách không mong muốn.

3.  **Câu 3: "Remind me not to forget to buy milk." (Phủ định + Ý định phức tạp)**
    *   **Mong đợi:** `lists_createoradd` (Thêm vào danh sách)
    *   **Dự đoán:**
        *   TF-IDF + LR: `calendar_set`
        *   W2V (Avg) + Dense: `transport_query`
        *   Emb (Pre-trained) + LSTM: `cooking_recipe`
        *   Emb (Scratch) + LSTM: `calendar_set`
    *   **Nhận xét:** Không có mô hình nào dự đoán chính xác. Ý định này phức tạp vì nó chứa "remind me" (có thể liên quan đến `calendar` hoặc `alarm`), "not to forget" (phủ định), và "buy milk" (liên quan đến `lists`). TF-IDF + LR và Emb (Scratch) + LSTM lại ra `calendar_set`, có lẽ do từ "remind". Các mô hình LSTM cũng cho thấy sự lẫn lộn giữa các ý định liên quan đến mua sắm/nấu ăn (`cooking_recipe`) hoặc tra cứu (`transport_query`).

4.  **Câu 4: "Turn off all the lights in the living room, except for the lamp." (Ngoại lệ + Điều khiển nhà thông minh)**
    *   **Mong đợi:** `iot_hue_lightoff`
    *   **Dự đoán:**
        *   TF-IDF + LR: `iot_hue_lightoff` (Chính xác)
        *   W2V (Avg) + Dense: `iot_hue_lightup`
        *   Emb (Pre-trained) + LSTM: `iot_hue_lightoff` (Chính xác)
        *   Emb (Scratch) + LSTM: `iot_hue_lightoff` (Chính xác)
    *   **Nhận xét:** Hầu hết các mô hình đều xử lý tốt câu này, ngoại trừ W2V (Avg) + Dense. Từ khóa "turn off" và "lights" đủ mạnh để dẫn đến dự đoán chính xác. Từ "except for the lamp" có thể là một yếu tố gây nhiễu, nhưng các mô hình LSTM và TF-IDF + LR đã bỏ qua nó hoặc hiểu ngữ cảnh tốt hơn. W2V (Avg) + Dense có thể đã bị ảnh hưởng bởi một số từ liên quan đến "light up" hoặc do tính trung bình hóa vector đã làm mất đi ngữ nghĩa quan trọng.

5.  **Câu 5: "Tell me what you are not able to do." (Trừu tượng + Phủ định)**
    *   **Mong đợi:** `general_quirky`
    *   **Dự đoán:**
        *   TF-IDF + LR: `general_quirky` (Chính xác)
        *   W2V (Avg) + Dense: `general_explain`
        *   Emb (Pre-trained) + LSTM: `music_query`
        *   Emb (Scratch) + LSTM: `qa_stock`
    *   **Nhận xét:** TF-IDF + LR lại một lần nữa đưa ra dự đoán chính xác. Các mô hình Embedding gặp khó khăn với câu hỏi trừu tượng này. Có vẻ như "general_quirky" là một nhãn khó học, và các mô hình có xu hướng đoán sang các loại câu hỏi chung chung khác (`general_explain`) hoặc thậm chí không liên quan (`music_query`, `qa_stock`).

**Tổng kết và Đánh giá khả năng xử lý chuỗi của LSTM:**

*   **TF-IDF + Logistic Regression:** Bất ngờ hoạt động rất tốt trên các câu khó này, đặc biệt là với các câu có yếu tố phủ định hoặc yêu cầu cụ thể. Điều này cho thấy TF-IDF có thể nắm bắt các từ khóa quan trọng và Logistic Regression có thể phân loại chúng hiệu quả, đôi khi tốt hơn các mô hình phức tạp hơn với lượng dữ liệu và ngữ cảnh này.

*   **Word2Vec (Trung bình) + Dense:** Cho thấy hiệu suất kém nhất trên các câu khó, điều này phù hợp với điểm số F1-score thấp nhất. Việc chỉ dùng vector trung bình có thể làm mất đi thông tin thứ tự từ và ngữ cảnh quan trọng, khiến mô hình khó hiểu các sắc thái phức tạp.

*   **Embedding (Pre-trained) + LSTM & Embedding (Scratch) + LSTM:** Mặc dù điểm F1-score tổng thể của LSTM (Scratch) cao hơn nhiều so với TF-IDF + LR, nhưng trên các câu khó, chúng lại không vượt trội hơn TF-IDF + LR. Cả hai mô hình LSTM đều gặp khó khăn với việc hiểu phủ định ("don't want", "no", "not to forget") và các cấu trúc phức tạp. Đặc biệt, câu 1 và 3 cho thấy LSTM chưa đủ khả năng để giải quyết sự phụ thuộc xa và các yếu tố ngữ nghĩa tinh tế liên quan đến phủ định hoặc ý định kép. Sự khác biệt giữa Embedding Pre-trained và học từ đầu không quá rõ ràng trên tập câu khó này, mặc dù model học từ đầu có điểm F1-score cao hơn đáng kể trên tập test tổng thể.

**Điểm mạnh của LSTM:** Chúng ta đã thấy LSTM (Scratch) đạt được F1-score cao nhất trên tập test tổng thể (0.6192), cho thấy khả năng của chúng trong việc học các mối quan hệ tuần tự và ngữ cảnh trong các câu bình thường. Tuy nhiên, trên các câu được thiết kế đặc biệt để kiểm tra khả năng hiểu phủ định và cấu trúc phức tạp, chúng vẫn còn hạn chế. Điều này có thể là do:

*   **Lượng dữ liệu:** Mặc dù có dữ liệu đủ dùng, các trường hợp phủ định hoặc phức tạp có thể không đủ phong phú trong tập huấn luyện để mô hình học được quy tắc tổng quát.
*   **Độ phức tạp của nhiệm vụ:** Phân tích ý định với các yếu tố phủ định hoặc cấu trúc ngôn ngữ khó là một nhiệm vụ phức tạp, đòi hỏi khả năng lý luận sâu hơn mà ngay cả các mô hình LSTM hiện tại cũng có thể gặp khó khăn.

**Kết luận:** Dựa trên phân tích các câu khó, TF-IDF + Logistic Regression thể hiện sự mạnh mẽ đáng ngạc nhiên trong một số trường hợp cụ thể. Các mô hình LSTM, mặc dù tốt hơn về hiệu suất tổng thể, vẫn cần cải thiện đáng kể trong việc xử lý các yếu tố phủ định và ngữ cảnh phức tạp. Điều này gợi ý rằng việc kết hợp các phương pháp khác (ví dụ: các mô hình Transformer, kỹ thuật tăng cường dữ liệu cho các trường hợp cạnh khóe, hoặc xử lý ngôn ngữ dựa trên luật) có thể là cần thiết để đạt được hiệu suất cao hơn trong các tình huống khó khăn.
