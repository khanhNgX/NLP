
# Lab5 — Báo cáo và Phân tích

## 1. Tóm tắt các bước triển khai

- Cài đặt một lớp `TextClassifier` (file `Lab5/src/models/text_classifier.py`) dùng `LogisticRegression` của scikit-learn. Lớp này:
  - `fit(texts, labels)` — gọi vectorizer.fit_transform(texts) rồi train LogisticRegression.
  - `predict(texts)` — gọi vectorizer.transform(texts) rồi dự đoán.
  - `evaluate(y_true, y_pred)` — tính accuracy / precision / recall / f1.
- Tạo test cơ bản `Lab5/test/lab5_test.py`:
  - Dùng `RegexTokenizer` (từ Lab1&Lab2) để tokenize, `TfidfVectorizer` cho đặc trưng, rồi `TextClassifier` để huấn luyện và đánh giá.
- Tạo kịch bản cải tiến `Lab5/test/lab5_improvement_test.py`:
  - Thêm tiền xử lý đơn giản (loại URL/HTML/ký tự đặc biệt, lowercase).
  - Hỗ trợ biểu diễn dense bằng Word2Vec (huấn luyện nhỏ với gensim) hoặc tải pretrained embeddings (GloVe/FastText) nếu có.
  - Nếu không có pretrained file, script tạo một pseudo‑GloVe deterministic để demo (không cần file ngoài).
  - Tính vector câu bằng cách trung bình các vector từ (sentence average) và dùng các mô hình: LogisticRegression, GradientBoostingClassifier, MLPClassifier để so sánh.
- Thêm ví dụ Spark ML `Lab5/test/lab5_spark_sentiment_analysis.py` (Tokenizer → StopWordsRemover → HashingTF → IDF → LogisticRegression) để chạy trên `data/sentiments.csv`.

## 2. Hướng dẫn chạy mã

Từ thư mục `D:\VSCode\NLP\Lab5`:

- Chạy test baseline (sample + file nếu có):
```powershell
python test\lab5_test.py
```

- Chạy kịch bản cải tiến (mặc định chạy cả sample và file nếu tồn tại; mặc định dùng pseudo‑GloVe pretrained demo):
```powershell
python test\lab5_improvement_test.py
```
- Chạy ví dụ Spark ML (cần Java + pyspark):
```powershell
# cài pyspark nếu chưa có
pip install pyspark
python test\lab5_spark_sentiment_analysis.py
```

Lưu ý: trên Windows, Spark có thể in một vài cảnh báo về `winutils.exe` / `HADOOP_HOME` — các cảnh báo này thường không chặn pipeline, nhưng để loại bỏ cảnh báo hoặc sử dụng tính năng HDFS bạn cần tải `winutils.exe` và đặt `HADOOP_HOME`.

## 3. Kết quả thử nghiệm

Ghi chú: có hai nguồn kết quả — test baseline (`lab5_test.py` dùng TF‑IDF + LogisticRegression) và test cải tiến (`lab5_improvement_test.py` dùng embeddings trung bình + các mô hình).

- Kết quả baseline (chạy `lab5_test.py`):
  - Sample (6 docs):
    - accuracy: 0.5000
    - precision: 0.0000
    - recall: 0.0000
    - f1: 0.0000
  - File `data/sentiments.csv`:
    - accuracy: 0.7788
    - precision: 0.7765
    - recall: 0.9162
    - f1: 0.8406

- Kết quả cải tiến (chạy `lab5_improvement_test.py` mặc định → sample pseudo‑GloVe + file nếu có):
  - Sample (6 docs, pseudo‑GloVe):
    - LogisticRegression (embeddings): 
        - accuracy 0.5000
        - f1 0.0000
    - GradientBoosting (embeddings): 
        - accuracy 1.0000
        - f1 1.0000
    - MLPClassifier (embeddings): 
        - accuracy 0.5000
        - f1 0.0000
  - File `data/sentiments.csv` (embeddings trung bình):
    - LogisticRegression (embeddings): 
        - accuracy 0.6644
        - precision 0.6839
        - recall 0.8794
        - f1 0.7694
    - GradientBoosting (embeddings): 
        - accuracy 0.6437
        - precision 0.6623
        - recall 0.8984
        - f1 0.7625
    - MLPClassifier (embeddings):
        - accuracy 0.6601
        - precision 0.7251
        - recall 0.7507
        - f1 0.7377

- Kết quả Spark ML (`lab5_spark_sentiment_analysis.py` trên `data/sentiments.csv`):
  - Số dòng ban đầu: 5792
  - Số dòng sau khi drop null sentiment: 5791
  - Accuracy trên tập test: 0.7295
  - F1 trên tập test: 0.7266

## 4. So sánh kết quả và phân tích lý do

Tóm tắt ngắn: trên dữ liệu `data/sentiments.csv` mô hình baseline TF‑IDF + LogisticRegression cho kết quả tốt hơn so với đồ thị sử dụng embeddings trung bình. Cụ thể:

- TF‑IDF + LogisticRegression (baseline): accuracy 0.7788, F1 0.8406
- Embeddings (sentence average) + LogisticRegression: accuracy 0.6644, F1 0.7694

Như vậy, trong thí nghiệm này phương pháp embeddings trung bình không cải thiện so với TF‑IDF. Dưới đây là các nguyên nhân chính và phân tích chi tiết:

1) Mất thông tin khi lấy trung bình (sentence averaging)
- Trung bình các vector từ làm mất thông tin về vị trí từ, trật tự và tương tác giữa các từ. Những từ mang tính phân biệt (ví dụ phủ định, từ mang cảm xúc) có thể bị trung hòa khi lấy trung bình.

2) Chất lượng và tính phù hợp của embeddings
- Nếu dùng `pseudo‑GloVe` (demo) hoặc embedding không phù hợp với miền dữ liệu (domain mismatch), vector sẽ không phản ánh tốt ngữ cảnh và cảm xúc trong câu. Embeddings pretrained lớn (GloVe/FastText/BERT) thường cho biểu diễn tốt hơn.

3) Tương thích giữa dạng đặc trưng và mô hình
- Một số mô hình (ví dụ MultinomialNB) vận hành tốt trên đặc trưng dạng đếm/TF‑IDF vì giả định tính độc lập tính xác suất token; trong khi embeddings dense lại phù hợp hơn với các mô hình xử lý dense (LR, MLP) nhưng cần được huấn luyện/tuned đúng cách.

4) Dữ liệu và overfitting
- Với tập mẫu quá nhỏ (ví dụ 6 câu), kết quả bị nhiễu và các mô hình phức tạp dễ overfit (ví dụ GBT đạt 1.0 trên sample do overfit). Ngay cả trên file lớn hơn, nếu không có đủ dữ liệu biểu diễn ngôn ngữ phong phú, embeddings trung bình cũng có thể không cho lợi ích rõ rệt.

5) Thiếu tuning và bước tiền xử lý bổ sung
- Chưa thực hiện chuẩn hoá/scale feature, cân bằng lớp (nếu imbalance), hay tuning hyperparameters. Những bước này đặc biệt quan trọng khi dùng MLP/GBT trên embeddings dense.

Khi nào embeddings có thể hiệu quả hơn
- Dùng pretrained embeddings chất lượng (GloVe/FastText) hoặc sentence-transformers (BERT/SBERT) thay vì averaging đơn giản.
- Áp dụng pooling thông minh (max, attention-weighted, TF‑IDF weighted average) thay vì mean đơn thuần.
- Sử dụng mô hình ngôn ngữ lớn hoặc fine‑tune một sentence-transformer cho nhiệm vụ downstream.

Khuyến nghị để cải thiện trong các thử nghiệm tiếp theo
- Thử các phương pháp pooling khác (max, weighted average), hoặc dùng sentence-transformers để lấy embedding câu.
- Nếu dùng embeddings pretrained, đảm bảo tải embeddings phù hợp với ngôn ngữ/miền và dùng kỹ thuật tiền xử lý tương thích.
- Thực hiện cross‑validation (StratifiedKFold) để có chỉ số ổn định (mean ± std) trước khi so sánh.
- Tối ưu hyperparameters (GridSearch/RandomizedSearch), chuẩn hoá features trước khi đưa vào MLP/GBT, và cân bằng lớp nếu cần.

Kết luận: kỹ thuật dùng embeddings trung bình trong thực nghiệm này không vượt trội so với TF‑IDF do sự mất mát thông tin khi pooling, chất lượng embeddings demo, và thiếu tuning. Tuy nhiên với embeddings pretrained tốt hoặc phương pháp pooling/encoder phù hợp, embeddings có thể đem lại lợi thế rõ rệt.

## 5. Thử thách gặp phải và cách giải quyết

- Vấn đề sparse/dense: một số mô hình (GradientBoosting/MLP) cần input dense trong khi TF‑IDF trả về ma trận sparse; giải pháp: chuyển sang `.toarray()` khi dữ liệu nhỏ, nhưng cần cẩn thận với dữ liệu lớn vì tốn bộ nhớ.
- Thiếu gensim hoặc pretrained embeddings: thêm fallback `pseudo‑GloVe` cho demo (không cần cài gensim), đồng thời hỗ trợ load gensim nếu người dùng cài.
- Lỗi code ban đầu: có lỗi indent và lỗi `pseudo_glove_vector` thiếu đối số — đã sửa.
- Trên Windows, khi chạy Spark có cảnh báo `winutils.exe`/`HADOOP_HOME` — giải pháp: tải `winutils.exe` phù hợp và set `HADOOP_HOME`, hoặc chấp nhận cảnh báo nếu không dùng HDFS.
- MLP có cảnh báo không hội tụ (ConvergenceWarning): tăng `max_iter` hoặc chuẩn hoá/scaling features giúp cải thiện.

## 6. Đề xuất cải tiến tiếp theo

- Dùng pretrained embeddings thực tế (GloVe / FastText) và fine-tune/điều chỉnh nếu cần.
- Thử các phương pháp biểu diễn câu tốt hơn: doc2vec, sentence-transformers (BERT-based) để lấy embedding ngữ nghĩa mạnh hơn.
- Chạy cross-validation (`StratifiedKFold`) để có metrics ổn định (mean ± std).
- Tối ưu hyperparameters bằng GridSearch/RandomizedSearch.

## 7. Tài liệu tham khảo

- scikit-learn documentation — LogisticRegression, metrics, model selection: https://scikit-learn.org
- gensim documentation — Word2Vec, KeyedVectors, glove2word2vec: https://radimrehurek.com/gensim/
- Apache Spark MLlib documentation — Pipelines và feature transformers: https://spark.apache.org/docs/latest/ml-guide.html
- Các hướng dẫn GloVe / FastText và sentence embedding

---
