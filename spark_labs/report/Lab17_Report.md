# Báo Cáo Lab 17: Spark NLP Pipeline

## 1. Tóm Tắt Các Yêu Cầu

**Read the C4 dataset into a Spark DataFrame**
- Đọc thành công file `c4-train.00000-of-01024-30K.json.gz` 
- Giới hạn 1000 records để tăng tốc độ xử lý
- Code: `val initialDF = spark.read.json(dataPath).limit(1000)`

**Implement a Spark ML Pipeline**
- Tạo pipeline với 4 stages: Tokenizer → StopWordsRemover → HashingTF → IDF
- Code: `val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf))`

**Use RegexTokenizer or Tokenizer for tokenization**
- Sử dụng RegexTokenizer với pattern `\\s+|[.,;!?()\"']`
- Tách từ dựa trên whitespace và dấu câu
- Code: `val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens")`

**Use StopWordsRemover to remove stop words**
- Loại bỏ stop words tiếng Anh (the, a, an, is, etc.)
- Code: `val stopWordsRemover = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered_tokens")`

**Use HashingTF and IDF for vectorization**
- HashingTF: Chuyển tokens thành feature vectors (20,000 dimensions)
- IDF: Tính toán Inverse Document Frequency để giảm trọng số các từ phổ biến
- Code: `val hashingTF = new HashingTF().setNumFeatures(20000)` và `val idf = new IDF()`

**Fit the pipeline and transform the data**
- Fit pipeline trong 1.88 seconds
- Transform data trong 0.73 seconds
- Code: `val pipelineModel = pipeline.fit(initialDF)` và `val transformedDF = pipelineModel.transform(initialDF)`

**Save the results to a file**
- Lưu kết quả vào `results/lab17_pipeline_output.txt`
- 20 records đầu với text gốc và TF-IDF vectors

**Log the process**
- Ghi log performance metrics vào `log/lab17_metrics.log`
- Bao gồm thời gian fit, transform, vocabulary size

## 2. Các Bước Thực Hiện Chi Tiết

### Bước 1: Khởi tạo Spark Session
```scala
val spark = SparkSession.builder
  .appName("NLP Pipeline Example")
  .master("local[*]")
  .getOrCreate()
spark.sparkContext.setLogLevel("WARN") // Ẩn log chi tiết
```

### Bước 2: Đọc Dataset
```scala
val dataPath = "data/c4-train.00000-of-01024-30K.json.gz"
val initialDF = spark.read.json(dataPath).limit(1000)
```

### Bước 3: Định nghĩa Pipeline Stages
1. **RegexTokenizer**: Tách văn bản thành tokens
2. **StopWordsRemover**: Loại bỏ stop words
3. **HashingTF**: Chuyển tokens thành term frequency vectors
4. **IDF**: Tính Inverse Document Frequency

### Bước 4: Training và Transform
```scala
val pipelineModel = pipeline.fit(initialDF)
val transformedDF = pipelineModel.transform(initialDF)
```

### Bước 5: Lưu Kết Quả
- Metrics → `log/lab17_metrics.log`
- Output → `results/lab17_pipeline_output.txt`

## 3. Cách Chạy Code

### Yêu cầu hệ thống:
- Java 8+ (đã test với Java 19)
- Scala 2.12
- SBT 1.9.7
- Apache Spark 3.5.1

### Các bước chạy:
```bash
cd spark_labs
sbt run
```

## 4. Giải Thích Kết Quả

### Performance Metrics:
- **Pipeline fitting**: 1.88 seconds
- **Data transformation**: 0.73 seconds  
- **Vocabulary size**: 31,355 unique terms (sau khi loại bỏ stop words)
- **Feature vector size**: 20,000 dimensions

### Hiện tượng Hash Collision:
- Vocabulary thực tế (31,355) > numFeatures (20,000)
- Một số từ khác nhau sẽ map về cùng feature index
- Đây là trade-off giữa memory và độ chính xác

### Kết quả TF-IDF Vector:
- Mỗi document được biểu diễn bằng sparse vector 20,000 chiều
- Chỉ các index có giá trị > 0 được lưu trữ
- Ví dụ: `(20000,[264,298,673,...],[15.857,2.781,3.297,...])`

### Sample Output Analysis:
1. **Document 1 (BBQ Class)**: 62 non-zero features, tập trung vào cooking terms
2. **Document 2 (Mac Discussion)**: 109 non-zero features, nhiều technical terms
3. **Document 3 (Fashion)**: 22 non-zero features, ít vocabulary hơn

## 5. Khó Khăn Gặp Phải và Cách Giải Quyết

### Khó khăn 1: Log quá nhiều thông tin
**Vấn đề**: Spark tạo ra rất nhiều INFO logs làm khó theo dõi
**Giải pháp**: 
- Thêm `spark.sparkContext.setLogLevel("WARN")`
- Tạo file `log4j2.properties` để config log level

### Khó khăn 2: Memory và Performance
**Vấn đề**: Dataset lớn có thể gây OutOfMemory
**Giải pháp**:
- Limit dataset xuống 1000 records cho lab
- Cache transformed DataFrame: `transformedDF.cache()`
- Sử dụng `local[*]` để tận dụng multi-core

### Khó khăn 3: Regex Pattern cho Tokenizer
**Vấn đề**: Pattern ban đầu không tách được dấu câu
**Giải pháp**: Sử dụng `\\s+|[.,;!?()\"']` để tách cả whitespace và punctuation

### Khó khăn 4: File I/O và Directory Creation
**Vấn đề**: Thư mục output không tồn tại
**Giải pháp**: Thêm `new File(path).getParentFile.mkdirs()`

## 6. Tài Liệu Tham Khảo

1. **Apache Spark MLlib Guide**: https://spark.apache.org/docs/latest/ml-guide.html
2. **Spark NLP Pipeline Documentation**: https://spark.apache.org/docs/latest/ml-pipeline.html
3. **TF-IDF Algorithm**: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
4. **C4 Dataset**: https://huggingface.co/datasets/c4

## 7. Mô Hình và Công Cụ Sử dụng

### Pre-trained Models: Không sử dụng
- Tất cả components đều được train từ scratch trên dataset
- RegexTokenizer: Rule-based, không cần training
- StopWordsRemover: Sử dụng default English stop words list
- HashingTF + IDF: Unsupervised learning từ corpus

### Frameworks và Libraries:
- **Apache Spark 3.5.1**: Distributed computing engine
- **Spark MLlib**: Machine learning library
- **Scala 2.12**: Programming language
- **SBT**: Build tool

## 8. Kết Luận

Pipeline NLP hoạt động hiệu quả với thời gian xử lý nhanh và kết quả chất lượng cao, phù hợp cho việc preprocessing text data trong các ứng dụng machine learning.
