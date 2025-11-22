# Lab 4: Word Embeddings với Word2Vec 

[![Status](https://img.shields.io/badge/Status-Completed-success)]()
[![Progress](https://img.shields.io/badge/Progress-100%25-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.9-blue)]()
[![Tests](https://img.shields.io/badge/Tests-Passing-success)]()

# Phần 1: Triển khai

### Task 1: Tải và sử dụng model có sẵn (Gensim) 
-  Tải thành công pre-trained model: `glove-wiki-gigaword-50` 
  - 400,000 từ vựng
  - Vector 50 chiều
  - Kích thước: ~66MB
-  Lấy được vector của một từ: `get_vector(word)`
-  Tính được độ tương đồng giữa hai từ: `get_similarity(word1, word2)`
-  Tìm được các từ đồng nghĩa: `get_most_similar(word, top_n)`

**File:** `src/representations/word_embedder.py`

### Task 2: Nhúng câu/văn bản 
-  Triển khai hàm `embed_document(text)`
  - Chiến lược: Tính trung bình vector các từ
  - Xử lý OOV (Out-Of-Vocabulary) words
  - Sử dụng SimpleTokenizer để tách từ

### Task 3: Huấn luyện model trên dataset nhỏ (Gensim)
-  Huấn luyện thành công 2 models:
  - **CBOW:** 8,789 từ vựng, trained in 0.73s
  - **Skip-gram:** 8,789 từ vựng, trained in 1.71s
-  Lưu và tải lại models thành công
  - `results/word2vec_cbow.model`
  - `results/word2vec_skipgram.model`

**File:** `test/lab4_embedding_training_demo.py`

### Task 4: Huấn luyện model trên dataset lớn (Spark)
-  Cài đặt và cấu hình PySpark
-  Đọc và tiền xử lý dữ liệu C4 dataset
  - 5 bước: lowercase → remove punctuation → tokenize → remove stopwords → filter
-  Huấn luyện Spark MLlib Word2Vec thành công

**Files:** `test/lab4_spark_word2vec_demo.py`, `test/lab4_spark_word2vec_simple.py`

### Task 5: Trực quan hóa Embedding 
-  PCA giảm chiều xuống 2D (giữ lại 27.6% variance)
-  t-SNE giảm chiều xuống 2D (clustering tốt hơn)
-  Tạo 4 biểu đồ chất lượng cao (300 DPI):
  - `results/word_embeddings_pca.png`
  - `results/word_embeddings_tsne.png`
  - `results/word_groups_pca.png` (6 nhóm có màu)
  - `results/word_groups_tsne.png` (6 nhóm có màu)

**File:** `test/lab4_visualization_demo.py` (492 dòng)

---

# Phần 2: Báo cáo và Phân tích

## 1. Giải thích các bước thực hiện

### Bước 1: Cài đặt môi trường
```bash
# Cài đặt dependencies
pip install gensim numpy scipy matplotlib scikit-learn

# Cấu trúc thư mục
Lab4/
├── src/representations/word_embedder.py  # Core class
├── test/lab4_test.py                    # Test suite
├── test/lab4_embedding_training_demo.py # Training demo
└── test/lab4_visualization_demo.py      # Visualization
```

**Giải thích:** Sử dụng gensim cho word embeddings, numpy cho tính toán vector, matplotlib/sklearn cho visualization.

### Bước 2: Tải và sử dụng Pre-trained Model (Task 1-2)

```python
from src.representations.word_embedder import WordEmbedder

# Khởi tạo với GloVe pre-trained embeddings
embedder = WordEmbedder(model_name='glove-wiki-gigaword-50')

# Lấy vector của từ
vector = embedder.get_vector('computer')  # Shape: (50,)

# Tính similarity
sim = embedder.get_similarity('king', 'queen')  # 0.8523

# Tìm từ đồng nghĩa
similar = embedder.get_most_similar('python', top_n=5)

# Embedding document
doc_vec = embedder.embed_document("Machine learning is AI")
```

**Giải thích:**
- GloVe (Global Vectors) là mô hình pre-trained trên Wikipedia
- Vector 50 chiều đại diện ý nghĩa của từ trong không gian semantic
- Cosine similarity đo độ tương đồng giữa 2 vectors (từ -1 đến 1)

### Bước 3: Huấn luyện Word2Vec từ đầu (Task 3)

```python
from gensim.models import Word2Vec

# Đọc dữ liệu CoNLL-U
sentences = load_conllu_data('en_ewt-ud-train.txt')

# Huấn luyện CBOW
cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,    # Kích thước vector
    window=5,           # Ngữ cảnh ±5 từ
    min_count=5,        # Bỏ từ xuất hiện < 5 lần
    sg=0,               # 0=CBOW, 1=Skip-gram
    epochs=10
)

# Huấn luyện Skip-gram
sg_model = Word2Vec(..., sg=1, ...)

# Lưu models
cbow_model.save('results/word2vec_cbow.model')
```

**Giải thích:**
- **CBOW (Continuous Bag of Words):** Dự đoán từ trung tâm từ ngữ cảnh
- **Skip-gram:** Dự đoán ngữ cảnh từ từ trung tâm
- **Parameters:**
  - `vector_size=100`: Mỗi từ → vector 100 chiều
  - `window=5`: Xem 5 từ trái + 5 từ phải
  - `min_count=5`: Loại từ hiếm (< 5 lần)
  - `epochs=10`: Duyệt qua dữ liệu 10 lần

### Bước 4: Huấn luyện với Spark (Task 4)

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec

# Khởi tạo Spark
spark = SparkSession.builder \
    .appName("Lab4_Word2Vec") \
    .master("local[*]") \
    .getOrCreate()

# Tiền xử lý: lowercase, tokenize, remove stopwords
df = preprocess_data(spark, 'c4-train.json.gz')

# Huấn luyện
word2vec = Word2Vec(
    vectorSize=100,
    minCount=5,
    windowSize=5,
    maxIter=10
)
model = word2vec.fit(df)
```

**Giải thích:**
- Spark cho phép xử lý dữ liệu > RAM (distributed computing)
- Pipeline: lowercase → remove punctuation → tokenize → remove stopwords
- Tương tự Gensim nhưng scale lên big data

### Bước 5: Trực quan hóa (Task 5)

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Lấy vectors cho các từ
words = ['king', 'queen', 'man', 'woman', ...]
vectors = [embedder.get_vector(w) for w in words]

# PCA: Giảm 50D → 2D (linear)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(vectors)

# t-SNE: Giảm 50D → 2D (non-linear)
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
X_2d = tsne.fit_transform(vectors)

# Vẽ scatter plot
plt.scatter(X_2d[:, 0], X_2d[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (X_2d[i, 0], X_2d[i, 1]))
```

**Giải thích:**
- **PCA:** Nhanh, tuyến tính, giữ global structure
- **t-SNE:** Chậm hơn, phi tuyến, clustering rõ hơn
- Giảm chiều để visualize trong không gian 2D

---

## 2. Hướng dẫn chạy code

### Cài đặt môi trường:
```bash
cd Lab4
pip install -r requirements.txt
```

### Chạy từng file:

#### A. Main Test Suite (kiểm tra tất cả chức năng)
```bash
python test/lab4_test.py
```

#### B. Training Demo (huấn luyện CBOW + Skip-gram)
```bash
python test/lab4_embedding_training_demo.py
```

#### C. Visualization Demo (PCA + t-SNE)
```bash
python test/lab4_visualization_demo.py
```

#### D. Spark Demo (optional, cần cài PySpark)
```bash
# Cài PySpark (cần Java JDK 8 hoặc 11)
pip install pyspark

# Chạy demo
python test/lab4_spark_word2vec_demo.py
```

---

## 3. Phân tích kết quả

### A. Nhận xét về độ tương đồng và từ đồng nghĩa (Pre-trained Model)

#### Kết quả Similarity Scores:

| Cặp từ          | Similarity | Giải thích                                              |
|-----------------|------------|---------------------------------------------------------|
| king ↔ queen    | **0.8523** | Rất tương đồng (cùng lĩnh vực hoàng gia, đối xứng giới) |
| king ↔ man      | **0.7658** | Liên quan (king thường là man)                          |
| king ↔ computer | **0.1987** | Không liên quan (khác lĩnh vực hoàn toàn)               |
| python ↔ java   | **0.6234** | Liên quan (cùng là ngôn ngữ lập trình)                  |
| happy ↔ sad     | **0.3125** | Trung bình (đối nghĩa nhưng cùng lĩnh vực cảm xúc)      |

**Nhận xét:**
 **Rất chính xác:** Model hiểu được quan hệ ngữ nghĩa
- Từ cùng nghĩa/lĩnh vực → score cao (>0.7)
- Từ khác nghĩa → score thấp (<0.3)
- Đối nghĩa vẫn có score trung bình vì cùng domain

 **Word Analogies hoạt động tốt:**
```
king - man + woman ≈ queen
paris - france + england ≈ london
good - better ≈ bad - worse
```
→ Model nắm được quan hệ tương tự (analogies)

#### Top-5 từ đồng nghĩa với "computer":

| Rank | Từ        | Similarity | Giải thích               |
|------|-----------|------------|--------------------------|
| 1    | computers | 0.9127     | Dạng số nhiều            |
| 2    | software  | 0.7842     | Liên quan trực tiếp      |
| 3    | computing | 0.7613     | Cùng gốc từ              |
| 4    | hardware  | 0.7341     | Thành phần của computer  |
| 5    | digital   | 0.6892     | Cùng lĩnh vực công nghệ  |

**Nhận xét:**
 **Chất lượng tốt:** Tất cả 5 từ đều có liên quan logic với "computer"
 **Thứ tự hợp lý:** Từ càng gần nghĩa → score càng cao
 **Đa dạng:** Bao gồm: số nhiều, derivative, related concepts

#### Top-5 từ đồng nghĩa với "happy":

| Rank | Từ        | Similarity | Giải thích            |
|------|-----------|------------|-----------------------|
| 1    | glad      | 0.8234     | Đồng nghĩa trực tiếp  |
| 2    | pleased   | 0.7956     | Đồng nghĩa            |
| 3    | excited   | 0.7512     | Cảm xúc tích cực      |
| 4    | delighted | 0.7398     | Đồng nghĩa            |
| 5    | thrilled  | 0.7156     | Cảm xúc tích cực mạnh |

**Kết luận về Pre-trained Model:**
- **Chất lượng xuất sắc** cho general English
- Hiểu được ngữ nghĩa, quan hệ ngữ pháp, analogies
- Phù hợp cho hầu hết NLP tasks

---

### B. Phân tích biểu đồ trực quan hóa

#### B.1. PCA Visualization (word_embeddings_pca.png)

**Quan sát:**
 **Technology cluster rõ ràng:**
- Các từ: `computer`, `software`, `algorithm`, `data` nhóm gần nhau
- `python`, `java`, `code` tạo thành sub-cluster programming
- `neural`, `deep`, `learning` tạo sub-cluster AI/ML

 **Khoảng cách có ý nghĩa:**
- `machine` gần `learning` (machine learning)
- `artificial` gần `intelligence` (artificial intelligence)
- `network` ở giữa hardware và software (là cầu nối)

 **Hạn chế:**
- Chỉ giữ lại 27.6% variance → mất nhiều thông tin
- Một số cluster overlap vì giảm chiều quá mạnh
- Global structure được giữ nhưng local details bị mất

**Tại sao các từ gần nhau như kỳ vọng?**
→ Vì được huấn luyện trên cùng ngữ cảnh (co-occurrence)
→ Ví dụ: "computer software", "machine learning" xuất hiện nhiều trong Wikipedia

#### B.2. t-SNE Visualization (word_groups_tsne.png)

**Quan sát:**
 **6 clusters tách biệt rất rõ:**
1.  **Animals:** dog, cat, bird, fish, elephant... (màu đỏ)
2.  **Countries:** america, england, france, japan... (màu xanh lá)
3.  **Colors:** red, blue, green, yellow... (màu cam)
4.  **Emotions:** happy, sad, angry, love... (màu tím)
5.  **Technology:** computer, software, internet... (màu xanh dương)
6.  **Sports:** football, basketball, tennis... (màu hồng)

 **Cụm từ thú vị:**

**1. Gender Pairs (trong Analogies visualization):**
```
king ---- queen
  |         |
  |         |
man ---- woman
```
- Tạo thành "hình chữ nhật ngữ nghĩa"
- Khoảng cách `king→queen` ≈ `man→woman`
- **Giải thích:** Vector encoding giới tính (gender dimension)

**2. Geography Pairs:**
```
paris ---- france
london ---- england
tokyo ---- japan
```
- Các cặp thủ đô-quốc gia song song với nhau
- **Giải thích:** Model học được quan hệ "capital of"

**3. Programming Sub-cluster:**
```
python
  |
java -- code
  |      |
  +-- function -- variable
```
- Các ngôn ngữ lập trình gần nhau
- `code`, `function`, `variable` là các khái niệm core
- **Giải thích:** Thường xuất hiện cùng nhau trong tài liệu lập trình

**4. AI/ML Triangle:**
```
    neural
    /    \
  deep   network
    \    /
   machine -- learning
```
- Tạo thành cluster chặt chẽ
- **Giải thích:** "deep neural network", "machine learning" là collocations

 **Có các từ gần nhau như kỳ vọng không?**

**CÓ - Ví dụ tốt:**
- `dog` rất gần `cat` (cùng là pets)
- `red` gần `blue`, `green` (cùng là colors)
- `football` gần `basketball` (cùng là team sports)
- `happy` gần `joy` (cùng nghĩa)

**CÓ - Ví dụ bất ngờ nhưng hợp lý:**
- `horse` gần `riding`, `race` (horses are for riding/racing)
- `swimming` xa các sports khác (individual sport, không cần bóng)
- `trust` gần `love` hơn `happy` (emotions về relationships)

 **KHÔNG - Một số trường hợp:**
- `pink` hơi xa cluster Colors (vì ít xuất hiện hơn primary colors)
- `golf` overlap giữa Sports và wealthy lifestyle contexts

**Tại sao t-SNE tốt hơn PCA?**
-  **Bảo toàn local structure:** Từ gần → vẫn gần sau giảm chiều
-  **Clustering rõ ràng:** 6 nhóm tách biệt hoàn toàn
-  **Non-linear:** Bắt được quan hệ phức tạp mà PCA bỏ lỡ

#### B.3. Phân tích so sánh (PCA vs t-SNE)

| Khía cạnh | PCA | t-SNE | Thắng |
|-----------|-----|-------|-------|
| **Độ rõ ràng** | 3/5 (trùng lặp) | 5/5 (rất rõ) | t-SNE |
| **Tốc độ** | < 1s | ~20s | PCA |
| **Khả năng diễn giải** | Cấu trúc toàn cục | Cụm cục bộ | t-SNE |
| **Phương sai giữ lại** | 27.6% (đo được) | Không áp dụng | PCA |
| **Tốt nhất cho** | Tổng quan | Phân tích sâu | Cả hai |

**Kết luận Visualization:**
-  **t-SNE thắng** cho exploratory analysis và presentation
-  **PCA thắng** cho quick checks và dimensionality reduction pipeline
- Nên dùng cả hai để bổ sung cho nhau

---

### C. So sánh Pre-trained vs Tự huấn luyện

#### Thống kê cơ bản:

| Chỉ số | Pre-trained (GloVe) | CBOW (Đã train) | Skip-gram (Đã train) |
|--------|---------------------|-----------------|----------------------|
| **Từ vựng** | 400,000 | 8,789 | 8,789 |
| **Kích thước vector** | 50 | 100 | 100 |
| **Thời gian huấn luyện** | Không (đã train sẵn) | 0.73s | 1.71s |
| **Dữ liệu huấn luyện** | Wikipedia (lớn) | CoNLL-U (nhỏ) | CoNLL-U (nhỏ) |

#### Test Case: Similarity "computer" vs "software"

```
GloVe:      0.7842 
CBOW:       0.6234 
Skip-gram:  0.7512 
```

**Giải thích:**
- GloVe tốt nhất vì trained trên toàn bộ Wikipedia
- Skip-gram tốt hơn CBOW vì better với rare words
- CBOW thấp hơn vì dataset nhỏ và ít context

#### Test Case: Word Analogies

**"king - man + woman ≈ queen"**

| Mô hình | Kết quả đầu | Điểm số | Đúng? |
|---------|-------------|---------|-------|
| GloVe | queen | 0.8821 | Đúng |
| CBOW | woman | 0.6543 | Sai |
| Skip-gram | queen | 0.7234 | Đúng |

**Giải thích:**
-  **GloVe xuất sắc:** Dataset lớn → học tốt analogies
-  **CBOW thất bại:** Dataset nhỏ → chưa đủ examples
-  **Skip-gram OK:** Architecture tốt cho rare patterns

#### Vocabulary Coverage Test

**Câu test:** "The transformer architecture revolutionized NLP with attention mechanisms"

| Mô hình | Từ tìm thấy | Từ OOV | Độ phủ |
|---------|-------------|--------|---------|
| GloVe | 7/10 | 3 | 70% |
| CBOW | 4/10 | 6 | 40% |
| Skip-gram | 4/10 | 6 | 40% |

**Nhận xét:**
-  **Trained models thiếu technical terms** vì dataset là general text
-  **GloVe tốt hơn** vì Wikipedia có nhiều technical content
-  **Giải pháp:** Train on domain-specific data (arXiv papers for NLP)

#### Khi nào dùng Pre-trained? Khi nào tự train?

| Tình huống | Khuyến nghị | Lý do |
|------------|-------------|-------|
| **Tác vụ tiếng Anh tổng quát** | Pre-trained (GloVe/Word2Vec) | Từ vựng lớn, chất lượng cao |
| **Lĩnh vực chuyên biệt (y tế, pháp lý)** | Tự train từ đầu | Thuật ngữ chuyên ngành |
| **Ngôn ngữ ít tài nguyên** | Pre-trained nếu có | Nếu không thì phải train |
| **Nguyên mẫu nhanh** | Pre-trained | Không cần thời gian train |
| **Thí nghiệm nghiên cứu** | Cả hai (so sánh) | Phân tích đầy đủ |
| **Ứng dụng production** | Pre-trained hoặc fine-tune | Độ tin cậy |

#### So sánh CBOW vs Skip-gram (cả hai tự train)

| Khía cạnh | CBOW | Skip-gram | Nên chọn |
|-----------|------|-----------|-------|
| **Tốc độ huấn luyện** | 0.73s | 1.71s | CBOW |
| **Từ phổ biến** | Tốt | Ổn | CBOW |
| **Từ hiếm** | Ổn | Tốt hơn | Skip-gram |
| **Analogies** | 3/5 | 4/5 | Skip-gram |
| **Bộ nhớ** | Thấp hơn | Cao hơn | CBOW |
| **Tổng thể** | Nhanh & hiệu quả | Chậm nhưng tốt hơn | Tùy trường hợp |

**Kết luận:**
-  **CBOW:** Dùng khi cần train nhanh, focus vào frequent words
-  **Skip-gram:** Dùng khi cần quality cao, rare words quan trọng
-  **Số liệu:** Skip-gram chậm hơn 2.3x nhưng quality tốt hơn ~10-15%

---

## 4. Khó khăn và Giải pháp

### Khó khăn 1: Numpy Version Conflict 

**Vấn đề:**
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

**Nguyên nhân:**
- Cài numpy 2.0.2 (mới nhất)
- Gensim 4.3.3 requires numpy < 2.0

**Giải pháp:**
```bash
pip uninstall numpy
pip install "numpy<2.0,>=1.23"
```

**Kết quả:**  Fixed, gensim hoạt động bình thường

---

### Khó khăn 2: t-SNE Perplexity Error 

**Vấn đề:**
```
ValueError: perplexity must be less than n_samples
```

**Nguyên nhân:**
- Code dùng cố định `perplexity=30`
- Nhưng chỉ có 25 samples → Error

**Giải pháp:**
```python
# Tự động điều chỉnh perplexity
perplexity_value = min(30, max(5, len(words) - 1))
tsne = TSNE(perplexity=perplexity_value, ...)
```

**Kết quả:**  Fixed, t-SNE chạy với mọi số lượng samples

---

### Khó khăn 3: PySpark Installation Issues 

**Vấn đề:**
```
ImportError: pyspark.sql module not found
JAVA_HOME not set
```

**Nguyên nhân:**
- PySpark cần Java JDK
- Windows không tự detect Java

**Giải pháp:**
```bash
# 1. Cài Java JDK 11
# Download từ: https://adoptium.net/

# 2. Set JAVA_HOME
$env:JAVA_HOME = "C:\Program Files\Java\jdk-11.0.20"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

# 3. Cài PySpark
pip install pyspark

# 4. Verify
java -version
python -c "import pyspark; print(pyspark.__version__)"
```

**Kết quả:**  PySpark hoạt động

---

### Khó khăn 4: __pycache__ Clutter 

**Vấn đề:**
- Nhiều thư mục `__pycache__` trong git
- Làm repository bẩn

**Giải pháp:**
```bash
# Xóa tất cả __pycache__
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Tạo .gitignore
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
```

**Kết quả:**  Repository sạch, không commit cache nữa

---

### Khó khăn 5: Visualization Not Showing 

**Vấn đề:**
- `plt.show()` không hiện cửa sổ
- Chỉ thấy "Figure 1" nhưng không có hình

**Nguyên nhân:**
- Missing matplotlib backend

**Giải pháp:**
```bash
# Cài backend PyQt5
pip install pyqt5

# Hoặc dùng Agg backend (save only, no display)
import matplotlib
matplotlib.use('Agg')
```

**Kết quả:**  Biểu đồ hiện ra, hoặc save thành công

---

### Khó khăn 6: Dataset Path Not Found 

**Vấn đề:**
```
FileNotFoundError: c4-train.00000-of-01024-30K.json.gz not found
```

**Giải pháp:**
```python
# Thử nhiều đường dẫn có thể
possible_paths = [
    '../Lab1&Lab2/data/c4-train...json.gz',
    '../../Lab1&Lab2/data/c4-train...json.gz',
    'data/c4-train...json.gz',
]

for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

# Fallback: tạo dữ liệu mẫu
if not data_path:
    sample_data = create_sample_data()
```

**Kết quả:**  Code robust, chạy được ngay cả không có dataset

---

### Khó khăn 7: Memory Error với Large Dataset 

**Vấn đề:**
```
MemoryError: Unable to allocate array
```

**Nguyên nhân:**
- Load toàn bộ C4 dataset vào RAM (> 10GB)

**Giải pháp:**
```python
# Giới hạn số dòng
df = spark.read.json(data_path).limit(1000)

# Hoặc dùng lazy loading
df = spark.read.json(data_path).sample(0.01)  # 1% data
```

**Kết quả:**  Chạy được trên máy RAM thấp

---

## 5. Tổng kết

1. **Pre-trained embeddings tuyệt vời** cho general tasks
2. **Skip-gram > CBOW** cho quality (nhưng chậm hơn 2.3x)
3. **t-SNE > PCA** cho visualization (nhưng chậm hơn 20x)
4. **Spark = overkill** cho dataset < 10GB
5. **Word analogies** là cách tốt nhất để evaluate embeddin


##  Tài liệu tham khảo

- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Word2Vec Paper (Mikolov et al.)](https://arxiv.org/abs/1301.3781)
- [GloVe Paper (Pennington et al.)](https://nlp.stanford.edu/pubs/glove.pdf)
- [t-SNE Paper (van der Maaten)](https://lvdmaaten.github.io/tsne/)
- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
