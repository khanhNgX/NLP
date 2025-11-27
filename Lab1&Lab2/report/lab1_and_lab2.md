# Báo cáo Lab 1 & Lab 2

## Mô tả công việc

### Lab 1
- Cài đặt `SimpleTokenizer`: Tokenizer đơn giản chuyển văn bản về chữ thường, tách từ dựa trên khoảng trắng và tách các dấu câu cơ bản bằng cách chèn khoảng trắng vào trước và sau nó.
- Cài đặt `RegexTokenizer`: Tokenizer sử dụng biểu thức chính quy để tách từ và dấu câu, giúp tách chính xác hơn các trường hợp đặc biệt.

### Lab 2
- Cài đặt `CountVectorizer`: Biểu diễn văn bản theo mô hình Bag-of-Words, xây dựng từ vựng và chuyển đổi tài liệu thành vector đếm, tích hợp khả năng sử dụng tokenizer từ Lab 1 vào CountVectorizer.

## Kết quả chạy code

### Kết quả tokenizer trên các câu mẫu

**SimpleTokenizer:**
<pre>
Input: Hello, world! This is a test.
Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input: NLP is fascinating... isn't it?
Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Input: Let's see how it handles 123 numbers and punctuation!
Tokens: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

**RegexTokenizer:**
Input: Hello, world! This is a test.
Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input: NLP is fascinating... isn't it?
Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Input: Let's see how it handles 123 numbers and punctuation!
Tokens: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
</pre>

### Kết quả tokenizer trên bộ dữ liệu UD_English-EWT
<pre>
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of ...
SimpleTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
</pre>

### Kết quả CountVectorizer trên corpus mẫu
<pre>
Vocabulary: {'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}

Tokenized Corpus:
i love nlp .
i love programming .
nlp is a subfield of ai .

Document-Term Matrix:
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
</pre>
## Giải thích kết quả
- `SimpleTokenizer` chỉ tách từ dựa trên khoảng trắng và dấu câu cơ bản, có thể bỏ sót các trường hợp đặc biệt.
- `RegexTokenizer` sử dụng regex nên tách từ và dấu câu chính xác hơn, đặc biệt với các ký tự đặc biệt hoặc số.
- `CountVectorizer` xây dựng từ vựng dựa trên toàn bộ corpus, mỗi tài liệu được biểu diễn thành vector đếm số lần xuất hiện của từng từ trong từ vựng.
- `SimpleTokenizer` và `RegexTokenizer` không khác nhau về kết quả nhưng `RegexTokenizer` sẽ triển khai thuật toán tốt hơn với những câu phức tạp
- Khó khăn: Xử lý encoding của file dataset lớn, đồng bộ tokenizer với vectorizer, kiểm tra kết quả đầu ra.
- Cách giải quyết: Sử dụng encoding 'utf-8', kiểm thử từng bước, thêm chú thích và kiểm tra kỹ các trường hợp đặc biệt.

---
