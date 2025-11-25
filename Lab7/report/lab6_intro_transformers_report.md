# **Bài 1: Khôi phục Masked Token (Masked Language Modeling)**
## Kết quả chạy code:

```powershell
Câu gốc: Hanoi is the <mask> of Vietnam.
Dự đoán: ' capital' với độ tin cậy: 0.9341
 -> Câu hoàn chỉnh: Hanoi is the capital of Vietnam.
Dự đoán: ' Republic' với độ tin cậy: 0.0300
 -> Câu hoàn chỉnh: Hanoi is the Republic of Vietnam.
Dự đoán: ' Capital' với độ tin cậy: 0.0105
 -> Câu hoàn chỉnh: Hanoi is the Capital of Vietnam.
Dự đoán: ' birthplace' với độ tin cậy: 0.0054
 -> Câu hoàn chỉnh: Hanoi is the birthplace of Vietnam.
Dự đoán: ' heart' với độ tin cậy: 0.0014
 -> Câu hoàn chỉnh: Hanoi is the heart of Vietnam.
```

## Trả lời câu hỏi: 

### 1. Mô hình đã dự đoán đúng từ "capital" không?

Mô hình đã dự đoán đúng từ `capital` với độ tin cậy rất cao: **0.9341**.


### 2. Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?

 Kết quả `Hanoi is the <mask> of Vietnam` $\rightarrow$ `capital` là một ví dụ sách giáo khoa để giải thích sức mạnh vượt trội của kiến trúc **Encoder-only**.

#### A. Hiểu ngữ cảnh hai chiều (Bidirectional Context)

Khác biệt lớn nhất so với các mô hình RNN/LSTM truyền thống là khả năng xử lý ngữ cảnh:

* **Hạn chế của mô hình đơn hướng:** Để điền đúng từ vào `<mask>`, mô hình không thể chỉ đọc từ trái sang phải (`Hanoi is the...`). Nếu chỉ đọc chiều này, mô hình có thể điền là `"city"`, `"pride"`, `"largest city"`, v.v.
* **Sức mạnh của Encoder:** Mô hình nhìn thấy **cả hai phía cùng lúc**: Nó thấy chủ ngữ là **"Hanoi"** **VÀ** bổ ngữ phía sau là **"of Vietnam"**.
* **Kết quả:** Sự kết hợp giữa "Hanoi" và "Vietnam" tạo ra một mối liên kết ngữ nghĩa mạnh mẽ nhất là quan hệ thủ đô - đất nước. Chính việc **"nhìn thấy tương lai"** (từ "Vietnam") đã giúp nó loại bỏ các từ sai và chọn đúng từ `capital`.

#### B. Masked Language Modeling (MLM)

Mô hình đã được huấn luyện chuyên biệt để làm nhiệm vụ này:

* **Mục tiêu huấn luyện:** Mô hình đã được huấn luyện bằng cách che đi (mask) hàng tỷ từ trong văn bản và buộc phải **đoán lại chúng** dựa trên ngữ cảnh xung quanh.
* **Tính phù hợp:** Bài toán điền từ vào chỗ trống là *sở trường* gốc của mô hình, giúp nó nhạy bén hơn bất kỳ mô hình RNN nào.
* **Lưu ý kỹ thuật:** Việc bạn thấy ký hiệu `<mask>` (thay vì `[MASK]`) và các từ có khoảng trắng phía trước (ví dụ `' capital'`) cho thấy bạn có thể đang sử dụng mô hình **RoBERTa** (một biến thể được tối ưu hóa hơn của BERT).


# **Bài 2: Dự đoán từ tiếp theo (Next Token Prediction)**
## Kết quả chạy code:

```powershell
Câu mồi: 'The best thing about learning NLP is'
Văn bản được sinh ra:
The best thing about learning NLP is that it's an academic discipline in which the faculty members are very smart and creative, and the students are extremely interested in the field. The NLP faculty are very smart and creative.

So if you're an NLP learner, you're not going to teach the NLP course without some work. You're going to learn all the techniques that I would have worked on in my first semester, and you're going to learn all the techniques that I would have worked on in my second semester. How can you get this? No, you aren't going to teach the NLP!

This is something that you might not have even thought of. The problem with being a NLP student is that you're just not going to get the benefit of the NLP. How do you think you're going to get the benefit of the NLP?

I think that is a good question. I think that there are a lot of different ways to get back into the NLP. If you're going to study in the NLP, then you have to go back to school, or you have to go back to law school, or you have to go to college. So I think you can get back into the NLP if you're going to
```

## **Trả lời câu hỏi:**

### 1. Kết quả sinh ra có hợp lý không?

Đoạn văn bản được sinh ra không hợp lý về mặt ngữ nghĩa và logic. Tuy có ngữ pháp đúng nhưng mắc các lỗi nghiêm trọng về cấu trúc và ý nghĩa (điển hình của các mô hình RNN/LSTM không được tinh chỉnh hoặc chiến thuật sinh văn bản tham lam):

| Vấn đề | Phân tích chi tiết |
| :--- | :--- |
| **Lặp lại (Repetition)** | Mô hình bị mắc kẹt trong việc lặp lại các cụm từ hoặc ý tưởng gần kề. Ví dụ: *"The NLP faculty are very smart and creative."* (lặp lại ý câu trước) và lặp lại việc nhắc đến *"go back to school"* nhiều lần. |
| **Mâu thuẫn logic** | Mô hình tự mâu thuẫn trong lập luận. Ví dụ: Dù câu mồi bắt đầu bằng ý tích cực (*"The best thing..."*), mô hình lại sinh ra: *"The problem with being a NLP student is that you're just **not going to get the benefit of the NLP**."* (phủ định ý chính). |
| **Trôi chủ đề (Topic Drift)** | Mô hình mất khả năng duy trì ngữ cảnh dài. Nó đang nói về việc học NLP, sau đó đột ngột chuyển sang các ngành khác như *"law school"* (trường luật) mà không có sự chuyển tiếp hợp lý. |

---

### 2. Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này?

Tác vụ này là **Sinh văn bản tự do (Open-ended Text Generation)**, và các mô hình Decoder-only (như GPT, GPT-2, GPT-3) là kiến trúc tối ưu nhất nhờ vào đặc tính **tự hồi quy** và kiến trúc **Transformer**:

#### A. Sinh văn bản Tự hồi quy (Autoregressive Generation)
* Mô hình Decoder được thiết kế để dự đoán token tiếp theo ($\text{token}$) dựa trên tất cả các token đã được sinh ra trước đó.
* Quá trình này lặp đi lặp lại cho đến khi sinh ra đủ độ dài hoặc gặp token kết thúc chuỗi ($\text{<EOS>}$). Đây là cơ chế cơ bản để tạo ra văn bản tuần tự và mạch lạc.

#### B. Cơ chế Masked Self-Attention
* Trong kiến trúc Decoder, cơ chế **Self-Attention** được "che mặt" (Masked). Điều này đảm bảo rằng khi mô hình đang dự đoán một từ, nó **chỉ nhìn vào các từ đã xuất hiện** (phía bên trái), chứ không nhìn vào các từ "tương lai" chưa được sinh ra.
* Đây là điều kiện bắt buộc và lý tưởng cho tác vụ sinh văn bản (chúng ta không thể biết trước kết quả).

#### C. Ngữ cảnh dài và Tính mạch lạc
* Nhờ sử dụng kiến trúc **Transformer**, mô hình Decoder-only có thể xử lý và duy trì thông tin ngữ cảnh qua **hàng ngàn token** (độ dài cửa sổ ngữ cảnh), khắc phục hoàn toàn vấn đề **Trôi chủ đề** và **Lặp lại** của các mô hình RNN/LSTM khi sinh văn bản dài.
* Việc huấn luyện trên dữ liệu khổng lồ (Pre-training) giúp GPT học được logic, cấu trúc tường thuật, và cách đặt câu hỏi/trả lời một cách tự nhiên và có ý nghĩa.


# **Bài 3: Tính toán Vector biểu diễn của câu (Sentence Representation)**
## Kết quả chạy code:

```powershell
Vector biểu diễn của câu:
tensor([[-6.3875e-02, -4.2837e-01, -6.6779e-02, -3.8430e-01, -6.5785e-02,
         -2.1826e-01,  4.7636e-01,  4.8659e-01,  3.9689e-05, -7.4274e-02,
         -7.4740e-02, -4.7635e-01, -1.9773e-01,  2.4824e-01, -1.2162e-01,
          1.6678e-01,  2.1045e-01, -1.4576e-01,  1.2637e-01,  1.8635e-02,
          2.4640e-01,  5.7090e-01, -4.7014e-01,  1.3782e-01,  7.3650e-01,
         -3.3808e-01, -5.0329e-02, -1.6453e-01, -4.3517e-01, -1.2900e-01,
          1.6516e-01,  3.4004e-01, -1.4930e-01,  2.2422e-02, -1.0488e-01,
         -5.1916e-01,  3.2964e-01, -2.2162e-01, -3.4206e-01,  1.1993e-01,
         -7.0148e-01, -2.3126e-01,  1.1224e-01,  1.2550e-01, -2.5191e-01,
         -4.6374e-01, -2.7261e-02, -2.8415e-01, -9.9250e-02, -3.7018e-02,
         -8.9192e-01,  2.5005e-01,  1.5816e-01,  2.2701e-01, -2.8497e-01,
          4.5300e-01,  5.0921e-03, -7.9441e-01, -3.1008e-01, -1.7403e-01,
          4.3029e-01,  1.6816e-01,  1.0590e-01, -4.8987e-01,  3.1856e-01,
          3.2861e-01, -1.3403e-02,  1.8807e-01, -1.0905e+00,  2.1010e-01,
         -6.7579e-01, -5.7076e-01,  8.5946e-02,  1.9121e-01, -3.3818e-01,
          2.7744e-01, -4.0539e-01,  3.1305e-01, -4.1197e-01, -5.6820e-01,
         -3.9074e-01,  4.0747e-01,  9.9898e-02,  2.3719e-01,  1.0154e-01,
         -2.5670e-01, -2.0583e-01,  1.1763e-01, -5.1439e-01,  4.0979e-01,
          1.2149e-01,  1.9333e-02, -5.9029e-02, -2.0141e-01,  7.0860e-01,
         -6.4610e-02,  2.4780e-02, -9.0585e-03,  1.9667e-02,  3.0815e-01,
         -4.9832e-02, -1.0691e+00,  6.1072e-01, -4.9723e-02, -1.5156e-01,
         -6.7778e-02,  4.7811e-02,  5.2102e-01,  1.6951e-01,  1.0145e-02,
          5.3093e-01, -7.8189e-02,  6.5843e-02, -2.9383e-01, -4.6046e-01,
          4.2071e-01,  1.1822e-01,  2.3631e-01, -4.5379e-02, -1.3740e-01,
         -4.4018e-01, -6.8122e-02,  1.9934e-01,  8.7062e-01, -2.2603e-01,
          3.3604e-01,  2.0236e-01,  3.7898e-01,  1.9533e-01, -3.0366e-01,
          3.8633e-01,  6.1949e-01,  6.8663e-01, -1.8968e-01, -3.6815e-01,
         -1.6616e-01, -7.0828e-02, -3.4610e-01, -8.5325e-01,  4.6646e-02,
          2.8512e-01,  1.0890e-01,  2.5938e-01, -4.2975e-01,  4.3345e-01,
          2.0637e-01, -3.8656e-01, -3.8187e-02,  3.6925e-01,  3.0130e-01,
          4.0251e-01,  1.2887e-01, -3.7689e-01, -3.4447e-01, -4.2116e-01,
         -1.0252e-01, -8.9736e-02,  4.7384e-01,  8.1717e-02,  1.5885e-01,
          7.6674e-01,  3.4493e-01,  9.8523e-04,  4.8932e-02,  2.6132e-01,
          3.8330e-02, -2.0035e-01,  2.6654e-01,  9.3773e-02, -4.6780e-02,
         -4.0519e-01, -4.4310e-01,  6.1268e-01, -1.8950e-01, -3.8333e-01,
          2.0583e-01,  1.5379e-01, -1.4664e-01,  5.3847e-01, -3.9618e-01,
         -2.0599e+00,  6.7052e-01,  2.1112e-01, -4.7306e-01,  3.4865e-01,
         -2.9919e-01,  5.4614e-01, -5.3925e-01, -2.4877e-01, -2.9069e-02,
         -2.0319e-01, -7.3276e-02, -3.8147e-01, -5.4455e-01,  3.5050e-01,
         -1.1249e-01, -2.1471e-01, -3.8439e-01, -1.0760e-01, -8.8821e-02,
          2.5263e-01,  2.1448e-01,  5.5798e-02, -6.5411e-02,  9.9838e-02,
          3.3435e-01,  2.4018e-01,  2.9876e-02, -1.1191e-01,  5.4330e-01,
         -5.5214e-01,  1.1125e+00,  5.4141e-01, -7.4160e-02,  3.5337e-01,
          1.2313e-01,  3.4856e-02, -2.8568e-01, -1.2517e-01, -4.4332e-02,
          1.3323e-01, -2.4996e-01, -4.9834e-01,  4.1959e-01, -3.1580e-01,
          6.1942e-01,  3.1113e-01,  4.8846e-01,  6.1518e-01, -3.6327e-02,
          2.1295e-02, -3.5715e-01,  5.9126e-01,  1.5102e-01, -2.9641e-01,
          2.9441e-01, -1.4139e-01,  1.1662e-01, -3.6223e-01, -1.4621e-01,
          6.5255e-02,  3.9270e-01,  3.8543e-01, -2.3996e-01, -3.1482e-01,
         -4.6861e-01, -1.1920e-01,  8.6234e-02, -3.4597e-02, -3.6275e-01,
         -3.9838e-01, -3.6006e-01, -1.9672e-01, -2.7738e-01, -4.1097e-01,
          3.6456e-01, -2.6012e-01,  1.2587e-01,  1.2752e-01,  5.4261e-01,
          1.0569e-01,  3.5704e-01,  1.4766e-01,  4.4929e-01, -8.1255e-01,
         -3.0410e-02,  5.8064e-02,  2.0699e-01,  6.6129e-01,  3.9243e-01,
         -6.8644e-01, -8.3415e-01, -1.2653e-01,  1.9644e-01, -4.0900e-01,
         -6.3775e-02, -1.8780e-01,  7.9474e-02, -1.7443e-01,  3.1936e-01,
          3.6761e-01,  4.3044e-01, -1.7471e-01,  1.3718e-01,  1.4272e-01,
         -6.0643e-01,  2.3549e-01,  2.7794e-01,  1.0539e-01, -4.5836e-01,
         -3.2561e-01,  1.5292e-02, -2.7672e-01, -4.8611e-01,  3.9087e-01,
          3.6016e-01,  6.3403e-01, -1.2816e-01, -1.6719e-02, -3.0123e-01,
         -1.7321e-01, -6.7296e-01, -2.7015e-01, -1.2533e-01, -8.0565e-01,
          3.6115e-01,  1.7370e-01, -3.5578e-01, -2.1725e+00, -2.8103e-02,
         -2.6773e-02, -2.2444e-01,  3.1249e-02,  6.4419e-02, -1.5017e-01,
         -3.4460e-01, -5.5676e-01,  1.8039e-01, -4.2200e-01, -9.1074e-01,
         -3.1343e-03,  7.2439e-01,  3.9006e-01, -4.4128e-02, -4.4784e-02,
          2.8708e-02, -1.2432e-01,  6.9166e-01, -1.3226e-02, -2.3539e-02,
         -7.0616e-02, -4.5062e-01,  4.5705e-01,  3.3198e-01, -2.2727e-01,
          3.2434e-01, -4.5709e-01, -5.1586e-01, -1.5693e-01, -1.0897e-01,
          3.9317e-01, -2.5950e-01, -1.5326e-01,  3.3276e-01,  3.2522e-01,
         -2.5241e-01,  4.7946e-01, -3.7339e-01, -2.8146e-01,  7.7628e-02,
          2.7131e-01, -3.7212e-01,  6.1400e-01, -2.9269e-01, -4.4389e-01,
         -3.7750e-01,  2.7135e-01,  3.6869e-01, -1.6904e-01, -1.7583e-01,
          2.9626e-01,  2.9393e-01, -8.2027e-03,  3.4546e-02,  4.5846e-01,
          3.0137e-01,  1.6171e-01, -2.7772e-01,  5.2397e-01, -6.1950e-01,
         -2.4818e-02, -5.1944e-02,  3.6764e-01, -5.8404e-01, -2.6651e-01,
         -7.5761e-02, -1.7428e-01,  4.1535e-01, -2.7556e-01, -5.6794e-02,
         -4.3509e-01, -9.6659e-01, -1.1799e-01, -3.8004e-01,  2.7555e-01,
         -2.9743e-01,  2.4023e-01, -3.8869e-01, -4.0248e-01, -8.3882e-01,
         -1.0652e-01, -9.4193e-02,  1.4810e-01,  9.0843e-03,  1.4658e-01,
         -1.4813e-01, -1.6078e-01, -4.3130e-01, -8.0683e-02,  4.3722e-01,
          4.2623e-01,  3.3201e-01, -2.8283e-01,  2.0751e-01,  5.9093e-01,
         -6.3454e-01,  5.7386e-01, -2.9870e-01,  1.0221e-02, -4.7624e-01,
          4.9509e-01,  4.7470e-02,  1.3193e-01,  3.6281e-01, -1.1642e+00,
          3.8372e-01,  1.7071e-01,  3.8881e-01,  1.7703e-01, -4.7019e-01,
          1.2768e-01, -1.3409e-01, -2.8794e-01,  3.2066e-01, -3.7853e-01,
          4.6259e-01,  5.2343e-01,  3.0741e-01,  2.7410e-01,  4.9933e-01,
         -5.6466e-01, -3.4677e-01, -6.6572e-01, -1.3347e-01, -8.5910e-02,
          6.2486e-02, -3.9922e-01, -3.5880e-01, -5.8337e-01, -1.3556e-02,
         -1.6812e-01,  1.3949e-01,  2.9142e-01, -4.5623e-01, -1.0705e-01,
          6.6569e-01,  7.6614e-01, -1.9306e-01,  4.3854e-01,  2.8110e-01,
         -3.6836e-01, -1.6012e-01, -2.5005e-01,  7.6297e-01,  1.9653e-01,
         -1.8120e-01,  1.1884e-03,  1.8755e-01, -1.8990e-01, -2.3725e-01,
          3.2633e-02, -2.7723e-01, -4.7987e-02, -6.2332e-01,  2.6807e-01,
         -1.2293e-01, -2.7098e-01, -6.9677e-01,  1.5738e-01,  5.3557e-01,
          1.2760e-01, -1.7979e-02,  1.2769e-01, -5.6452e-02,  6.7964e-02,
          1.8555e-01, -3.6374e-01,  2.8518e-01, -4.3920e-01, -2.4276e-01,
          5.1755e-01, -2.3519e-01,  6.4010e-02,  3.9268e-01,  5.7986e-01,
         -1.7500e-01,  7.1670e-02,  5.7915e-01,  5.1699e-02, -1.1077e-03,
         -4.8444e-02,  1.5531e-01,  2.8402e-01,  6.8268e-01,  8.1525e-02,
          1.5325e-01,  1.9466e-01,  1.2260e-02, -3.3223e-01,  2.5763e-02,
         -1.6071e-01, -3.7663e-01, -7.3670e-01, -5.0067e-01,  1.1540e-01,
         -3.3789e-01,  1.2889e-01,  2.1528e-02,  6.1149e-01,  3.3549e-01,
         -2.0217e-01, -6.3961e-02,  2.4056e-02, -9.3071e-02, -2.7770e-02,
          1.8373e-01, -4.1812e-02, -1.0456e-01, -2.7569e-01, -3.9216e-01,
         -3.2092e-01, -1.0158e+00,  1.6407e-01,  4.5044e-02,  2.3079e-01,
          2.6935e-02, -2.1047e-01, -3.1392e-01, -4.6154e-01, -4.0347e-01,
          7.3271e-02,  1.1470e-01, -2.4129e-01, -3.6199e-01, -5.3254e-01,
         -5.2185e-01, -4.0713e-01,  2.1619e-02,  1.4186e-01, -1.2105e-01,
         -1.4054e-02, -4.2986e-02, -1.2459e-01, -6.6652e-01, -6.4169e-01,
         -2.2399e-01,  6.2557e-02, -3.3323e-01,  1.8866e-02,  1.6464e-01,
         -2.8729e-02, -5.9477e-01,  2.0963e-02, -3.3761e-01,  1.8089e-01,
          7.4362e-01,  1.5554e-01,  2.7824e-01, -2.1975e-01,  5.1316e-01,
         -3.9708e-01, -2.4769e-01,  4.3027e-01, -2.3078e-01, -2.9392e-01,
          1.3250e-01, -6.1646e-01,  2.6501e-01,  5.6891e-01, -1.3585e-01,
         -1.2774e-01,  8.1189e-01,  3.6497e-01,  5.0179e-01,  2.9736e-01,
          8.7772e-01,  7.3390e-02,  2.5788e-01, -3.3609e-01,  8.8206e-02,
          2.1283e-02,  1.4487e-01,  7.6685e-03, -3.9123e-01, -6.3920e-02,
         -3.7236e-01,  8.2941e-02,  3.0822e-02,  3.1529e-02,  2.0262e-01,
         -5.0066e-01, -1.2373e-01,  2.2661e-01,  1.6069e-01, -3.6415e-01,
          2.3418e-01, -1.6900e-01, -1.3540e-01, -1.6678e-01,  1.5227e-01,
         -2.6064e-01,  4.4843e-02, -3.4591e-02, -1.2043e-01,  6.4725e-01,
          4.8944e-01, -3.0347e-01, -2.3118e-01, -8.3765e-02,  2.2163e-01,
          1.0404e-01,  1.3495e-01, -5.3097e-01,  1.4525e-01,  4.9890e-01,
         -4.9265e-01,  3.7358e-01,  2.2077e-01, -5.4249e-02, -6.7142e-02,
          6.2195e-01,  4.6524e-01, -4.2303e-01, -3.2715e-01,  3.8370e-01,
         -5.7111e-01, -1.6922e-01,  4.2353e-01, -2.0156e-01, -1.2482e-01,
          4.3334e-01, -4.0270e-02, -5.8664e-01,  7.2658e-01, -5.5645e-01,
         -5.7467e-02, -2.1052e-01,  1.0038e-01, -2.5424e-03,  7.7563e-01,
         -3.9355e-01,  6.4184e-01, -5.9658e-01,  2.1974e-02,  1.8323e-01,
          1.7593e-01,  4.8541e-01, -4.6240e-01,  3.5692e-01,  3.2622e-01,
         -2.0756e-01,  5.7904e-01, -2.7194e-01, -5.2925e-01,  7.4888e-02,
         -2.6069e-02,  3.5997e-01,  5.5750e-01,  3.2160e-01,  4.0078e-01,
          5.1017e-01, -4.6596e-02,  2.9056e-01,  2.4928e-01,  2.0993e-01,
          4.9611e-01, -4.1696e-02, -1.5711e-01,  1.5638e-01,  8.1301e-02,
          3.2565e-01, -2.6684e-01, -2.1355e-01,  1.9676e-01,  4.6960e-01,
          1.5972e-01, -2.5917e-01, -1.0547e-01,  1.3562e-01,  3.5989e-01,
         -1.0882e-01, -7.1565e-02, -5.3039e-01,  8.8760e-01, -3.4283e-01,
         -5.0052e-02, -4.8836e-01,  2.0944e-01,  2.6859e-01,  4.4361e-01,
         -4.6622e-01, -1.3640e-01, -1.4363e-01, -3.5663e-01, -1.1210e-01,
         -1.9890e-01, -1.2909e-01, -3.0802e-03, -6.2016e-02, -4.2345e-01,
          2.7059e-01, -3.1317e-01,  5.7516e-01, -2.2525e-03,  1.7034e-01,
          3.9410e-01,  8.1126e-01, -3.6260e-01,  5.2088e-01, -5.4591e-01,
         -5.8636e-02,  1.5576e-01,  1.7441e-01,  1.3422e-01, -4.4369e-01,
          2.6824e-01, -2.6424e-01, -5.6735e-01,  2.7223e-01,  5.5829e-01,
         -9.1909e-01,  2.2039e-01, -3.5612e-01,  1.3164e-01, -1.1517e-01,
         -2.0684e-01, -2.7872e-02,  3.9112e-01, -6.6897e-01, -3.8353e-01,
         -5.6090e-02,  8.0477e-01, -2.5700e-01, -1.0725e-01,  7.5041e-02,
          2.4736e-01, -6.1457e-01, -1.9508e-01,  5.4607e-01,  3.3887e-01,
          2.7338e-01,  4.4597e-01,  4.4805e-01, -7.3450e-01,  2.2959e-01,
         -3.8095e-02, -1.4963e-01, -2.4957e-01, -2.8457e-01,  5.6483e-01,
          5.4733e-02,  8.0650e-02, -1.2184e+00,  5.7510e-01,  1.3625e-01,
         -4.4055e-01,  6.9751e-02, -4.0260e-01,  1.0932e-01, -6.6830e-02,
         -3.9554e-02, -5.4193e-01, -4.4191e-01,  2.4927e-01,  6.6517e-01,
         -1.7534e-01, -1.2388e-01,  3.1970e-01]])

Kích thước của vector: torch.Size([1, 768])
```

## **Trả lời câu hỏi:**

### 1. Kích thước (chiều) của vector biểu diễn là bao nhiêu? Tương ứng tham số nào?

* **Kích thước:** Chiều của vector là **768**.
    * Bạn có thể thấy điều này trong `torch.Size([1, 768])`, trong đó `1` là batch size và `768` là số chiều đặc trưng.
* **Tham số tương ứng:** Con số này tương ứng với tham số **`hidden_size`** (kích thước lớp ẩn) trong kiến trúc của mô hình **BERT Base**.
    * Mô hình BERT Base có cấu trúc: 12 lớp (layers), 12 đầu attention (heads), và **768 đơn vị ẩn (hidden units)**.


### 2. Tại sao cần sử dụng `attention_mask` khi thực hiện Mean Pooling?

Mục đích chính: Để **loại bỏ ảnh hưởng của các token đệm (padding tokens)**, đảm bảo tính chính xác của vector đại diện.

**Giải thích chi tiết:**
1.  **Vấn đề Padding:** Khi xử lý theo batch, các câu ngắn phải thêm token `[PAD]` để có cùng độ dài với câu dài nhất.
2.  **Nếu không có Mask:** Việc tính trung bình cộng (Mean) cả các vector của token `[PAD]` sẽ làm vector đại diện câu bị sai lệch và "loãng" thông tin (vì token `[PAD]` không mang nghĩa).
3.  **Vai trò của `attention_mask`:**
    * Mask chứa giá trị **1** cho token thật và **0** cho token padding.
    * Khi tính toán, ta chỉ tổng hợp các vector có mask là 1 và chia cho tổng số lượng token thật (thay vì chia cho tổng độ dài chuỗi).












