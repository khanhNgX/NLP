# Lab6 — PyTorch basics

## Nội dung đã thực hiện
- Phần 1 — Khám phá Tensor
  - Tạo tensor từ list và từ NumPy array.
  - Tạo tensor constant (`ones_like`) và random (`rand_like`).
  - In `shape`, `dtype`, `device`.
  - Các phép toán cộng, nhân, nhân ma trận (`@`) và indexing/slicing.
  - Thay đổi hình dạng bằng `view` / `reshape`.

- Phần 2 — Autograd (tự động tính đạo hàm)
  - Ví dụ tính đạo hàm đơn giản với `requires_grad=True`.
  - Trình bày nguyên lý: PyTorch xây dựng đồ thị tính toán (compute graph) khi ta thực hiện forward trên các tensor có `requires_grad=True`.
  - Minh hoạ các trường hợp:
    - Gọi `backward()` hai lần trên cùng một đồ thị mà **không** dùng `retain_graph=True` sẽ gây `RuntimeError` (các giá trị trung gian đã bị giải phóng).
    - Dùng `backward(retain_graph=True)` để giữ đồ thị và có thể gọi backward lần nữa — lưu ý gradient sẽ được cộng dồn vào `.grad` trừ khi ta reset bằng `x.grad.zero_()`.
    - Cách an toàn hơn: tính lại forward (recompute) rồi `backward()` — không giữ đồ thị cũ, tiêu thụ ít bộ nhớ hơn.
    - Tính đạo hàm bậc hai bằng `torch.autograd.grad(..., create_graph=True)`.

- Phần 3 — `torch.nn` (mạng neural cơ bản)
  - Ví dụ `nn.Linear` chuyển đổi tuyến tính (in shape input/output).
  - Ví dụ `nn.Embedding` ánh xạ chỉ số thành vector embedding.
  - Định nghĩa một `nn.Module` đơn giản (`MyFirstModel`) kết hợp Embedding → Linear → ReLU → Output.

## File thực thi
- `Lab6/lab6.ipynb` chứa toàn bộ các ví dụ trên. Chạy bằng Colab: Chạy từng ô lệnh.

Lưu ý: script kiểm tra việc import `torch` và sẽ in hướng dẫn cài đặt nếu PyTorch chưa được cài.

## Vấn đề thường gặp: RuntimeError khi gọi backward nhiều lần
Mô tả lỗi bạn có thể gặp (đã xuất hiện trong notebook):

```
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
```

Nguyên nhân và cách xử lý:
- Nguyên nhân: PyTorch giải phóng các tensor trung gian của đồ thị tính toán sau lần gọi `.backward()` để tiết kiệm bộ nhớ. Vì vậy không thể tái sử dụng cùng đồ thị để backward lần nữa.
- Cách khắc phục:
  1. Nếu thực sự cần backward nhiều lần trên cùng đồ thị: dùng `z.backward(retain_graph=True)` — nhưng lưu ý tốn bộ nhớ; và gradient sẽ cộng dồn (dùng `x.grad.zero_()` để reset).
  2. Thông thường khuyến nghị: tái tính toán forward (recompute) để tạo đồ thị mới rồi gọi `.backward()` lần nữa; cách này tiết kiệm bộ nhớ.
  3. Nếu cần đạo hàm bậc cao (higher-order derivatives), dùng `create_graph=True` khi gọi `autograd.grad` để giữ graph cho bước tính đạo hàm tiếp theo.

## Kết quả chạy (ví dụ đã thu được khi chạy trên môi trường có torch và dữ liệu mẫu)
- Ví dụ autograd: với `x = 1.0`, `z = 3*(x+2)**2`:
  - `dz/dx` = 18.
  - Gọi `backward()` hai lần mà không `zero_()` sẽ làm `x.grad` tăng gấp đôi (18 → 36) vì gradient được cộng dồn.


## Gợi ý mở rộng
- Thử thay `sentence averaging` bằng các pooling khác khi dùng embeddings.
- Thử tính đạo hàm bậc hai / higher-order và minh hoạ với ví dụ `create_graph=True`.

---
