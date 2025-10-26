from typing import List, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    """Bộ phân loại văn bản đơn giản dùng vectorizer bên ngoài
    (có phương thức fit_transform và transform) và mô hình LogisticRegression
    từ scikit-learn.
    """

    def __init__(self, vectorizer) -> None:
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """Huấn luyện vectorizer rồi huấn luyện mô hình LogisticRegression.

        Tham số:
            texts: danh sách văn bản thô.
            labels: danh sách nhãn tương ứng (số nguyên).
        """
        # Lấy ma trận đặc trưng từ vectorizer (có thể là dense hoặc sparse)
        X = self.vectorizer.fit_transform(texts)

        model = LogisticRegression(solver="liblinear")
        model.fit(X, labels)
        self._model = model

    def predict(self, texts: List[str]) -> List[int]:
        """Dự đoán nhãn cho một danh sách văn bản.

        Trả về danh sách nhãn (số nguyên).
        """
        if self._model is None:
            raise ValueError("Mô hình chưa được huấn luyện. Gọi fit() trước.")

        # Lấy ma trận đặc trưng cho dữ liệu mới (giữ dạng sparse nếu vectorizer trả về)
        X = self.vectorizer.transform(texts)
        preds = self._model.predict(X)
        return preds.tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Tính các chỉ số: accuracy, precision, recall và F1-score.

        Trả về dict với các khóa: accuracy, precision, recall, f1.
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        return metrics
