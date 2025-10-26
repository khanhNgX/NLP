import os
import sys
import importlib.util

# Đảm bảo thư mục Lab1&Lab2 (chứa tokenizer/vectorizer) có thể import được
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
lab12_path = os.path.join(repo_root, 'Lab1&Lab2')
if lab12_path not in sys.path:
    sys.path.insert(0, lab12_path)

from preprocessing.regex_tokenizer import RegexTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import csv

import argparse
import os

# Tải động module TextClassifier từ Lab5/src/models
tc_path = os.path.join(repo_root, 'Lab5', 'src', 'models', 'text_classifier.py')
spec = importlib.util.spec_from_file_location("text_classifier", tc_path)
tc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc_mod)
TextClassifier = tc_mod.TextClassifier


def load_sentiments(path: str):
    """Đọc file CSV chứa cột text và sentiment. Trả về texts, labels.

    Thực hiện map cho nhãn: nếu sentiment là -1 -> 0, nếu 1/0 giữ nguyên.
    """
    texts = []
    labels = []
    if not os.path.exists(path):
        return texts, labels

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text') or row.get('Text') or row.get('sentence') or row.get('review')
            sentiment = row.get('sentiment') or row.get('label')
            if text is None or sentiment is None:
                continue
            sentiment = sentiment.strip()
            try:
                s = float(sentiment)
            except ValueError:
                # try to map common strings
                s = 1.0 if sentiment.lower() in ('positive', 'pos', '1') else 0.0

            # Map -1 -> 0, keep 1 as 1
            if s == -1.0:
                s = 0
            else:
                s = int(s)

            texts.append(text)
            labels.append(int(s))

    return texts, labels


def safe_split(X, y, test_size=0.33, random_state=42):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except Exception:
        # nếu stratify thất bại do dữ liệu nhỏ/không cân bằng, thử không stratify
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def run_with_texts(texts, labels, title="Sample test"):
    print(f"\n===== {title} =====")
    # Tiền xử lý: tách từ dùng RegexTokenizer (từ Lab1&Lab2)
    tokenizer = RegexTokenizer()
    tokenized_texts = [" ".join(tokenizer.tokenize(t)) for t in texts]

    # Chia dữ liệu (thử stratify, fallback nếu lỗi)
    X_train, X_test, y_train, y_test = safe_split(tokenized_texts, labels)

    # Dùng TfidfVectorizer của sklearn làm vectorizer
    vectorizer = TfidfVectorizer()

    # Khởi tạo bộ phân loại
    clf = TextClassifier(vectorizer)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    metrics = clf.evaluate(y_test, preds)

    # Chỉ in ra các chỉ số đánh giá (metrics)
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiments.csv'))
    parser.add_argument('--data_path', type=str, default=default_path, help='Đường dẫn csv chứa cột text và sentiment')
    parser.add_argument('--mode', type=str, choices=['sample', 'file', 'both'], default='both', help='Chạy sample nội bộ, file dữ liệu, hoặc cả hai')
    args = parser.parse_args()

    # Sample nội bộ
    sample_texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    sample_labels = [1, 0, 1, 0, 1, 0]

    if args.mode in ('sample', 'both'):
        run_with_texts(sample_texts, sample_labels, title='Sample test (6 docs)')

    if args.mode in ('file', 'both'):
        texts, labels = load_sentiments(args.data_path)
        if not texts:
            print(f"\nNo sentiments file found at {args.data_path}. Skipping file test.")
        else:
            run_with_texts(texts, labels, title=f'File test ({args.data_path})')


if __name__ == "__main__":
    main()
