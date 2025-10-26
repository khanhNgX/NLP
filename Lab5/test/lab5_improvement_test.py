"""So sánh cải tiến: tiền xử lý + logistic regression vs MultinomialNB.

Kỹ thuật áp dụng:
- Tiền xử lý sạch (loại URL, HTML tag, ký tự đặc biệt, lowercase)
- Giảm kích thước đặc trưng bằng `max_features` trong TfidfVectorizer
- So sánh hai mô hình: LogisticRegression (baseline) và MultinomialNB

Chạy:
    python Lab5/test/lab5_improvement_test.py
"""
from typing import List
import re
import numpy as np
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os
import argparse

# Thử import gensim (tùy chọn). Nếu không có, sẽ fallback về TF-IDF và in hướng dẫn cài đặt.
try:
    import gensim
    from gensim.models import Word2Vec
    from gensim.models import KeyedVectors
    _HAS_GENSIM = True
except Exception:
    Word2Vec = None
    KeyedVectors = None
    _HAS_GENSIM = False


def clean_text(text: str) -> str:
    """Tiền xử lý đơn giản: loại URL, HTML, ký tự đặc biệt, chuyển về lowercase."""
    if text is None:
        return ""
    # loại URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # loại thẻ HTML
    text = re.sub(r"<.*?>", " ", text)
    # chỉ giữ chữ, số và khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)
    # nhiều khoảng trắng -> 1
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def evaluate(y_true: List[int], y_pred: List[int]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_word2vec_on_corpus(tokenized_texts, vector_size=100, window=5, min_count=1, workers=1, epochs=5):
    """Huấn luyện mô hình Word2Vec trên corpus tokenized (list[list[str]]).

    Trả về object gensim Word2Vec.
    """
    if not _HAS_GENSIM:
        raise RuntimeError("gensim không được cài đặt. Cài đặt bằng: pip install gensim")

    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window,
                     min_count=min_count, workers=workers)
    # epochs có thể gọi .train nếu muốn, nhưng khởi tạo với sentences đã train một lần
    try:
        model.train(tokenized_texts, total_examples=len(tokenized_texts), epochs=epochs)
    except Exception:
        # một số phiên bản gensim không cần hoặc không hỗ trợ train lại như thế này
        pass
    return model


def load_pretrained_embeddings(path: str, binary: bool = False):
    """Tải embeddings đã huấn luyện (word2vec hoặc glove được chuyển đổi).

    Trả về (KeyedVectors, vector_size)
    """
    if not _HAS_GENSIM:
        raise RuntimeError("gensim không được cài đặt. Cài đặt bằng: pip install gensim")

    # Nếu có từ 'glove' trong tên file và không phải dạng word2vec, người dùng nên chuyển đổi
    # Nhưng thử load thẳng first - nhiều file pretrained của FastText/GloVe có định dạng .txt tương thích
    kv = None
    try:
        kv = KeyedVectors.load_word2vec_format(path, binary=binary, unicode_errors='ignore')
    except Exception:
        # thử convert nếu có sẵn glove2word2vec
        try:
            from gensim.scripts.glove2word2vec import glove2word2vec
            tmp_w2v = path + '.w2v'
            glove2word2vec(path, tmp_w2v)
            kv = KeyedVectors.load_word2vec_format(tmp_w2v, binary=False)
            # không xóa file tạm để tránh lỗi trên Windows nếu dùng sau
        except Exception as e:
            raise RuntimeError(f"Không thể load pretrained embeddings từ {path}: {e}")
    return kv, kv.vector_size


def sentence_average_vector_from_model(word_vector_model, tokens: List[str], vector_size: int):
    """Tính vector trung bình cho một câu từ mô hình Word2Vec/KeyedVectors.

    Nếu từ không có trong vocab, bỏ qua nó. Nếu không có từ nào hợp lệ, trả về zero vector.
    """
    vecs = []
    # KeyedVectors and Word2Vec have similar interfaces via .wv
    wv = getattr(word_vector_model, 'wv', word_vector_model)
    for w in tokens:
        if w in wv:
            try:
                vecs.append(wv[w])
            except Exception:
                # một số trường hợp với fastText trả về lỗi truy xuất
                continue
    if not vecs:
        return np.zeros(vector_size, dtype=float)
    return np.mean(np.vstack(vecs), axis=0)


def load_sentiments(path: str):
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
                s = 1.0 if sentiment.lower() in ('positive', 'pos', '1') else 0.0

            if s == -1.0:
                s = 0
            else:
                s = int(s)

            texts.append(text)
            labels.append(int(s))
    return texts, labels


def safe_split(X, y, test_size=0.2, random_state=42):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def main():
    parser = argparse.ArgumentParser()
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiments.csv'))
    # Mặc định: chạy cả sample và file (nếu file tồn tại) — chỉ cần chạy `python test\lab5_improvement_test.py` để có cả 2 kết quả
    parser.add_argument('--data_path', type=str, default=default_path, help='Đường dẫn csv chứa cột text và sentiment')
    parser.add_argument('--mode', type=str, choices=['sample', 'file', 'both'], default='both', help='Chạy sample nội bộ, file dữ liệu, hoặc cả hai')
    parser.add_argument('--embed-method', type=str, choices=['tfidf', 'word2vec', 'pretrained'], default='pretrained', help='Phương pháp biểu diễn: tfidf|word2vec|pretrained')
    parser.add_argument('--pretrained-path', type=str, default=None, help='Đường dẫn đến file embeddings đã huấn luyện (word2vec/GloVe chuyển đổi)')
    parser.add_argument('--w2v-dim', type=int, default=100, help='Kích thước embedding khi train word2vec')
    parser.add_argument('--w2v-epochs', type=int, default=5, help='Số epoch khi train word2vec (demo)')
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

    def run_improvement(texts, labels, title="Sample test", embed_method='pretrained', pretrained_path=None, w2v_dim=100, w2v_epochs=5):
        print(f"\n===== {title} =====")
        cleaned = [clean_text(t) for t in texts]

        # Nếu user vẫn chọn tfidf, chuyển sang pretrained (theo yêu cầu: bỏ tfidf)
        if embed_method == 'tfidf':
            print("TF-IDF đã bị bỏ. Chuyển sang pretrained (GloVe pseudo) theo yêu cầu.")
            embed_method = 'pretrained'

        # Chia dữ liệu trước khi train bất kỳ embedding nào (tránh data leakage)
        X_train, X_test, y_train, y_test = safe_split(cleaned, labels)

        # Helper: tokenization đơn giản (đã clean, nên split whitespace đủ cho demo)
        def tokenize(sentences):
            return [s.split() for s in sentences]

        # Nếu dùng TF-IDF (mặc định cũ)
        if embed_method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # MultinomialNB phù hợp với TF-IDF/count features
            nb = MultinomialNB()
            nb.fit(X_train_vec, y_train)
            preds_nb = nb.predict(X_test_vec)
            metrics_nb = evaluate(y_test, preds_nb)

            def to_dense_if_needed(X):
                toarr = getattr(X, "toarray", None)
                if callable(toarr):
                    return X.toarray()
                return X

            X_train_dense = to_dense_if_needed(X_train_vec)
            X_test_dense = to_dense_if_needed(X_test_vec)

            # GBT
            gbt = GradientBoostingClassifier(n_estimators=100)
            try:
                gbt.fit(X_train_dense, y_train)
                preds_gbt = gbt.predict(X_test_dense)
                metrics_gbt = evaluate(y_test, preds_gbt)
            except Exception:
                metrics_gbt = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

            # MLP
            mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
            try:
                mlp.fit(X_train_dense, y_train)
                preds_mlp = mlp.predict(X_test_dense)
                metrics_mlp = evaluate(y_test, preds_mlp)
            except Exception:
                metrics_mlp = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

            # In metrics
            print("\nMetrics MultinomialNB:")
            for k, v in metrics_nb.items():
                print(f"{k}: {v:.4f}")

            print("\nMetrics GradientBoosting:")
            for k, v in metrics_gbt.items():
                print(f"{k}: {v:.4f}")

            print("\nMetrics MLPClassifier:")
            for k, v in metrics_mlp.items():
                print(f"{k}: {v:.4f}")

        elif embed_method in ('word2vec', 'pretrained'):
            # embeddings dense: chỉ dùng các mô hình chấp nhận dense numeric features
            # Tokenize
            X_train_tokens = tokenize(X_train)
            X_test_tokens = tokenize(X_test)

            model = None
            vec_size = w2v_dim
            try:
                if embed_method == 'word2vec':
                    model = train_word2vec_on_corpus(X_train_tokens, vector_size=w2v_dim, epochs=w2v_epochs)
                    vec_size = w2v_dim
                else:  # pretrained
                        if not pretrained_path:
                            # Nếu chạy sample và user chọn pretrained nhưng không cho file,
                            # tạo các vector GloVe demo (deterministic pseudo-glove) để minh hoạ.
                            def pseudo_glove_vector(word, dim):
                                # Tạo vector deterministically từ hash của từ
                                h = int(hashlib.sha1(word.encode('utf-8')).hexdigest()[:8], 16)
                                rng = np.random.RandomState(h % (2**32))
                                return rng.normal(scale=0.5, size=(dim,)).astype(float)

                            vec_size = w2v_dim
                            model = None
                            # Bind dimension so later calls can be pseudo_glove(word)
                            pseudo_glove = lambda w: pseudo_glove_vector(w, vec_size)
                        else:
                            kv, vec_size = load_pretrained_embeddings(pretrained_path)
                            model = kv
            except Exception as e:
                print(f"Embedding setup failed: {e}. Falling back to TF-IDF.")
                return run_improvement(texts, labels, title=title, embed_method='tfidf')

            # Build average vectors
            if model is None:
                # Use pseudo_glove for demo (no gensim required)
                X_train_vec = np.vstack([np.mean([pseudo_glove(t) for t in toks], axis=0) if toks else np.zeros(vec_size) for toks in X_train_tokens])
                X_test_vec = np.vstack([np.mean([pseudo_glove(t) for t in toks], axis=0) if toks else np.zeros(vec_size) for toks in X_test_tokens])
            else:
                X_train_vec = np.vstack([sentence_average_vector_from_model(model, toks, vec_size) for toks in X_train_tokens])
                X_test_vec = np.vstack([sentence_average_vector_from_model(model, toks, vec_size) for toks in X_test_tokens])

            # Logistic Regression (works with dense)
            try:
                lr = LogisticRegression(max_iter=300)
                lr.fit(X_train_vec, y_train)
                preds_lr = lr.predict(X_test_vec)
                metrics_lr = evaluate(y_test, preds_lr)
            except Exception as e:
                metrics_lr = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

            # Gradient Boosting
            try:
                gbt = GradientBoostingClassifier(n_estimators=100)
                gbt.fit(X_train_vec, y_train)
                preds_gbt = gbt.predict(X_test_vec)
                metrics_gbt = evaluate(y_test, preds_gbt)
            except Exception:
                metrics_gbt = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

            # MLP
            try:
                mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
                mlp.fit(X_train_vec, y_train)
                preds_mlp = mlp.predict(X_test_vec)
                metrics_mlp = evaluate(y_test, preds_mlp)
            except Exception:
                metrics_mlp = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

            # In ra kết quả
            print("\nMetrics LogisticRegression (embeddings):")
            for k, v in metrics_lr.items():
                print(f"{k}: {v:.4f}")

            print("\nMetrics GradientBoosting (embeddings):")
            for k, v in metrics_gbt.items():
                print(f"{k}: {v:.4f}")

            print("\nMetrics MLPClassifier (embeddings):")
            for k, v in metrics_mlp.items():
                print(f"{k}: {v:.4f}")

        else:
            print(f"Unknown embed_method: {embed_method}. Skipping.")

    if args.mode in ('sample', 'both'):
        run_improvement(sample_texts, sample_labels, title='Sample improvement test (6 docs)', embed_method=args.embed_method, pretrained_path=args.pretrained_path, w2v_dim=args.w2v_dim, w2v_epochs=args.w2v_epochs)

    if args.mode in ('file', 'both'):
        texts, labels = load_sentiments(args.data_path)
        if not texts:
            print(f"\nNo sentiments file found at {args.data_path}. Skipping file test.")
        else:
            run_improvement(texts, labels, title=f'File improvement test ({args.data_path})', embed_method=args.embed_method, pretrained_path=args.pretrained_path, w2v_dim=args.w2v_dim, w2v_epochs=args.w2v_epochs)


if __name__ == '__main__':
    main()
