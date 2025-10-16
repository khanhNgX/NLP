"""
Lab 4 Bonus: Huan luyen mo hinh Word2Vec tu dau
Demo huan luyen mo hinh Word2Vec tren du lieu cu the (UD English-EWT)

Mo hinh Word2Vec co 2 kien truc:
1. CBOW (Continuous Bag of Words): Du doan tu tu cac tu xung quanh
2. Skip-gram: Du doan cac tu xung quanh tu mot tu

Demo nay su dung ca hai va so sanh ket qua.
"""
import sys
import os

# Them duong dan goc du an vao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gensim.models import Word2Vec
import time
import re
from collections import Counter


def load_text_data(file_path):
    """
    Doc du lieu van ban tu file.
    
    Args:
        file_path: Duong dan den file du lieu
        
    Returns:
        Danh sach cac cau (moi cau la list cac tu)
    """
    sentences = []
    current_sentence = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Bo qua dong trong hoac dong comment
                if not line or line.startswith('#'):
                    continue
                
                # Kiem tra dinh dang CoNLL-U
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        # Lay tu (cot thu 2 trong dinh dang CoNLL-U)
                        word = parts[1].lower()
                        # Loai bo dau cau va ky tu dac biet
                        if re.match(r'^[a-z0-9]+$', word):
                            current_sentence.append(word)
                else:
                    # Dong trong danh dau ket thuc cau
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
            
            # Them cau cuoi cung neu con
            if current_sentence:
                sentences.append(current_sentence)
                
    except FileNotFoundError:
        print(f"Loi: Khong tim thay file {file_path}")
        return []
    
    return sentences


def load_simple_text(file_path):
    """
    Doc du lieu van ban don gian (moi dong la mot cau).
    
    Args:
        file_path: Duong dan den file
        
    Returns:
        Danh sach cac cau
    """
    sentences = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Loai bo dau cau va chuyen thanh chu thuong
                    words = re.findall(r'\b[a-z0-9]+\b', line.lower())
                    if words:
                        sentences.append(words)
    except FileNotFoundError:
        print(f"Loi: Khong tim thay file {file_path}")
        return []
    
    return sentences


def analyze_corpus(sentences):
    """
    Phan tich thong ke corpus.
    
    Args:
        sentences: Danh sach cac cau
        
    Returns:
        Dictionary chua cac thong ke
    """
    # Dem tu
    word_counts = Counter()
    total_words = 0
    
    for sentence in sentences:
        word_counts.update(sentence)
        total_words += len(sentence)
    
    stats = {
        'num_sentences': len(sentences),
        'total_words': total_words,
        'unique_words': len(word_counts),
        'avg_sentence_length': total_words / len(sentences) if sentences else 0,
        'most_common': word_counts.most_common(10)
    }
    
    return stats


def train_word2vec_model(sentences, model_name, vector_size=100, window=5, 
                         min_count=2, sg=0, epochs=5, workers=4):
    """
    Huan luyen mo hinh Word2Vec.
    
    Args:
        sentences: Danh sach cac cau
        model_name: Ten mo hinh (de luu file)
        vector_size: Kich thuoc vector embedding
        window: Kich thuoc cua so ngu canh
        min_count: So lan xuat hien toi thieu
        sg: 0=CBOW, 1=Skip-gram
        epochs: So epoch huan luyen
        workers: So luong thread
        
    Returns:
        Mo hinh da huan luyen, thoi gian huan luyen
    """
    print(f"\nHuan luyen mo hinh {model_name}...")
    print(f"  - Kien truc: {'Skip-gram' if sg == 1 else 'CBOW'}")
    print(f"  - Vector size: {vector_size}")
    print(f"  - Window: {window}")
    print(f"  - Min count: {min_count}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Workers: {workers}")
    
    start_time = time.time()
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
        seed=42  # De ket qua nhat quan
    )
    
    training_time = time.time() - start_time
    
    print(f"Hoan thanh trong {training_time:.2f} giay")
    print(f"Tu vung: {len(model.wv)} tu")
    
    return model, training_time


def evaluate_model(model, model_name):
    """
    Danh gia mo hinh voi cac test khac nhau.
    
    Args:
        model: Mo hinh Word2Vec
        model_name: Ten mo hinh
    """
    print(f"\n{'='*70}")
    print(f"DANH GIA MO HINH: {model_name}")
    print(f"{'='*70}")
    
    # Test 1: Tim tu tuong dong
    print("\n1. TU TUONG DONG")
    print("-" * 70)
    
    test_words = ['government', 'people', 'time', 'work', 'good']
    
    for word in test_words:
        if word in model.wv:
            print(f"\nTop 5 tu tuong dong voi '{word}':")
            similar = model.wv.most_similar(word, topn=5)
            for i, (w, score) in enumerate(similar, 1):
                print(f"  {i}. {w:20s} ({score:.4f})")
        else:
            print(f"\n'{word}' khong co trong tu vung")
    
    # Test 2: Word analogies
    print(f"\n2. WORD ANALOGIES")
    print("-" * 70)
    print("Cong thuc: A - B + C ≈ D")
    print("(Vi du: king - man + woman ≈ queen)\n")
    
    analogies = [
        ('man', 'woman', 'king', 'queen?'),
        ('good', 'better', 'bad', 'worse?'),
        ('go', 'went', 'do', 'did?'),
        ('big', 'bigger', 'small', 'smaller?'),
        ('france', 'paris', 'england', 'london?')
    ]
    
    for positive1, negative, positive2, expected in analogies:
        if all(w in model.wv for w in [positive1, negative, positive2]):
            result = model.wv.most_similar(
                positive=[positive2, negative],
                negative=[positive1],
                topn=3
            )
            print(f"{positive1} - {negative} + {positive2} ≈ {expected}")
            for i, (word, score) in enumerate(result, 1):
                marker = "✓" if word.lower() in expected.lower().replace('?', '') else " "
                print(f"  {marker} {i}. {word:15s} ({score:.4f})")
            print()
        else:
            missing = [w for w in [positive1, negative, positive2] if w not in model.wv]
            print(f"{positive1} - {negative} + {positive2}: Tu khong co trong tu vung: {missing}\n")
    
    # Test 3: Do tuong dong giua cac cap tu
    print(f"\n3. DO TUONG DONG GIUA CAC CAP TU")
    print("-" * 70)
    
    word_pairs = [
        ('man', 'woman'),
        ('king', 'queen'),
        ('good', 'bad'),
        ('big', 'small'),
        ('city', 'town'),
        ('doctor', 'hospital'),
        ('student', 'teacher'),
        ('happy', 'sad')
    ]
    
    print(f"{'Tu 1':15s} {'Tu 2':15s} {'Similarity':>12s}")
    print("-" * 45)
    
    for word1, word2 in word_pairs:
        if word1 in model.wv and word2 in model.wv:
            sim = model.wv.similarity(word1, word2)
            print(f"{word1:15s} {word2:15s} {sim:12.4f}")
        else:
            print(f"{word1:15s} {word2:15s} {'N/A':>12s}")
    
    # Test 4: Tu khong lien quan
    print(f"\n4. TIM TU KHONG LIEN QUAN")
    print("-" * 70)
    
    word_lists = [
        ['breakfast', 'lunch', 'dinner', 'car'],
        ['cat', 'dog', 'bird', 'table'],
        ['red', 'blue', 'green', 'fast'],
        ['run', 'walk', 'jump', 'apple']
    ]
    
    for words in word_lists:
        if all(w in model.wv for w in words):
            outlier = model.wv.doesnt_match(words)
            print(f"Danh sach: {words}")
            print(f"  → Tu khong lien quan: '{outlier}'")
        else:
            missing = [w for w in words if w not in model.wv]
            print(f"Danh sach: {words}")
            print(f"  → Khong the test (tu khong co: {missing})")


def compare_models(model1, model2, name1, name2):
    """
    So sanh hai mo hinh.
    
    Args:
        model1, model2: Cac mo hinh can so sanh
        name1, name2: Ten cac mo hinh
    """
    print(f"\n{'='*70}")
    print(f"SO SANH: {name1} vs {name2}")
    print(f"{'='*70}")
    
    # So sanh tu vung
    vocab1 = set(model1.wv.index_to_key)
    vocab2 = set(model2.wv.index_to_key)
    
    print(f"\nKich thuoc tu vung:")
    print(f"  {name1}: {len(vocab1):,} tu")
    print(f"  {name2}: {len(vocab2):,} tu")
    print(f"  Tu chung: {len(vocab1 & vocab2):,} tu")
    
    # So sanh ket qua tren cung mot test word
    test_words = ['government', 'people', 'time']
    
    print(f"\nSo sanh tu tuong dong:")
    for word in test_words:
        if word in vocab1 and word in vocab2:
            print(f"\nTu: '{word}'")
            
            sim1 = model1.wv.most_similar(word, topn=3)
            sim2 = model2.wv.most_similar(word, topn=3)
            
            print(f"  {name1:20s} | {name2:20s}")
            print(f"  {'-'*20} | {'-'*20}")
            
            for i in range(3):
                w1, s1 = sim1[i] if i < len(sim1) else ('', 0)
                w2, s2 = sim2[i] if i < len(sim2) else ('', 0)
                print(f"  {i+1}. {w1:15s} {s1:.3f} | {w2:15s} {s2:.3f}")


def main():
    print("=" * 70)
    print("LAB 4 BONUS: HUAN LUYEN MO HINH WORD2VEC TU DAU")
    print("=" * 70)
    print()
    
    # Thu tim file du lieu
    possible_paths = [
        os.path.join('..', 'Lab1&Lab2', 'UD_English-EWT', 'en_ewt-ud-train.txt'),
        os.path.join('..', '..', 'Lab1&Lab2', 'UD_English-EWT', 'en_ewt-ud-train.txt'),
        os.path.join('data', 'sample_text.txt'),
        'en_ewt-ud-train.txt'
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    # Neu khong tim thay file, tao du lieu mau
    if data_path is None:
        print("Khong tim thay file du lieu. Tao du lieu mau...")
        sentences = create_sample_data()
    else:
        print(f"Doc du lieu tu: {data_path}")
        
        # Thu doc voi dinh dang CoNLL-U
        sentences = load_text_data(data_path)
        
        # Neu khong doc duoc, thu dinh dang text thuong
        if not sentences:
            sentences = load_simple_text(data_path)
        
        # Neu van khong co du lieu, tao mau
        if not sentences:
            print("Khong doc duoc du lieu. Tao du lieu mau...")
            sentences = create_sample_data()
    
    print()
    
    # Phan tich corpus
    print("PHAN TICH CORPUS")
    print("-" * 70)
    stats = analyze_corpus(sentences)
    
    print(f"So cau: {stats['num_sentences']:,}")
    print(f"Tong so tu: {stats['total_words']:,}")
    print(f"Tu duy nhat: {stats['unique_words']:,}")
    print(f"Do dai trung binh cau: {stats['avg_sentence_length']:.2f} tu")
    
    print(f"\n10 tu pho bien nhat:")
    for i, (word, count) in enumerate(stats['most_common'], 1):
        print(f"  {i:2d}. {word:15s} ({count:,} lan)")
    
    # Huan luyen mo hinh CBOW
    print("\n" + "=" * 70)
    print("HUAN LUYEN MO HINH")
    print("=" * 70)
    
    model_cbow, time_cbow = train_word2vec_model(
        sentences=sentences,
        model_name="CBOW",
        vector_size=100,
        window=5,
        min_count=2,
        sg=0,  # CBOW
        epochs=10,
        workers=4
    )
    
    # Huan luyen mo hinh Skip-gram
    model_sg, time_sg = train_word2vec_model(
        sentences=sentences,
        model_name="Skip-gram",
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,  # Skip-gram
        epochs=10,
        workers=4
    )
    
    # Luu cac mo hinh
    os.makedirs('results', exist_ok=True)
    
    cbow_path = os.path.join('results', 'word2vec_cbow.model')
    sg_path = os.path.join('results', 'word2vec_skipgram.model')
    
    model_cbow.save(cbow_path)
    model_sg.save(sg_path)
    
    print(f"\nDa luu mo hinh:")
    print(f"  - CBOW: {cbow_path}")
    print(f"  - Skip-gram: {sg_path}")
    
    # Danh gia cac mo hinh
    evaluate_model(model_cbow, "CBOW")
    evaluate_model(model_sg, "Skip-gram")
    
    # So sanh hai mo hinh
    compare_models(model_cbow, model_sg, "CBOW", "Skip-gram")
    
    # Tong ket
    print(f"\n{'='*70}")
    print("TONG KET")
    print(f"{'='*70}")
    print(f"\nDu lieu:")
    print(f"  - So cau: {stats['num_sentences']:,}")
    print(f"  - Tong so tu: {stats['total_words']:,}")
    print(f"  - Tu duy nhat: {stats['unique_words']:,}")
    
    print(f"\nMo hinh CBOW:")
    print(f"  - Tu vung: {len(model_cbow.wv):,} tu")
    print(f"  - Vector size: {model_cbow.wv.vector_size}")
    print(f"  - Thoi gian huan luyen: {time_cbow:.2f}s")
    print(f"  - File: {cbow_path}")
    
    print(f"\nMo hinh Skip-gram:")
    print(f"  - Tu vung: {len(model_sg.wv):,} tu")
    print(f"  - Vector size: {model_sg.wv.vector_size}")
    print(f"  - Thoi gian huan luyen: {time_sg:.2f}s")
    print(f"  - File: {sg_path}")
    
    print(f"\n{'='*70}")
    print("HOAN THANH DEMO!")
    print(f"{'='*70}")


def create_sample_data():
    """
    Tao du lieu mau de demo neu khong co file du lieu.
    
    Returns:
        Danh sach cac cau
    """
    sample_text = """
    Natural language processing is a field of artificial intelligence.
    Machine learning algorithms can learn from data.
    Deep learning uses neural networks with multiple layers.
    Word embeddings represent words as dense vectors.
    The cat sat on the mat and looked at the dog.
    The dog played in the garden with other dogs.
    Python is a popular programming language for data science.
    Data scientists use Python for machine learning tasks.
    The king ruled the country with wisdom and power.
    The queen was beloved by all the people in the kingdom.
    Good students study hard and get better grades.
    Bad weather makes people feel sad and gloomy.
    The big elephant is bigger than the small mouse.
    Paris is the capital city of France.
    London is the capital city of England.
    Computer science involves programming and algorithms.
    Software engineers write code to solve problems.
    The doctor works at the hospital every day.
    Teachers educate students in schools and universities.
    Happy children play games in the playground.
    """
    
    sentences = []
    for line in sample_text.strip().split('\n'):
        line = line.strip()
        if line:
            words = re.findall(r'\b[a-z]+\b', line.lower())
            if words:
                sentences.append(words)
    
    return sentences


if __name__ == "__main__":
    main()
