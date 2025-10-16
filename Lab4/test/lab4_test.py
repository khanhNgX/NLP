"""
Lab 4: Kiem thu Word Embeddings
Test cac chuc nang cua lop WordEmbedder
"""
import sys
import os

# Them duong dan goc du an vao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.word_embedder import WordEmbedder


def main():
    print("=" * 70)
    print("LAB 4: KIEM THU WORD EMBEDDINGS VOI WORD2VEC")
    print("=" * 70)
    print()
    
    # Khoi tao WordEmbedder voi mo hinh GloVe
    print("Buoc 1: Khoi tao WordEmbedder...")
    embedder = WordEmbedder(model_name='glove-wiki-gigaword-50')
    print(f"Tu vung co kich thuoc: {embedder.get_vocabulary_size():,} tu")
    print()
    
    # Test 1: Lay vector cho tu 'king'
    print("-" * 70)
    print("Test 1: Lay vector cho tu 'king'")
    print("-" * 70)
    king_vector = embedder.get_vector('king')
    print(f"Kich thuoc vector: {len(king_vector)}")
    print(f"10 phan tu dau tien cua vector 'king':")
    print(king_vector[:10])
    print()
    
    # Test 2: Do tuong dong giua cac tu
    print("-" * 70)
    print("Test 2: Do tuong dong giua cac tu")
    print("-" * 70)
    
    # So sanh king va queen
    sim_king_queen = embedder.get_similarity('king', 'queen')
    print(f"Do tuong dong giua 'king' va 'queen': {sim_king_queen:.4f}")
    
    # So sanh king va man
    sim_king_man = embedder.get_similarity('king', 'man')
    print(f"Do tuong dong giua 'king' va 'man': {sim_king_man:.4f}")
    
    # So sanh king va computer
    sim_king_computer = embedder.get_similarity('king', 'computer')
    print(f"Do tuong dong giua 'king' va 'computer': {sim_king_computer:.4f}")
    
    print()
    print("Nhan xet: 'king' va 'queen' co do tuong dong cao hon 'king' va 'man',")
    print("cho thay mo hinh da hoc duoc quan he ngon ngu hoc.")
    print()
    
    # Test 3: Tim cac tu tuong dong nhat
    print("-" * 70)
    print("Test 3: Tim 10 tu tuong dong nhat voi 'computer'")
    print("-" * 70)
    similar_words = embedder.get_most_similar('computer', top_n=10)
    for i, (word, score) in enumerate(similar_words, 1):
        print(f"{i:2d}. {word:20s} (do tuong dong: {score:.4f})")
    print()
    
    # Test 4: Embedding cho tai lieu
    print("-" * 70)
    print("Test 4: Embedding cho tai lieu")
    print("-" * 70)
    sentence = "The queen rules the country."
    print(f"Cau: '{sentence}'")
    doc_vector = embedder.embed_document(sentence)
    print(f"Kich thuoc vector tai lieu: {len(doc_vector)}")
    print(f"10 phan tu dau tien cua vector tai lieu:")
    print(doc_vector[:10])
    print()
    
    # Test 5: Cac test them
    print("-" * 70)
    print("Test 5: Cac vi du them")
    print("-" * 70)
    
    # Vi du voi OOV (Out-of-Vocabulary)
    print("Test tu khong co trong tu vung (OOV):")
    oov_vector = embedder.get_vector('asdfghjkl123')
    print(f"Vector cho tu OOV (nen la vector 0): {oov_vector[:5]}")
    print()
    
    # Vi du voi cac cap tu khac
    print("Cac cap tu tuong dong khac:")
    word_pairs = [
        ('woman', 'man'),
        ('paris', 'france'),
        ('python', 'programming'),
        ('happy', 'sad'),
        ('good', 'bad')
    ]
    
    for word1, word2 in word_pairs:
        if embedder.word_in_vocabulary(word1) and embedder.word_in_vocabulary(word2):
            sim = embedder.get_similarity(word1, word2)
            print(f"  {word1:15s} - {word2:15s}: {sim:.4f}")
        else:
            print(f"  {word1:15s} - {word2:15s}: Mot hoac ca hai tu khong co trong tu vung")
    print()
    
    # Test 6: Embedding cho nhieu tai lieu
    print("-" * 70)
    print("Test 6: Embedding cho nhieu tai lieu")
    print("-" * 70)
    documents = [
        "The cat sits on the mat.",
        "The dog plays in the garden.",
        "Machine learning is fascinating.",
        "Natural language processing uses embeddings."
    ]
    
    doc_vectors = []
    for i, doc in enumerate(documents, 1):
        vec = embedder.embed_document(doc)
        doc_vectors.append(vec)
        print(f"Tai lieu {i}: '{doc}'")
        print(f"  Vector (5 phan tu dau): {vec[:5]}")
    print()
    
    # Tinh do tuong dong giua cac tai lieu
    print("Do tuong dong giua cac tai lieu (dung cosine similarity):")
    import numpy as np
    
    def cosine_similarity(v1, v2):
        """Tinh do tuong dong cosine giua hai vector"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)
    
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            sim = cosine_similarity(doc_vectors[i], doc_vectors[j])
            print(f"  Tai lieu {i+1} - Tai lieu {j+1}: {sim:.4f}")
    print()
    
    print("=" * 70)
    print("HOAN THANH !")
    print("=" * 70)


if __name__ == "__main__":
    main()
