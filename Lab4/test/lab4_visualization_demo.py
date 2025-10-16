"""
Lab 4 Task 5: Truc quan hoa Word Embeddings
============================================

GIAI THICH:
Truc quan hoa embedding giup chung ta:
1. Hieu duoc cau truc ngon ngu (tu nao gan nhau ve nghia)
2. Phat hien nhom tu (clustering)
3. Danh gia chat luong mo hinh
4. Tim loi trong embedding

CAC PHUONG PHAP GIAM CHIEU:
- PCA (Principal Component Analysis): Nhanh, tuyen tinh
- t-SNE (t-Distributed Stochastic Neighbor Embedding): Cham hon nhung tot hon cho visualization
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Them duong dan goc du an vao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.word_embedder import WordEmbedder


def visualize_embeddings_pca(embedder, words, title="Word Embeddings - PCA"):
    """
    Truc quan hoa word embeddings bang PCA (giam xuong 2D).
    
    PCA (Principal Component Analysis):
    - La phuong phap giam chieu tuyen tinh
    - Tim cac thanh phan chinh (principal components) chua nhieu thong tin nhat
    - Nhanh va hieu qua cho du lieu lon
    - Phu hop de xem tong quan
    
    Args:
        embedder: Doi tuong WordEmbedder
        words: Danh sach cac tu can truc quan hoa
        title: Tieu de cua bieu do
    """
    print(f"\n{'='*70}")
    print("TRUC QUAN HOA VOI PCA")
    print(f"{'='*70}\n")
    
    # Lay vectors cho cac tu
    print(f"Dang lay vectors cho {len(words)} tu...")
    vectors = []
    valid_words = []
    
    for word in words:
        vec = embedder.get_vector(word)
        # Chi lay cac tu co trong tu vung (khong phai vector 0)
        if not np.all(vec == 0):
            vectors.append(vec)
            valid_words.append(word)
    
    print(f"Co {len(valid_words)}/{len(words)} tu co trong tu vung")
    
    if len(valid_words) < 2:
        print("Khong du tu de truc quan hoa!")
        return
    
    # Chuyen thanh numpy array
    X = np.array(vectors)
    print(f"Ma tran vectors: {X.shape} (so_tu x kich_thuoc_vector)")
    
    # Ap dung PCA de giam xuong 2 chieu
    print("\nAp dung PCA de giam xuong 2D...")
    print("PCA tim 2 huong co phuong sai lon nhat de chieu du lieu len")
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Ty le phuong sai duoc giu lai
    explained_variance = pca.explained_variance_ratio_
    print(f"Thanh phan chinh 1 giu lai: {explained_variance[0]:.1%} phuong sai")
    print(f"Thanh phan chinh 2 giu lai: {explained_variance[1]:.1%} phuong sai")
    print(f"Tong phuong sai giu lai: {sum(explained_variance):.1%}")
    
    # Ve bieu do
    print("\nVe bieu do scatter plot...")
    plt.figure(figsize=(14, 10))
    
    # Ve cac diem
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=100, c='steelblue', edgecolors='navy')
    
    # Them label cho moi diem
    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            xy=(X_2d[i, 0], X_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )
    
    plt.title(f"{title}\n(Giu lai {sum(explained_variance):.1%} phuong sai)", fontsize=14, fontweight='bold')
    plt.xlabel(f"PC1 ({explained_variance[0]:.1%} variance)", fontsize=11)
    plt.ylabel(f"PC2 ({explained_variance[1]:.1%} variance)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Luu file
    os.makedirs("results", exist_ok=True)
    output_file = "results/word_embeddings_pca.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Da luu bieu do tai: {output_file}")
    
    plt.show()
    print()


def visualize_embeddings_tsne(embedder, words, title="Word Embeddings - t-SNE"):
    """
    Truc quan hoa word embeddings bang t-SNE (giam xuong 2D).
    
    t-SNE (t-Distributed Stochastic Neighbor Embedding):
    - La phuong phap giam chieu phi tuyen (non-linear)
    - Giu lai cau truc local (tu gan nhau van gan nhau sau khi giam chieu)
    - Tot hon PCA cho visualization nhung cham hon
    - Phu hop de phat hien cluster
    
    Args:
        embedder: Doi tuong WordEmbedder
        words: Danh sach cac tu can truc quan hoa
        title: Tieu de cua bieu do
    """
    print(f"\n{'='*70}")
    print("TRUC QUAN HOA VOI t-SNE")
    print(f"{'='*70}\n")
    
    # Lay vectors cho cac tu
    print(f"Dang lay vectors cho {len(words)} tu...")
    vectors = []
    valid_words = []
    
    for word in words:
        vec = embedder.get_vector(word)
        if not np.all(vec == 0):
            vectors.append(vec)
            valid_words.append(word)
    
    print(f"Co {len(valid_words)}/{len(words)} tu co trong tu vung")
    
    if len(valid_words) < 2:
        print("Khong du tu de truc quan hoa!")
        return
    
    # Chuyen thanh numpy array
    X = np.array(vectors)
    print(f"Ma tran vectors: {X.shape}")
    
    # Ap dung t-SNE de giam xuong 2 chieu
    print("\nAp dung t-SNE de giam xuong 2D...")
    print("Luu y: t-SNE mat khoang 10-30 giay tuy so luong tu...")
    print()
    print("GIAI THICH CAC THAM SO t-SNE:")
    print("-" * 70)
    print("1. n_components=2")
    print("   - Giam xuong 2 chieu (de ve bieu do 2D)")
    print()
    print("2. perplexity=30")
    print("   - Can bang giua cau truc local va global")
    print("   - Gia tri: 5-50 (thap = focus local, cao = focus global)")
    print("   - Khuyien nghi: 30 cho du lieu trung binh")
    print()
    print("3. n_iter=1000")
    print("   - So buoc toi uu hoa")
    print("   - Nhieu hon = tot hon nhung cham hon")
    print("   - Toi thieu: 250, khuyien nghi: 1000+")
    print()
    print("4. random_state=42")
    print("   - Seed de ket qua nhat quan")
    print("   - Cung seed = cung ket qua moi lan chay")
    print("-" * 70)
    print()
    
    # Tu dong dieu chinh perplexity de phu hop voi so luong mau
    perplexity_value = min(30, max(5, len(valid_words) - 1))
    print(f"Su dung perplexity = {perplexity_value} (tuy chinh theo so mau)")
    print()
    
    tsne = TSNE(
        n_components=2,      # Giam xuong 2D
        perplexity=perplexity_value,  # Can bang local/global structure
        n_iter=1000,         # So buoc toi uu
        random_state=42      # Seed de ket qua nhat quan
    )
    
    X_2d = tsne.fit_transform(X)
    print("✓ Hoan thanh t-SNE!")
    
    # Ve bieu do
    print("\nVe bieu do scatter plot...")
    plt.figure(figsize=(14, 10))
    
    # Ve cac diem
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=100, c='coral', edgecolors='darkred')
    
    # Them label cho moi diem
    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            xy=(X_2d[i, 0], X_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Luu file
    output_file = "results/word_embeddings_tsne.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Da luu bieu do tai: {output_file}")
    
    plt.show()
    print()


def visualize_word_groups(embedder, word_groups, method='tsne'):
    """
    Truc quan hoa cac nhom tu voi mau sac khac nhau.
    
    Vi du: animals (do), countries (xanh), tech terms (tim), ...
    
    Args:
        embedder: Doi tuong WordEmbedder
        word_groups: Dictionary {ten_nhom: [danh_sach_tu]}
        method: 'pca' hoac 'tsne'
    """
    print(f"\n{'='*70}")
    print(f"TRUC QUAN HOA NHOM TU VOI {method.upper()}")
    print(f"{'='*70}\n")
    
    # Lay tat ca vectors va labels
    all_vectors = []
    all_words = []
    all_labels = []
    
    print("Dang xu ly cac nhom tu:")
    for group_name, words in word_groups.items():
        print(f"  - {group_name}: ", end="")
        valid_count = 0
        for word in words:
            vec = embedder.get_vector(word)
            if not np.all(vec == 0):
                all_vectors.append(vec)
                all_words.append(word)
                all_labels.append(group_name)
                valid_count += 1
        print(f"{valid_count}/{len(words)} tu")
    
    print(f"\nTong cong: {len(all_words)} tu hop le tu {len(word_groups)} nhom")
    
    if len(all_words) < 2:
        print("Khong du tu de truc quan hoa!")
        return
    
    # Chuyen thanh numpy array
    X = np.array(all_vectors)
    
    # Giam chieu
    print(f"\nAp dung {method.upper()} de giam xuong 2D...")
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
        variance = reducer.explained_variance_ratio_
        subtitle = f"(Giu lai {sum(variance):.1%} phuong sai)"
    else:  # tsne
        print("Dang chay t-SNE (co the mat 10-30 giay)...")
        # Tu dong dieu chinh perplexity
        perplexity_value = min(30, max(5, len(all_words) // 2))
        print(f"Su dung perplexity = {perplexity_value}")
        reducer = TSNE(n_components=2, perplexity=perplexity_value, 
                      n_iter=1000, random_state=42)
        X_2d = reducer.fit_transform(X)
        subtitle = ""
    
    print("✓ Hoan thanh giam chieu!")
    
    # Tao color map cho cac nhom
    colors = plt.cm.tab10(np.linspace(0, 1, len(word_groups)))
    group_colors = dict(zip(word_groups.keys(), colors))
    
    # Ve bieu do
    print("\nVe bieu do scatter plot voi mau sac theo nhom...")
    plt.figure(figsize=(16, 12))
    
    # Ve tung nhom voi mau rieng
    for group_name in word_groups.keys():
        # Lay chi so cua cac tu thuoc nhom nay
        indices = [i for i, label in enumerate(all_labels) if label == group_name]
        if not indices:
            continue
        
        # Ve cac diem cua nhom
        group_points = X_2d[indices]
        plt.scatter(
            group_points[:, 0],
            group_points[:, 1],
            c=[group_colors[group_name]],
            label=group_name,
            alpha=0.7,
            s=150,
            edgecolors='black',
            linewidth=1.5
        )
    
    # Them label cho moi diem
    for i, word in enumerate(all_words):
        plt.annotate(
            word,
            xy=(X_2d[i, 0], X_2d[i, 1]),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    plt.title(f"Word Embeddings Visualization - {method.upper()}\n{subtitle}", 
              fontsize=16, fontweight='bold', pad=20)
    
    if method.lower() == 'pca':
        plt.xlabel(f"PC1 ({variance[0]:.1%} variance)", fontsize=12)
        plt.ylabel(f"PC2 ({variance[1]:.1%} variance)", fontsize=12)
    else:
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
    
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Luu file
    output_file = f"results/word_groups_{method}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Da luu bieu do tai: {output_file}")
    
    plt.show()
    print()


def main():
    """Ham chinh de chay demo visualization."""
    print("=" * 70)
    print("LAB 4 TASK 5: TRUC QUAN HOA WORD EMBEDDINGS")
    print("=" * 70)
    print()
    
    # Khoi tao WordEmbedder
    print("BUOC 1: KHOI TAO WORD EMBEDDER")
    print("-" * 70)
    embedder = WordEmbedder(model_name='glove-wiki-gigaword-50')
    print()
    
    # =======================================================================
    # VI DU 1: TRUC QUAN HOA CAC TU LIEN QUAN DEN TECHNOLOGY
    # =======================================================================
    print("=" * 70)
    print("VI DU 1: TRUC QUAN HOA CAC TU VE TECHNOLOGY")
    print("=" * 70)
    
    tech_words = [
        # Core tech terms
        'computer', 'software', 'hardware', 'algorithm', 'data',
        'internet', 'network', 'server', 'database', 'program',
        # AI/ML terms
        'artificial', 'intelligence', 'machine', 'learning', 'neural',
        'deep', 'model', 'training', 'prediction', 'classification',
        # Programming
        'python', 'java', 'code', 'function', 'variable',
        'debug', 'compile', 'execute', 'script', 'library'
    ]
    
    # Truc quan hoa voi PCA
    visualize_embeddings_pca(embedder, tech_words, 
                             "Technology Terms Visualization - PCA")
    
    # Truc quan hoa voi t-SNE
    visualize_embeddings_tsne(embedder, tech_words,
                              "Technology Terms Visualization - t-SNE")
    
    # =======================================================================
    # VI DU 2: TRUC QUAN HOA CAC NHOM TU KHAC NHAU
    # =======================================================================
    print("=" * 70)
    print("VI DU 2: TRUC QUAN HOA NHIEU NHOM TU (TOI MAU)")
    print("=" * 70)
    
    word_groups = {
        'Animals': [
            'dog', 'cat', 'bird', 'fish', 'horse', 'elephant',
            'lion', 'tiger', 'bear', 'wolf', 'rabbit', 'deer'
        ],
        'Countries': [
            'america', 'england', 'france', 'germany', 'japan',
            'china', 'india', 'russia', 'brazil', 'canada'
        ],
        'Colors': [
            'red', 'blue', 'green', 'yellow', 'orange',
            'purple', 'pink', 'brown', 'black', 'white'
        ],
        'Emotions': [
            'happy', 'sad', 'angry', 'fear', 'love',
            'joy', 'hate', 'surprise', 'disgust', 'trust'
        ],
        'Technology': [
            'computer', 'software', 'internet', 'algorithm', 'data',
            'network', 'program', 'machine', 'digital', 'cyber'
        ],
        'Sports': [
            'football', 'basketball', 'tennis', 'baseball', 'soccer',
            'hockey', 'golf', 'swimming', 'running', 'boxing'
        ]
    }
    
    # Truc quan hoa voi PCA
    visualize_word_groups(embedder, word_groups, method='pca')
    
    # Truc quan hoa voi t-SNE
    visualize_word_groups(embedder, word_groups, method='tsne')
    
    # =======================================================================
    # VI DU 3: WORD ANALOGY VISUALIZATION
    # =======================================================================
    print("=" * 70)
    print("VI DU 3: TRUC QUAN HOA WORD ANALOGIES")
    print("=" * 70)
    print()
    print("Bieu dien cac cap tu co quan he tuong tu:")
    print("  - king : queen = man : woman")
    print("  - paris : france = london : england")
    print("  - good : better = bad : worse")
    print()
    
    analogy_words = [
        # Gender
        'king', 'queen', 'man', 'woman', 'boy', 'girl',
        'father', 'mother', 'brother', 'sister', 'uncle', 'aunt',
        # Geography
        'paris', 'france', 'london', 'england', 'tokyo', 'japan',
        'beijing', 'china', 'berlin', 'germany', 'rome', 'italy',
        # Comparatives
        'good', 'better', 'best', 'bad', 'worse', 'worst',
        'big', 'bigger', 'small', 'smaller', 'fast', 'faster'
    ]
    
    visualize_embeddings_tsne(embedder, analogy_words,
                              "Word Analogies Visualization - t-SNE")
    
    # =======================================================================
    # TONG KET
    # =======================================================================
    print("=" * 70)
    print("TONG KET")
    print("=" * 70)
    print()
    print("Da tao cac bieu do:")
    print("  1. results/word_embeddings_pca.png")
    print("  2. results/word_embeddings_tsne.png")
    print("  3. results/word_groups_pca.png")
    print("  4. results/word_groups_tsne.png")
    print()
    print("NHAN XET:")
    print("-" * 70)
    print("1. PCA vs t-SNE:")
    print("   - PCA: Nhanh, tuyen tinh, tot cho tong quan")
    print("   - t-SNE: Cham hon, phi tuyen, tot hon cho phat hien cluster")
    print()
    print("2. Cac tu co nghia gan nhau thuong nam gan nhau tren bieu do")
    print("   - Vi du: 'king' gan 'queen', 'man' gan 'woman'")
    print()
    print("3. Cac nhom tu khac nhau thuong tach biet thanh cac cluster")
    print("   - Vi du: Animals rieng biet voi Countries")
    print()
    print("4. Word analogies duoc the hien qua khoang cach vector")
    print("   - Vi du: vector(king) - vector(man) ≈ vector(queen) - vector(woman)")
    print()
    print("=" * 70)
    print("HOAN THANH!")
    print("=" * 70)


if __name__ == "__main__":
    main()
