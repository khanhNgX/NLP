"""
Lab 4 Advanced: Huan luyen Word2Vec voi Apache Spark
=======================================================
"""

import sys
import os

# Them duong dan goc du an vao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import Tokenizer, Word2Vec, StopWordsRemover
    from pyspark.sql.functions import col, lower, regexp_replace, split
    import re
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


def main():
    if not PYSPARK_AVAILABLE:
        return
    
    print("="*70)
    print("LAB 4 ADVANCED: WORD2VEC VOI APACHE SPARK")
    print("="*70)
    print()
    
    # ========================================================================
    # BUOC 1: KHOI TAO SPARK SESSION
    # ========================================================================
    print("BUOC 1: KHOI TAO SPARK SESSION")
    print("-"*70)
    
    # Spark Session la diem vao cho tat ca chuc nang Spark
    # Giai thich cac config:
    # - appName: Ten ung dung (de nhan dien trong Spark UI)
    # - master("local[*]"): Chay local, dung tat ca cores
    # - spark.driver.memory: Bo nho cho driver (process chinh)
    # - spark.sql.shuffle.partitions: So phan vung khi shuffle du lieu
    
    spark = SparkSession.builder \
        .appName("Lab4_Word2Vec_Training") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    
    # Giam log de output sach hon
    spark.sparkContext.setLogLevel("WARN")
    
    # ========================================================================
    # BUOC 2: TAI DU LIEU
    # ========================================================================
    print("BUOC 2: TAI DU LIEU")
    print("-"*70)
    
    # Tim file C4 dataset
    data_path = None
    possible_paths = [
        os.path.join('..', 'Lab1&Lab2', 'data', 'c4-train.00000-of-01024-30K.json.gz'),
        os.path.join('..', '..', 'Lab1&Lab2', 'data', 'c4-train.00000-of-01024-30K.json.gz'),
        os.path.join('data', 'c4-train.00000-of-01024-30K.json.gz'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    # Doc du lieu
    if data_path:
        print(f"Doc tu file: {data_path}")
        # C4 dataset o dinh dang JSON, moi dong la 1 object JSON
        # Chung ta quan tam toi truong 'text'
        df = spark.read.json(data_path)
        df = df.select("text").limit(1000)  # Lay 1000 dong de demo
        print(f"Da doc {df.count()} dong tu file JSON")
    else:
        print("Khong tim thay file C4")
    
    print("\nVi du du lieu:")
    df.show(5, truncate=60)
    print()
    
    # ========================================================================
    # BUOC 3: TIEN XU LY DU LIEU
    # ========================================================================
    print("BUOC 3: TIEN XU LY DU LIEU")
    print("-"*70)
    
    # 1. Chon cot text va chuyen sang chu thuong
    print("  1. Chuyen thanh chu thuong...")
    df = df.withColumn("text_lower", lower(col("text")))
    
    # 2. Loai bo dau cau va ky tu dac biet
    # Giu lai chi chu cai, so va khoang trang
    print("  2. Loai bo dau cau va ky tu dac biet...")
    df = df.withColumn(
        "text_clean",
        regexp_replace(col("text_lower"), "[^a-z0-9\\s]", " ")
    )
    
    # 3. Loai bo nhieu khoang trang lien tiep
    print("  3. Chuan hoa khoang trang...")
    df = df.withColumn(
        "text_clean",
        regexp_replace(col("text_clean"), "\\s+", " ")
    )
    
    # 4. Tach van ban thanh mang cac tu (tokenization)
    print("  4. Tach tu (tokenization)...")
    tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")
    df = tokenizer.transform(df)
    
    # 5. Loai bo stop words (cac tu pho bien: the, is, a, ...)
    print("  5. Loai bo stop words...")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df = remover.transform(df)
    
    print("Hoan thanh tien xu ly")
    
    print("\nVi du sau khi tien xu ly:")
    df.select("text", "filtered_words").show(5, truncate=40)
    print()
    
    # ========================================================================
    # BUOC 4: HUAN LUYEN MO HINH WORD2VEC
    # ========================================================================
    print("BUOC 4: HUAN LUYEN MO HINH WORD2VEC")
    # print("-"*70)
    
    # print("\nGIAI THICH CAC THAM SO:")
    # print("-"*70)
    # print("1. vectorSize = 100")
    # print("   - Kich thuoc vector embedding cho moi tu")
    # print("   - Gia tri thuong dung: 50, 100, 200, 300")
    # print("   - Lon hon = bieu dien tot hon nhung cham hon")
    # print()
    # print("2. minCount = 5")
    # print("   - Bo qua cac tu xuat hien it hon 5 lan")
    # print("   - Giup loai bo tu hiem, giam overfitting")
    # print("   - Gia tri thuong dung: 2-10")
    # print()
    # print("3. windowSize = 5")
    # print("   - So tu xung quanh duoc xem xet (trai + phai)")
    # print("   - Window = 5 nghia la xem 5 tu truoc va 5 tu sau")
    # print("   - Lon hon = bat duoc ngu canh rong hon")
    # print("   - Gia tri thuong dung: 3-10")
    # print()
    # print("4. maxIter = 10")
    # print("   - So lan duyet qua toan bo du lieu (epochs)")
    # print("   - Nhieu hon = hoc tot hon nhung cham hon")
    # print("   - Gia tri thuong dung: 5-20")
    # print()
    # print("5. inputCol = 'filtered_words'")
    # print("   - Ten cot input (mang cac tu da xu ly)")
    # print()
    # print("6. outputCol = 'result'")
    # print("   - Ten cot output (vector embedding cua tai lieu)")
    # print()
    # print("-"*70)
    
    # Cau hinh mo hinh Word2Vec
    word2vec = Word2Vec(
        vectorSize=100,      # Kich thuoc vector
        minCount=5,          # So lan xuat hien toi thieu
        windowSize=5,        # Kich thuoc cua so ngu canh
        maxIter=10,          # So epoch
        inputCol="filtered_words",
        outputCol="result"
    )
    
    # Huan luyen mo hinh
    print("\nBat dau huan luyen...")
    import time
    start_time = time.time()
    
    model = word2vec.fit(df)
    
    training_time = time.time() - start_time
    
    print(f"Hoan thanh huan luyen trong {training_time:.2f} giay")
    print()
    
    # ========================================================================
    # BUOC 5: TIM TU TUONG DONG
    # ========================================================================
    print("BUOC 5: TIM TU TUONG DONG")
    print("-"*70)
    
    # Danh sach tu de test
    test_words = ["computer", "learning", "data", "neural", "science"]
    
    for word in test_words:
        print(f"\nTop 5 tu tuong dong voi '{word}':")
        try:
            # findSynonyms tim cac tu co vector gan nhat (cosine similarity)
            synonyms = model.findSynonyms(word, 5)
            
            # Hien thi ket qua
            results = synonyms.collect()
            if results:
                for i, row in enumerate(results, 1):
                    print(f"  {i}. {row['word']:20s} (similarity: {row['similarity']:.4f})")
            else:
                print(f"  (Khong tim thay tu tuong dong)")
        except Exception as e:
            print(f"  (Tu '{word}' khong co trong tu vung)")
    
    print()
    
    # ========================================================================
    # THONG KE TONG KET
    # ========================================================================
    print("="*70)
    print("THONG KE TONG KET")
    print("="*70)
    
    # Lay thong tin mo hinh
    vocab_df = model.getVectors()
    vocab_size = vocab_df.count()
    
    print(f"\nMo hinh:")
    print(f"  - Kich thuoc tu vung: {vocab_size:,} tu")
    print(f"  - Kich thuoc vector: {model.getVectorSize()}")
    print(f"  - Thoi gian huan luyen: {training_time:.2f}s")
    
    print(f"\nDu lieu:")
    print(f"  - So tai lieu: {df.count():,}")
    
    # Dung Spark session
    print("="*70)
    print("\nDung Spark session...")
    spark.stop()
    print("Da dung Spark session")
    print()
    print("="*70)
    print("HOAN THANH!")
    print("="*70)


if __name__ == "__main__":
    main()
