"""Ví dụ: Phân tích cảm xúc dùng PySpark ML

Tập trung vào xây dựng pipeline tiền xử lý và mô hình:
Tokenization -> StopWordsRemoval -> HashingTF -> IDF -> LogisticRegression

Chạy:
    python Lab5/test/lab5_spark_sentiment_analysis.py --data_path Lab5/data/sentiments.csv

Lưu ý: cần cài pyspark trong môi trường để chạy file này.
"""
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def build_pipeline(num_features: int = 10000, reg_param: float = 0.001):
    """Tạo và trả về một Spark ML Pipeline theo cấu trúc mô tả."""
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=num_features)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=reg_param, featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])
    return pipeline


def main(data_path: str, num_features: int, reg_param: float):
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

    # Đọc dữ liệu csv (cột 'text' và 'sentiment' mong có trong file)
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Lưu số dòng ban đầu và loại bỏ những hàng thiếu giá trị sentiment
    initial_row_count = df.count()
    df = df.dropna(subset=["sentiment"])

    # Chuẩn hóa nhãn: chuyển -1/1 -> 0/1
    # Công thức: (sentiment + 1) / 2  => -1 -> 0, 1 -> 1
    df = df.withColumn("label", ((col("sentiment").cast("integer") + 1) / 2).cast("integer"))

    # Chia dữ liệu train/test
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    pipeline = build_pipeline(num_features=num_features, reg_param=reg_param)

    # Huấn luyện toàn bộ pipeline (gồm cả LogisticRegression)
    model = pipeline.fit(train_df)

    # Dự đoán trên tập test
    predictions = model.transform(test_df)

    # Đánh giá: accuracy và f1
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print(f"Số dòng ban đầu: {initial_row_count}")
    print(f"Số dòng sau khi drop null sentiment: {df.count()}")
    print(f"Accuracy trên tập test: {accuracy:.4f}")
    print(f"F1 trên tập test: {f1:.4f}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark sentiment analysis pipeline demo")
    parser.add_argument("--data_path", type=str, default="data/sentiments.csv", help="Đường dẫn tới file CSV chứa dữ liệu (cột text,sentiment)")
    parser.add_argument("--num_features", type=int, default=10000, help="Số chiều cho HashingTF")
    parser.add_argument("--reg_param", type=float, default=0.001, help="Tham số regularization cho LogisticRegression")
    args = parser.parse_args()

    main(args.data_path, args.num_features, args.reg_param)
