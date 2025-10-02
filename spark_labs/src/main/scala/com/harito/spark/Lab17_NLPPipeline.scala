package com.harito.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer, Normalizer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row}
import java.io.{File, PrintWriter}
// import com.harito.spark.Utils._

object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    // 1. Tuy chinh gioi han so luong tai lieu
    val limitDocuments = 1000 // Thay doi gia tri nay de xu ly nhieu/it tai lieu hon
    
    println(s"Bat dau NLP Pipeline voi gioi han tai lieu: $limitDocuments")
    
    val spark = SparkSession.builder
      .appName("Vi du NLP Pipeline")
      .master("local[*]")
      .getOrCreate()

    // Thiet lap log level de an log chi tiet
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._
    println("Tao Spark Session thanh cong.")
    println(s"Spark UI co san tai http://localhost:4040")
    println("Tam dung 10 giay de ban co the mo Spark UI...")
    //Thread.sleep(10000)

    // Ham tro giup de do luong hieu suat chi tiet
    def timeOperation[T](operationName: String)(operation: => T): T = {
      val startTime = System.nanoTime()
      val result = operation
      val duration = (System.nanoTime() - startTime) / 1e9d
      println(f"==> $operationName mat $duration%.2f giay")
      result
    }

    // 2. --- Doc Dataset voi Do luong Hieu suat ---
    val dataPath = "data/c4-train.00000-of-01024-30K.json.gz"
    val initialDF = timeOperation("Tai Du lieu") {
      spark.read.json(dataPath).limit(limitDocuments)
    }
    println(s"Doc thanh cong ${initialDF.count()} ban ghi.")
    initialDF.printSchema()
    println("\nMau DataFrame ban dau:")
    initialDF.show(5, truncate = false)

    // --- Dinh nghia cac Giai doan Pipeline ---

    // 3. --- Tach tu (Tokenization) ---
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']") // Su dung \\s cho regex va \" cho dau ngoac kep

    /*
    // Tokenizer thay the: Tokenizer don gian dua tren khoang trang.
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    */

    // 4. --- Loai bo Stop Words ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // 5. --- Vec-to hoa (Term Frequency) ---
    // Chuyen doi tokens thanh feature vectors bang HashingTF (cach nhanh de thuc hien count vectorization).
    // setNumFeatures dinh nghia kich thuoc cua feature vector. Day la so luong features toi da
    // (chieu) trong output vector. Moi tu duoc hash thanh mot index trong pham vi nay.
    //
    // Neu setNumFeatures nho hon kich thuoc vocabulary thuc te (so tu duy nhat),
    // se xay ra hash collisions. Dieu nay co nghia la cac tu khac nhau se map ve cung feature index.
    // Mac du dan den mat thong tin, nhung cho phep vector co kich thuoc co dinh, de quan ly
    // bat ke vocabulary lon den dau, tiet kiem bo nho va tinh toan cho datasets rat lon.
    // 20,000 la diem khoi dau pho bien cho nhieu tac vu NLP.
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(20000) // Dat kich thuoc cua feature vector

    // 6. --- Vec-to hoa (Inverse Document Frequency) ---
    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("tfidf_features")

    // 7. --- Chuan hoa Vector ---
    val normalizer = new Normalizer()
      .setInputCol(idf.getOutputCol)
      .setOutputCol("features")
      .setP(2.0) // Chuan hoa L2

    // 8. --- Lap rap Pipeline ---
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))

    // --- Do luong thoi gian cac thao tac chinh voi do luong hieu suat chi tiet ---

    println("\n=== HUAN LUYEN PIPELINE ===")
    val pipelineModel = timeOperation("Huan luyen Pipeline") {
      pipeline.fit(initialDF)
    }

    println("\n=== CHUYEN DOI DU LIEU ===")
    val transformedDF = timeOperation("Chuyen doi Du lieu") {
      val result = pipelineModel.transform(initialDF)
      result.cache() // Cache de tang hieu qua
      result.count() // Buoc danh gia
      result
    }

    // Tinh toan kich thuoc vocabulary thuc te sau tokenization va loai bo stop words
    val actualVocabSize = timeOperation("Tinh toan Kich thuoc Tu vung") {
      transformedDF
        .select(explode($"filtered_tokens").as("word"))
        .filter(length($"word") > 1) // Loc bo tokens co 1 ky tu
        .distinct()
        .count()
    }
    println(s"--> Kich thuoc tu vung thuc te sau tokenization va loai bo stop words: $actualVocabSize tu duy nhat.")

    // --- Hien thi Ket qua ---
    println("\n=== DU LIEU DA CHUYEN DOI MAU ===")
    transformedDF.select("text", "features").show(5, truncate = 50)

    // 9. --- Tim Tai lieu Tuong tu ---
    println("\n=== PHAN TICH DO TUONG TU TAI LIEU ===")
    
    // Ham tinh cosine similarity giua hai vectors
    def cosineSimilarity(vec1: Vector, vec2: Vector): Double = {
      val dot = vec1.toArray.zip(vec2.toArray).map { case (a, b) => a * b }.sum
      // val norm1 = math.sqrt(vec1.toArray.map(x => x * x).sum)
      // val norm2 = math.sqrt(vec2.toArray.map(x => x * x).sum)
      // if (norm1 == 0.0 || norm2 == 0.0) 0.0 else dot / (norm1 * norm2)
      return dot
    }
    
    val documentsWithFeatures = timeOperation("Thu thap Tai lieu va Features") {
      transformedDF.select("text", "features").collect()
    }
    
    // Chon tai lieu dau tien lam tham chieu
    val referenceDoc = documentsWithFeatures(0)
    val referenceText = referenceDoc.getAs[String]("text")
    val referenceVector = referenceDoc.getAs[Vector]("features")
    
    println(s"Tai lieu tham chieu (100 ky tu dau): ${referenceText.take(100)}...")
    
    // Tinh toan do tuong tu voi tat ca tai lieu khac
    val similarities = timeOperation("Tinh toan Do tuong tu") {
      documentsWithFeatures.zipWithIndex.map { case (row, index) =>
        val text = row.getAs[String]("text")
        val vector = row.getAs[Vector]("features")
        val similarity = if (index == 0) 1.0 else cosineSimilarity(referenceVector, vector)
        (index, text, similarity)
      }
    }
    
    val k = 5
    // Tim top k tai lieu tuong tu nhat (loai tru tai lieu tham chieu)
    val topSimilar = similarities
      .filter(_._1 != 0) // Loai tru tai lieu tham chieu
      .sortBy(-_._3) // Sap xep theo do tuong tu giam dan
      .take(k)
    
    println("\n=== TOP 5 TAI LIEU TUONG DONG NHAT ===")
    topSimilar.zipWithIndex.foreach { case ((docIndex, text, similarity), rank) =>
      println(f"${rank + 1}. Tai lieu $docIndex (Do tuong dong: $similarity%.4f)")
      println(f"   Noi dung: ${text.take(100)}...")
      println()
    }

    val n_results = 20
    val results = transformedDF.select("text", "features").take(n_results)

    // 10. --- Ghi Metrics va Ket qua vao File rieng biet ---

    // Ghi metrics vao thu muc log
    val log_path = "log/lab17_metrics.log" // Duong dan da sua
    new File(log_path).getParentFile.mkdirs() // Dam bao thu muc ton tai
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- Metrics Hieu suat Nang cao ---")
      logWriter.println(f"Gioi han tai lieu: $limitDocuments")
      logWriter.println(s"Kich thuoc tu vung thuc te (sau tien xu ly): $actualVocabSize tu duy nhat")
      logWriter.println(s"HashingTF numFeatures dat thanh: 20000")
      logWriter.println(s"Chuan hoa: Da ap dung chuan hoa L2")
      if (20000 < actualVocabSize) {
        logWriter.println(s"Luu y: numFeatures (20000) nho hon kich thuoc tu vung thuc te ($actualVocabSize). Hash collisions du kien se xay ra.")
      }
      logWriter.println("\n--- Phan tich Thoi gian ---")
      // Luu y: Thoi gian tung phan duoc in ra console qua ham timeOperation
      logWriter.println("Xem console output de biet thoi gian chi tiet cua tung thao tac")
      logWriter.println(s"File metrics duoc tao tai: ${new File(log_path).getAbsolutePath}")
      logWriter.println("\n--- Phan tich Do tuong dong Tai lieu ---")
      logWriter.println(s"Tai lieu tham chieu: ${referenceText.take(100)}...")
      logWriter.println("Top 5 tai lieu tuong dong:")
      topSimilar.zipWithIndex.foreach { case ((docIndex, text, similarity), rank) =>
        logWriter.println(f"  ${rank + 1}. Tai lieu $docIndex (Do tuong dong: $similarity%.4f) - ${text.take(80)}...")
      }
      logWriter.println("\nDe xem metrics chi tiet theo stage, truy cap Spark UI tai http://localhost:4040 trong qua trinh thuc thi.")
      println(s"\nDa ghi thanh cong metrics nang cao vao $log_path")
    } finally {
      logWriter.close()
    }

    // Ghi ket qua du lieu vao thu muc results
    val result_path = "results/lab17_pipeline_output.txt" // Duong dan da sua
    new File(result_path).getParentFile.mkdirs() // Dam bao thu muc ton tai
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println(s"--- Ket qua NLP Pipeline Nang cao ($n_results ket qua dau tien) ---")
      resultWriter.println(s"Gioi han tai lieu: $limitDocuments")
      resultWriter.println(s"Cac stage pipeline: Tokenizer -> StopWordsRemover -> HashingTF -> IDF -> Normalizer")
      resultWriter.println(s"File ket qua duoc tao tai: ${new File(result_path).getAbsolutePath}\n")
      
      resultWriter.println("="*80)
      resultWriter.println("PHAN TICH DO TUONG DONG TAI LIEU")
      resultWriter.println("="*80)
      resultWriter.println(s"Tai lieu Tham chieu: ${referenceText.take(100)}...")
      resultWriter.println("\nTop 5 Tai lieu Tuong dong Nhat:")
      topSimilar.zipWithIndex.foreach { case ((docIndex, text, similarity), rank) =>
        resultWriter.println(f"${rank + 1}. Tai lieu $docIndex (Do tuong dong: $similarity%.4f)")
        resultWriter.println(f"   ${text.take(100)}...")
        resultWriter.println()
      }
      
      resultWriter.println("\n" + "="*80)
      resultWriter.println("TAI LIEU MAU VOI TF-IDF VECTORS")
      resultWriter.println("="*80)
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        resultWriter.println("="*80)
        resultWriter.println(s"Van ban Goc: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"Vector TF-IDF Chuan hoa: ${features.toString}")
        resultWriter.println("="*80)
        resultWriter.println()
      }
      println(s"Da ghi thanh cong ket qua nang cao voi phan tich do tuong tu vao $result_path")
    } finally {
      resultWriter.close()
    }

    spark.stop()
    println("Da dung Spark Session.")
  }
}