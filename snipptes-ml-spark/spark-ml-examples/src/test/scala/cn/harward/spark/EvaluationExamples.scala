package cn.harward.spark

import org.apache.spark.SparkContext
import org.apache.spark.ml.evaluation.MultilabelClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, length}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class EvaluationExamples extends AnyFunSuite with BeforeAndAfterAll {
  val spark: SparkSession = SparkSession
    .builder
    .appName(s"${this.getClass.getSimpleName}")
    .master("local[*]")
    .getOrCreate()
  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("WARN")

  import spark.implicits._


  test("多标签评估:MultilabelClassificationEvaluator") {
    val scoreAndLabels = Seq(
      (Array(0.0, 1.0), Array(0.0, 2.0)),
      (Array(0.0, 2.0), Array(0.0, 1.0)),
      (Array.empty[Double], Array(0.0)),
      (Array(2.0), Array(2.0)),
      (Array(2.0, 0.0), Array(2.0, 0.0)),
      (Array(0.0, 1.0, 2.0), Array(0.0, 1.0)),
      (Array(1.0), Array(1.0, 2.0))
    ).toDF("prediction", "label")

    import org.apache.spark.sql.functions.count
    scoreAndLabels.withColumn("size", col("prediction").isNotNull)
      .show()

    val evaluator = new MultilabelClassificationEvaluator()
      .setMetricName("precision")
    println(evaluator.evaluate(scoreAndLabels))

    evaluator.setMetricName("recallByLabel")
      .setMetricLabel(0.0)
    println(evaluator.evaluate(scoreAndLabels))

    evaluator.setMetricName("microPrecision")
    println(evaluator.evaluate(scoreAndLabels))

    evaluator.setMetricName("microRecall")
    println(evaluator.evaluate(scoreAndLabels))

  }

}
