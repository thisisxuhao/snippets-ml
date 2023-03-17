package cn.harward.spark

import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class FPGrowthExamples extends AnyFunSuite with Serializable with BeforeAndAfterAll{
  val spark: SparkSession = SparkSession
    .builder
    .appName(s"${this.getClass.getSimpleName}")
    .master("local[*]")
    .getOrCreate()
  import spark.implicits._

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  override def afterAll(): Unit = {
    super.afterAll()
    spark.close()
  }

  test("An example for FPGrowth") {
    /**
     * <a href="https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/FPGrowthExample.scala">源自apache spark examples</a>
     * <p>输入数据: 商品集合 Array[Integer]</p>
     */
    // $example on$
    // 商品集合
    val dataset = spark.createDataset(Seq(
      "1 2 5",
      "1 2 3 5",
      "1 2")
    ).map(t => t.split("\\s"))
      .toDF("items")

    val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.5).setMinConfidence(0.6)
    val model = fpgrowth.fit(dataset)

    // Display frequent itemsets.
    model.freqItemsets.show()

    // Display generated association rules.
    model.associationRules.show()

    // transform examines the input items against all the association rules and summarize the
    // consequents as prediction
    model.transform(dataset).show()

  }


}
