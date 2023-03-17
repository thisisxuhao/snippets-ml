package cn.harward.xgboost4j

import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import java.io.File
import scala.collection.mutable

class XGBoost4jExamples extends AnyFunSuite with BeforeAndAfterAll {
  test("测试") {
    println("good")
  }


  test("GLM模型") {
    val trainMat = new DMatrix("../data-examples/agaricus.txt.train")
    val testMat = new DMatrix("../data-examples/agaricus.txt.test")

    // specify parameters
    // change booster to gblinear, so that we are fitting a linear model
    // alpha is the L1 regularizer
    // lambda is the L2 regularizer
    // you can also set lambda_bias which is L2 regularizer on the bias term
    val params = new mutable.HashMap[String, Any]()
    params += "alpha" -> 0.0001
    params += "boosterh" -> "gblinear"
    params += "silent" -> 1
    params += "objective" -> "binary:logistic"

    // normally, you do not need to set eta (step_size)
    // XGBoost uses a parallel coordinate descent algorithm (shotgun),
    // there could be affection on convergence with parallelization on certain cases
    // setting eta to be smaller value, e.g 0.5 can make the optimization more stable
    // param.put("eta", "0.5");
    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> trainMat
    watches += "test" -> testMat

    val booster = XGBoost.train(trainMat, params.toMap, 1, watches.toMap)
    val predicts = booster.predict(testMat)
    predicts.foreach {
      res =>
        println(res.mkString("[", ",", "]"))
    }

  }



}
