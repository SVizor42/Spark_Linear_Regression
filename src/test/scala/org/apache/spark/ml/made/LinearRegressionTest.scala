package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta: Double = 0.01
  val weights: DenseVector[Double] = LinearRegressionTest._weights
  val bias: Double  = LinearRegressionTest._bias
  val real: DenseVector[Double] = LinearRegressionTest._y

  val dataFrame: DataFrame = LinearRegressionTest._dataFrame

  private def validateModel(model: LinearRegressionModel, data: DataFrame): Unit = {

    val predictions = data.collect().map(_.getAs[Double](1))

    predictions.length should be (10000)
    for (i <- 0 until predictions.length - 1) {
      predictions(i) should be (real(i) +- delta)
    }

  }

  private def validateEstimator(model: LinearRegressionModel) = {

    val parameters = model.weights

    parameters.size should be(weights.size)
    parameters(0) should be (weights(0) +- delta)
    parameters(1) should be (weights(1) +- delta)
    parameters(2) should be (weights(2) +- delta)
    model.bias should be (bias +- delta)

  }

  "Estimator" should "calculate parameters" in {

    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setMaxIter(100)
      .setStepSize(1.0)

    val model = estimator.fit(dataFrame)

    validateEstimator(model)

  }

  "Model" should "make predictions" in {

    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.fromBreeze(weights).toDense,
      bias = bias
    ).setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")

    validateModel(model, model.transform(dataFrame))

  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("y")
        .setPredictionCol("prediction")
        .setMaxIter(100)
        .setStepSize(1.0)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath)
      .fit(dataFrame).stages(0).asInstanceOf[LinearRegressionModel]

    validateEstimator(model)

  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("y")
        .setPredictionCol("prediction")
        .setMaxIter(100)
        .setStepSize(1.0)
    ))

    val model = pipeline.fit(dataFrame)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(dataFrame))

  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand(10000, 3)
  lazy val _weights: DenseVector[Double] = DenseVector(0.5, -0.1, 0.2)
  lazy val _bias: Double = 1.2
  lazy val _y: DenseVector[Double] = _X * _weights + _bias + DenseVector.rand(10000) * 0.0001

  lazy val _dataFrame: DataFrame = createDataFrame(_X, _y)

  def createDataFrame(X: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {

    import sqlc.implicits._

    lazy val data: DenseMatrix[Double] = DenseMatrix.horzcat(X, y.asDenseMatrix.t)

    lazy val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq.toDF("x1", "x2", "x3", "y")

    lazy val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")

    lazy val _dataFrame: DataFrame = assembler
      .transform(df)
      .select("features", "y")

    _dataFrame
  }

}