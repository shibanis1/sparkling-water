/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.ml.spark.models.svm

import breeze.stats.distributions.Binomial
import hex.ModelMetricsSupervised.MetricBuilderSupervised
import hex.{Model, ModelCategory, ModelMetricsBinomial, ModelMetricsRegression}
import org.dmg.pmml.Regression
import water.codegen.CodeGeneratorPipeline
import water.fvec.Frame
import water.util.{JCodeGen, SBPrintStream}
import water.{H2O, Key, Keyed}

object SVMModel {

  class SVMParameters extends Model.Parameters {
    def algoName: String = "SVM"

    def fullName: String = "Support Vector Machine(*)"

    def javaName: String = classOf[SVMModel].getName

    def progressUnits: Long = _max_iterations

    final def initialWeights(): Frame =
      if(null == _initial_weights)  null
      else _initial_weights.get()

    var _max_iterations: Int = 1000
    var _add_intercept: Boolean = false
    var _step_size: Double = 1.0
    var _reg_param: Double = 0.01
    var _convergence_tol: Double = 0.001
    var _mini_batch_fraction: Double = 1.0
    var _threshold: Double = 0.0
    var _updater: Updater = Updater.L2
    var _gradient: Gradient = Gradient.Hinge
    var _initial_weights: Key[Frame] = null
  }

  class SVMOutput(val b: SVM) extends Model.Output(b) {
    var interceptor: Double = .0
    var weights: Array[Double] = null
  }

}

class SVMModel private[svm](val selfKey: Key[_ <: Keyed[_ <: Keyed[_ <: AnyRef]]],
                              val parms: SVMModel.SVMParameters,
                              val output: SVMModel.SVMOutput)
  extends Model[SVMModel, SVMModel.SVMParameters, SVMModel.SVMOutput](selfKey, parms, output) {

  override def makeMetricBuilder(domain: Array[String]): MetricBuilderSupervised[Nothing] =
    _output.getModelCategory match {
      case ModelCategory.Binomial =>
        new ModelMetricsBinomial.MetricBuilderBinomial(domain)
      case ModelCategory.Regression =>
        new ModelMetricsRegression.MetricBuilderRegression
      case _ =>
        throw H2O.unimpl
    }

  protected def score0(data: Array[Double], preds: Array[Double]): Array[Double] = {
    java.util.Arrays.fill(preds, 0)
    preds(0) = _output.interceptor
    val threshold: Double = _parms._threshold
    preds(0) += data.zip(_output.weights).foldRight(0.0){ case ((d, w), acc) => d * w + acc}
    preds(0) =
      if (threshold.isNaN) preds(0)
      else if (preds(0) > threshold) 1
      else 0
    preds
  }

  override protected def toJavaInit(sb: SBPrintStream, fileCtx: CodeGeneratorPipeline): SBPrintStream = {
    val sbInitialized = super.toJavaInit(sb, fileCtx)
    sbInitialized.ip("public boolean isSupervised() { return " + isSupervised + "; }").nl
    JCodeGen.toStaticVar(sbInitialized, "WEIGHTS", _output.weights, "Weights.")
    sbInitialized
  }

  override protected def toJavaPredictBody(bodySb: SBPrintStream,
                                           classCtx: CodeGeneratorPipeline,
                                           fileCtx: CodeGeneratorPipeline,
                                           verboseCode: Boolean) {
    bodySb.i.p("java.util.Arrays.fill(preds,0);").nl
    bodySb.i.p(s"preds[0] = ${_output.interceptor};").nl
    bodySb.i.p(s"final double threshold = ${_parms._threshold};").nl
    bodySb.i.p("for(int i = 0; i < data.length; i++) {").nl
    bodySb.i(1).p("preds[0] += (data[i] * WEIGHTS[i]);").nl
    bodySb.i.p("}").nl
    bodySb.i.p("preds[0] = Double.isNaN(threshold) ? preds[0] : (preds[0] > threshold ? 1 : 0);").nl
  }

}
