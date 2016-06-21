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

package org.apache.spark.h2o

import org.apache.spark.Logging
import org.apache.spark.h2o.utils.H2OSchemaUtils
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import water.DKV


object DataSourceUtils{

  def getSparkSQLSchema(key: String): StructType = {
    val frame = DKV.getGet[Frame](key)
    H2OSchemaUtils.createSchema(frame)
  }

  def overwrite(key: String, originalFrame: Frame, newDataFrame: DataFrame)(implicit h2oContext: H2OContext): Unit = {
    originalFrame.remove()
    h2oContext.asH2OFrame(newDataFrame, key)
  }
}

case class H2ORelation(
              key: String)
              (implicit @transient val sqlContext: SQLContext) extends BaseRelation with TableScan with PrunedScan with Logging {


  implicit lazy private val h2oContext = {
    if(H2OContext.get().isEmpty){
      throw new RuntimeException("H2OContext has to be started in order to save/load data using H2O Data source.")
    }else{
      H2OContext.get().get
    }
  }
  val schema = DataSourceUtils.getSparkSQLSchema(key)

  override def buildScan(): RDD[Row] = {
    h2oContext.asDataFrame(DKV.getGet[Frame](key)).rdd
  }

  override def buildScan(requiredColumns: Array[String]): RDD[Row] = {
    if(requiredColumns.length == 0){
        // if no required columns are specified, return all
        buildScan()
    }else{
      import h2oContext.implicits._
      val frame: H2OFrame = DKV.getGet[Frame](key).subframe(requiredColumns)
      h2oContext.asDataFrame(frame).rdd
  }
  }

}
