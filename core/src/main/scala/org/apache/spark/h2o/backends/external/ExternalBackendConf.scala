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

package org.apache.spark.h2o.backends.external

import org.apache.spark.h2o.H2OConf
import org.apache.spark.h2o.backends.SharedH2OConf

/**
  * External backend configuration
  */
trait ExternalBackendConf extends SharedH2OConf {
  self: H2OConf =>

  import ExternalBackendConf._
  def flatFilePath = sparkConf.getOption(PROP_FLAT_FILE_PATH._1)
  def numOfExternalH2ONodes = sparkConf.getOption(PROP_EXTERNAL_H2O_NODES._1)
  /**
    * Sets path to flat file containing lines in a form. When H2O is started in external cluster mode it connects to
    * cluster using this flatfile and cloud name which can be set using setCloudName method on this configuration
    * node1_ip:node1_port
    * node2_ip:node2_port
    * @param flatfilePath path to flat file
    * @return H2O Configuration
    */
  def setFlatFilePath(flatfilePath: String): H2OConf = {
    sparkConf.set(PROP_FLAT_FILE_PATH._1, flatfilePath)
    self
  }

  def setNumOfExternalH2ONodes(numOfExternalH2ONodes: Int): H2OConf = {
    sparkConf.set(PROP_EXTERNAL_H2O_NODES._1, numOfExternalH2ONodes.toString)
    self
  }

  def externalConfString: String =
    s"""Sparkling Water configuration:
        |  backend cluster mode : ${backendClusterMode}
        |  cloudName            : ${cloudName.get}
        |  flatfile path        : ${flatFilePath.getOrElse("NOT SET")}
        |  clientBasePort       : ${clientBasePort}
        |  h2oClientLog         : ${h2oClientLogLevel}
        |  nthreads             : ${nthreads}""".stripMargin
}

object ExternalBackendConf {
  /** Path to flat file representing the cluster to which connect */
  val PROP_FLAT_FILE_PATH = ("spark.ext.h2o.cloud.flatfile",null.asInstanceOf[String])

  /** Number of nodes to wait for when connecting to external H2O cluster */
  val PROP_EXTERNAL_H2O_NODES = ("spark.ext.h2o.external.cluster.num.h2o.nodes", null.asInstanceOf[String])
}
