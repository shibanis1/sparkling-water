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

import java.nio.channels.SocketChannel

import org.apache.spark.h2o._
import org.apache.spark.h2o.converters.WriteConverterContext
import org.apache.spark.h2o.utils.NodeDesc
import water.AutoBufferUtils._
import water.{AutoBuffer, AutoBufferUtils, ExternalFrameHandler, UDP}

import scala.collection.mutable

class ExternalWriteConverterContext(nodeDesc: NodeDesc) extends ExternalBackendUtils with WriteConverterContext {

  val socketChannel = ExternalWriteConverterContext.getOrCreateConnection(nodeDesc)
  var rowCounter: Long = 0
  private val ab = new AutoBuffer().flipForReading() // using default constructor AutoBuffer is created with
  // private property _read set to false, in order to satisfy call clearForWriting it has to be set to true
  // which does the call of flipForReading method

  override def numOfRows(): Long = rowCounter
  /**
    * This method closes the communication after the chunks have been closed
    */
  override def closeChunks(): Unit = {
    AutoBufferUtils.clearForWriting(ab)
    ab.putInt(ExternalFrameHandler.CLOSE_NEW_CHUNK)
    writeToChannel(ab, socketChannel)

    // put connection back to the pool of free connections
    ExternalWriteConverterContext.putAvailableConnection(nodeDesc, socketChannel)
  }

  /**
    * Initialize the communication before the chunks are created
    */
  override def createChunks(keystr: String, vecTypes: Array[Byte], chunkId: Int): Unit = {
    AutoBufferUtils.clearForWriting(ab)
    AutoBufferUtils.putUdp(UDP.udp.external_frame, ab)
    ab.putInt(ExternalFrameHandler.CREATE_FRAME)
    ab.putInt(ExternalFrameHandler.CREATE_NEW_CHUNK)
    ab.putStr(keystr)
    ab.putA1(vecTypes)
    ab.putInt(chunkId)
    writeToChannel(ab, socketChannel)
  }


  override def put(columnNum: Int, n: Number) = {
    AutoBufferUtils.clearForWriting(ab)
    ab.putInt(ExternalFrameHandler.ADD_TO_FRAME)
    ab.putInt(ExternalFrameHandler.TYPE_NUM)
    ab.putInt(columnNum)
    ab.put8d(n.doubleValue())
    writeToChannel(ab, socketChannel)
  }


  override def put(columnNum: Int, n: Boolean) = {
    AutoBufferUtils.clearForWriting(ab)
    ab.putInt(ExternalFrameHandler.ADD_TO_FRAME)
    ab.putInt(ExternalFrameHandler.TYPE_NUM)
    ab.putInt(columnNum)
    ab.put8d(if (n) 1 else 0)
    writeToChannel(ab, socketChannel)
  }

  override def put(columnNum: Int, n: java.sql.Timestamp) = {
    AutoBufferUtils.clearForWriting(ab)
    ab.putInt(ExternalFrameHandler.ADD_TO_FRAME)
    ab.putInt(ExternalFrameHandler.TYPE_NUM)
    ab.putInt(columnNum)
    ab.put8d(n.getTime())
    writeToChannel(ab, socketChannel)
  }

  override def put(columnNum: Int, n: String) = {
    AutoBufferUtils.clearForWriting(ab)
    ab.putInt(ExternalFrameHandler.ADD_TO_FRAME)
    ab.putInt(ExternalFrameHandler.TYPE_STR)
    ab.putInt(columnNum)
    ab.putStr(n)
    writeToChannel(ab, socketChannel)
  }

  override def putNA(columnNum: Int) = {
    AutoBufferUtils.clearForWriting(ab)
    ab.putInt(ExternalFrameHandler.ADD_TO_FRAME)
    ab.putInt(ExternalFrameHandler.TYPE_NA)
    ab.putInt(columnNum)
    writeToChannel(ab, socketChannel)
  }

  override def increaseRowCounter(): Unit = rowCounter = rowCounter + 1
}




object ExternalWriteConverterContext extends ExternalBackendUtils{

  // since one executor can work on multiple tasks at the same time it can also happen that it needs
  // to communicate with the same node using 2 or more connections at the given time. For this we use
  // this helper which internally stores connections to one node and remember which ones are being used and which
  // ones are free. Programmer then can get connection using getAvailableConnection. This method creates a new connection
  // if all connections are currently used or reuse the existing free one. Programmer needs to put the connection back to the
  // pool of available connections using the method putAvailableConnection
  private class PerOneNodeConnection(val nodeDesc: NodeDesc) extends ExternalBackendUtils{

    // ordered list of connections where the available connections are at the start of the list and the used at the end.
    private val availableConnections = new mutable.SynchronizedQueue[SocketChannel]()
    def getAvailableConnection(): SocketChannel = this.synchronized {
      if(availableConnections.isEmpty){
        System.out.println("Creating connection for " + nodeDesc)
        getConnection(nodeDesc)
      }else{
        val socketChannel =  availableConnections.dequeue()
        if(!socketChannel.isOpen || !socketChannel.isConnected){
          System.out.println("RERECreating connection for " + nodeDesc)
          // connection closed, open a new one to replace it
          getConnection(nodeDesc)
        }else{
          System.out.println("Reusing connection for " + nodeDesc)
          socketChannel
        }
      }
    }

    def putAvailableConnection(sock: SocketChannel): Unit = {
      System.out.println("Putting back connection to " + nodeDesc)
      availableConnections += sock
    }
  }

  // this map is created in each executor so we don't have to specify executor Id
  private[this] val connectionMap = mutable.HashMap.empty[NodeDesc, PerOneNodeConnection]

  def getOrCreateConnection(nodeDesc: NodeDesc): SocketChannel = this.synchronized{
    if(!connectionMap.contains(nodeDesc)){
      connectionMap += nodeDesc -> new PerOneNodeConnection(nodeDesc)
    }
    connectionMap.get(nodeDesc).get.getAvailableConnection()
  }

  def putAvailableConnection(nodeDesc: NodeDesc, sock: SocketChannel): Unit = this.synchronized{

    if(!connectionMap.contains(nodeDesc)){
      connectionMap += nodeDesc -> new PerOneNodeConnection(nodeDesc)
    }
    connectionMap.get(nodeDesc).get.putAvailableConnection(sock)
  }

  def scheduleUpload[T](rdd: RDD[T]): (RDD[T], Map[Int, NodeDesc]) = {
    val nodes = cloudMembers
    val preparedRDD =  if (rdd.getNumPartitions < nodes.length) {
      // repartition to get same amount of partitions as is number of h2o nodes
      rdd.repartition(nodes.length)
    } else{
      // in case number of partitions is equal or higher than number of H2O nodes return original rdd
      // We are not repartitioning when number of partitions is bigger than number of h2o nodes since the data
      // in one partition could then become too big for one h2o node, we rather create multiple chunks at one h2o node
      rdd
    }

    // numPartitions is always >= than nodes.length at this step
    val partitionIdxs = 0 until preparedRDD.getNumPartitions
    val uploadPlan = partitionIdxs.zip(Stream.continually(nodes).flatten).toMap

    (preparedRDD, uploadPlan)
  }
}
