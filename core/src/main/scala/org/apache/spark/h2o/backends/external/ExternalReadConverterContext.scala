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

import org.apache.spark.h2o.converters.ReadConverterContext
import org.apache.spark.h2o.utils.NodeDesc
import water.{UDP, AutoBuffer, AutoBufferUtils, ExternalFrameHandler}
import water.AutoBufferUtils._
/**
  *
  * @param keyName key name of frame to query data from
  * @param chunkIdx chunk index
  * @param node the h2o node which has data for chunk with the chunkIdx
  */
class ExternalReadConverterContext(override val keyName: String, override val chunkIdx: Int,
                                    val node: NodeDesc) extends ExternalBackendUtils with ReadConverterContext {

  private val socketChannel = ExternalBackendUtils.getConnection(node)
  private val inputAb = createInputAutoBuffer()
  private val numOfRows: Int =  AutoBufferUtils.getInt(inputAb)
  private def createInputAutoBuffer(): AutoBuffer = {
    val ab = new AutoBuffer()
    AutoBufferUtils.putUdp(UDP.udp.external_frame, ab)
    ab.putInt(ExternalFrameHandler.DOWNLOAD_FRAME)
    ab.putStr(keyName)
    ab.putInt(chunkIdx)
    writeToChannel(ab, socketChannel)
    val inAb = AutoBufferUtils.create(socketChannel)
    AutoBufferUtils.getPort(inAb) // skip 2 bytes for port set by ab.putUdp
    inAb
  }

  override def numRows: Int = numOfRows

  /**
    * method isNA has to be called before any other method which gets data:
    * getDouble, getLong, getString, getShort, getByte, getFloat, getUTF8String, getBoolean, getTimestamp
    *
    * It should be always called like : isNA(..);get..(..); isNA(..);get..(..);....
    */
  override def isNA(columnNum: Int): Boolean = AutoBufferUtils.getInt(inputAb) == 1

  override def getDouble(columnNum: Int): Double = inputAb.get8d()

  override def getLong(columnNum: Int): Long = inputAb.get8()

  override def getString(columnNum: Int): String = inputAb.getStr()

  override def hasNext: Boolean = {
    val isNext = super.hasNext
    if(!isNext){
      // close socket channel after we get the last element
      socketChannel.close()
    }
    isNext
  }
}
