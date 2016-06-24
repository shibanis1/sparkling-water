package org.apache.spark.h2o.backends.external

import java.nio.channels.SocketChannel

import org.apache.spark.h2o.utils.NodeDesc

import scala.collection.mutable

/**
  * Helper containing connections to H2O
  */
object ConnectionToH2OHelper {

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

}
