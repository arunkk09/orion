#!/usr/bin/env bash
export HADOOP_HOME=/path/to/Hive/hadoop-0.23.10
export HADOOP_MAPRED_HOME=/path/to/Hive/hadoop-0.23.10
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export HIVE_HOME=/path/to/Hive/hive-0.12.0
export PATH=$HADOOP_HOME/bin:$HIVE_HOME/bin:$PATH
