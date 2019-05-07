#!/usr/bin/python

from kafka import KafkaConsumer;


kafkaHosts=["kafka01.paas.longfor.sit:9092"
            ,"kafka02.paas.longfor.sit:9092"
            ,"kafka03.paas.longfor.sit:9092"]

'''
earliest 
当各分区下有已提交的offset时，从提交的offset开始消费；无提交的offset时，从头开始消费 
latest 
当各分区下有已提交的offset时，从提交的offset开始消费；无提交的offset时，消费新产生的该分区下的数据 
none 
topic各分区都存在已提交的offset时，从offset后开始消费；只要有一个分区不存在已提交的offset，则抛出异常
'''
consumer = KafkaConsumer(
    bootstrap_servers=kafkaHosts,group_id='mdf_group',auto_offset_reset='latest');

consumer.subscribe("testapplog_plm-prototype");

for msg in consumer:
    print(msg.value)