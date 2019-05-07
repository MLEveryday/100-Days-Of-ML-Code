#!/usr/bin/python

from kafka import KafkaProducer

kafkaHosts=["kafka01.paas.longfor.sit:9092"
            ,"kafka02.paas.longfor.sit:9092"
            ,"kafka03.paas.longfor.sit:9092"]

producer = KafkaProducer(bootstrap_servers=kafkaHosts);

for _ in range(20):
    producer.send("testapplog_plm-prototype",b"Hello....")
producer.flush();