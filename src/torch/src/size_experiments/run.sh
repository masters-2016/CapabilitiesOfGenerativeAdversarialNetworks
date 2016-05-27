#!/bin/bash

STATUS_FILE=status.log
echo "" > $STATUS_FILE

th train.lua tt 2 2 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 10 | tee tt.log

## Batch size experiments
#echo "batch-2" >> $STATUS_FILE
#th train.lua batch-2 2 2 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-2.log
#echo "batch-4" >> $STATUS_FILE
#th train.lua batch-4 2 4 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-4.log
#echo "batch-8" >> $STATUS_FILE
#th train.lua batch-8 2 8 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-8.log
#echo "batch-16" >> $STATUS_FILE
#th train.lua batch-16 2 16 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-16.log
#echo "batch-32" >> $STATUS_FILE
#th train.lua batch-32 2 32 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-32.log
#echo "batch-64" >> $STATUS_FILE
#th train.lua batch-64 2 64 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-64.log
#echo "batch-128" >> $STATUS_FILE
#th train.lua batch-128 2 128 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-128.log
#echo "batch-256" >> $STATUS_FILE
#th train.lua batch-256 2 256 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-256.log
#echo "batch-512" >> $STATUS_FILE
#th train.lua batch-512 2 512 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-512.log
#echo "batch-1024" >> $STATUS_FILE
#th train.lua batch-1024 2 1024 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-1024.log
#echo "batch-2048" >> $STATUS_FILE
#th train.lua batch-2048 2 2048 ../../data/celeba_supervised_cropped 100000 memory 9999999999 9999999999 3600 | tee batch-2048.log
#
## Dataset size experiments
#echo "dataset-64" >> $STATUS_FILE
#th train.lua dataset-64 2 64 ../../data/celeba_supervised_cropped 64 memory 10000 1000 9999999999 | tee dataset-64.log
#echo "dataset-128" >> $STATUS_FILE
#th train.lua dataset-128 2 64 ../../data/celeba_supervised_cropped 128 memory 10000 1000 9999999999 | tee dataset-128.log
#echo "dataset-256" >> $STATUS_FILE
#th train.lua dataset-256 2 64 ../../data/celeba_supervised_cropped 256 memory 10000 1000 9999999999 | tee dataset-256.log
#echo "dataset-512" >> $STATUS_FILE
#th train.lua dataset-512 2 64 ../../data/celeba_supervised_cropped 512 memory 10000 1000 9999999999 | tee dataset-512.log
#echo "dataset-1024" >> $STATUS_FILE
#th train.lua dataset-1024 2 64 ../../data/celeba_supervised_cropped 1024 memory 10000 1000 9999999999 | tee dataset-1024.log
#echo "dataset-2048" >> $STATUS_FILE
#th train.lua dataset-2048 2 64 ../../data/celeba_supervised_cropped 2048 memory 10000 1000 9999999999 | tee dataset-2048.log
#echo "dataset-4096" >> $STATUS_FILE
#th train.lua dataset-4096 2 64 ../../data/celeba_supervised_cropped 4096 memory 10000 1000 9999999999 | tee dataset-4096.log
#echo "dataset-8192" >> $STATUS_FILE
#th train.lua dataset-8192 2 64 ../../data/celeba_supervised_cropped 8192 memory 10000 1000 9999999999 | tee dataset-8192.log
#echo "dataset-16384" >> $STATUS_FILE
#th train.lua dataset-16384 2 64 ../../data/celeba_supervised_cropped 16384 memory 10000 1000 9999999999 | tee dataset-16384.log
#echo "dataset-32768" >> $STATUS_FILE
#th train.lua dataset-32768 2 64 ../../data/celeba_supervised_cropped 32768 memory 10000 1000 9999999999 | tee dataset-32768.log
#echo "dataset-65536" >> $STATUS_FILE
#th train.lua dataset-65536 2 64 ../../data/celeba_supervised_cropped 65536 memory 10000 1000 9999999999 | tee dataset-65536.log
#echo "dataset-131072" >> $STATUS_FILE
#th train.lua dataset-131072 2 64 ../../data/celeba_supervised_cropped 131072 memory 10000 1000 9999999999 | tee dataset-131072.log
