#!/bin/bash

for i in {1..3} ; do
    for dataset in gan_generated unmodified unmodified_plus_gan_generated random_crop ; do
        python cnn_experiments/train.py 20 "data/$dataset" 100 > "$dataset""_$1"".log"
    done
done
