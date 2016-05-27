#!/bin/bash

for i in {0..9}
do
    th train.lua 2 $i gan_datasets/$i memory 5000 5000 | tee "$i".log
done
