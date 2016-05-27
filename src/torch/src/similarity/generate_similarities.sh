#!/bin/bash

#algs="training training_no_discriminator training_binary_labels"
algs="training_no_discriminator training_binary_labels"
nums="1"

for algorithm in $algs ; do
    for d in datasets/* ; do
        dataset="$(basename $d)"

        for i in $nums ; do
            outfile="out/$algorithm""_""$dataset""_""$i"""
            infolder="metrics/$algorithm/$dataset/$i/"

            th  compute_similarities.lua datasets/$dataset/ "$infolder" \
                "$outfile".png l2  > "$outfile".out
        done
    done
done
