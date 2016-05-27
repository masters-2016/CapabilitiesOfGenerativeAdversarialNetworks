#!/bin/bash

out=gan_datasets

rm -rf $out 2> /dev/null; mkdir $out

function extract() {
    labels=$1
    output_dir=$2

    mkdir $out/$output_dir

    i=0
    
    for file in $(grep $labels unmodified/labels.txt | awk '{print $1}')
    do
        i=$((++i))
        filename=`printf %06d $i`".jpg"
        convert unmodified/$file  -gravity center -background black -extent 32x32 $out/$output_dir/$filename
    done
}

extract 1,0,0,0,0,0,0,0,0,0 0
extract 0,1,0,0,0,0,0,0,0,0 1
extract 0,0,1,0,0,0,0,0,0,0 2
extract 0,0,0,1,0,0,0,0,0,0 3
extract 0,0,0,0,1,0,0,0,0,0 4
extract 0,0,0,0,0,1,0,0,0,0 5
extract 0,0,0,0,0,0,1,0,0,0 6
extract 0,0,0,0,0,0,0,1,0,0 7
extract 0,0,0,0,0,0,0,0,1,0 8
extract 0,0,0,0,0,0,0,0,0,1 9
