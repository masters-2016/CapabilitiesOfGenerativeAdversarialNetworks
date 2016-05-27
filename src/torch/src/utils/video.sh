#!/bin/bash

input_dir=$1 # Directory of images
output_file=$2 # .mp4 file
images_per_second=$3 # int or fraction

output_dir=collected

# Collect the images in a single dir
rm -rf $output_dir || true
mkdir $output_dir

ls -v $input_dir > $output_dir.txt

i=0
while read line
do
    i=$((++i))
    filename=`printf %06d $i`".png"
    cp $input_dir/$line $output_dir/$filename
done < $output_dir.txt

rm $output_dir.txt

# Generate the video
ffmpeg -framerate $images_per_second -i $output_dir/%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p $output_file

# Clean
rm -r $output_dir
