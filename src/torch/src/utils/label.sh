#!/bin/bash

# Makes a leftmost column of labels in an image of 64x64 pixel images
#
# Arguments:
# input image. path to the image file to decorate with labels
# out image. where to save the resulting image
# labels. path to file with a label on each line

image_in=$1
image_out=$2
labels=$3

tmp="image_out"_tmp

rm -rf $tmp; mkdir $tmp

function create_label_image() {
    label="$1"
    out="$2"

    convert \
        -size 128x64 \
        -gravity center \
        -background white \
        -fill black \
        -font Courier-Bold \
        -pointsize 12 \
        label:"$label" \
        $out
}

label_files=""
i=0
while read line
do
    i=$((++i))
    label_file="$tmp/$i.png"
    create_label_image "$line" "$label_file"
    label_files="$label_files $label_file"
done < $labels

convert $label_files -append $tmp/labels.png

convert $tmp/labels.png $image_in +append $image_out

rm -r $tmp
