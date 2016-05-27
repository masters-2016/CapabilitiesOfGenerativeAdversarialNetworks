#!/bin/bash

IMG_DIR=$1

echo "<html>"

echo "<head>"
echo "<title>Image Webify</title>"
echo "</head>"

echo "<body>"
for file in $IMG_DIR/*
do
    name=$(basename $file)
    echo "<h1>$name</h1>"
    echo "<audio controls><source src='$file' type='audio/wav'></audio>"
done
echo "</body>"

echo "</html>"
