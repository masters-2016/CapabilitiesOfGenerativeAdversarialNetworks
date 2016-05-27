#!/bin/bash

IMG_DIR=$1

echo "<html>"

echo "<head>"
echo "<title>Image Webify</title>"
echo "</head>"

echo "<body>"
for file in $(ls -v $IMG_DIR)
do
    echo "<h1>$file</h1>"
    echo "<img src='$IMG_DIR/$file' />"
done
echo "</body>"

echo "</html>"
