#!/bin/bash
rm -rf out || true
for i in $(seq 1000)
do
    mkdir out
    python gan.py | tee out/out.log
    mv out out$i
done
