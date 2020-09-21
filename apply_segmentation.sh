#!/usr/bin/env bash

for d in $1/*/ ; do
    python test_segmentation_deeplab.py -i $d
done
