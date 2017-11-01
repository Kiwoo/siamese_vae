#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"
./src/main.py --disentangled_feat 1 --chkfiles chk1 --logfiles log1 --validatefiles valid1
./src/main.py --disentangled_feat 2 --chkfiles chk2 --logfiles log2 --validatefiles valid2
./src/main.py --disentangled_feat 3 --chkfiles chk3 --logfiles log3 --validatefiles valid3
