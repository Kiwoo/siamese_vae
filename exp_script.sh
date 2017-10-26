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
# ./src/main.py --disentangled_feat 4 --chkfiles chk4 --logfiles log4 --validatefiles valid4
./src/main.py --disentangled_feat 5 --chkfiles chk5 --logfiles log5 --validatefiles valid5
./src/main.py --disentangled_feat 10 --chkfiles chk10 --logfiles log10 --validatefiles valid10
./src/main.py --disentangled_feat 7 --chkfiles chk7 --logfiles log7 --validatefiles valid7
./src/main.py --disentangled_feat 4 --chkfiles chk4 --logfiles log4 --validatefiles valid4
./src/main.py --disentangled_feat 6 --chkfiles chk6 --logfiles log6 --validatefiles valid6
./src/main.py --disentangled_feat 8 --chkfiles chk8 --logfiles log8 --validatefiles valid8
./src/main.py --disentangled_feat 12 --chkfiles chk12 --logfiles log12 --validatefiles valid12
./src/main.py --disentangled_feat 0 --chkfiles chk0 --logfiles log0 --validatefiles valid0