#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

# parser.add_argument('--dataset') # chairs, celeba, dsprites
# parser.add_argument('--mode') # train, test

# parser.add_argument('--disentangled_feat', type=int)
# parser.add_argument('--chkfiles')
# parser.add_argument('--logfiles')
# parser.add_argument('--validatefiles')

set -x
set -e

export PYTHONUNBUFFERED="True"

# chairs experiment
# ./src/main.py --dataset chairs --mode train --disentangled_feat 1
# ./src/main.py --dataset chairs --mode train --disentangled_feat 2
# ./src/main.py --dataset chairs --mode train --disentangled_feat 3

# dsprites
./src/main.py --dataset dsprites --mode train --disentangled_feat 1
# ./src/main.py --dataset dsprites --mode train --disentangled_feat 2