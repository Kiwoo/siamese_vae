#!/usr/bin/env python
from misc_util import set_global_seeds
import tf_util as U
# import benchmarks
import os.path as osp
import sys
from misc_util import set_global_seeds, read_dataset, get_cur_dir, header
from models import mymodel
from train import train_net
import argparse


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('mode', type=string)
    # args = parser.parse_args()
    # print args.mode
    sess = U.single_threaded_session()
    sess.__enter__()
    set_global_seeds(0)

    dir_name = "dataset/training_img"

    cur_dir = get_cur_dir()
    img_dir = osp.join(cur_dir, dir_name)
    header("Load model")
    mynet = mymodel(name="mynet", img_shape = [64, 64, 1], latent_dim = 32)
    train_net(model = mynet, img_dir = img_dir)

if __name__ == '__main__':
    main()
