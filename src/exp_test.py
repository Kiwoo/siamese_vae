#!/usr/bin/env python
# from misc_util import set_global_seeds
# import tf_util as U
# # import benchmarks
# import os.path as osp
# import sys
# from misc_util import set_global_seeds, read_dataset, get_cur_dir, header
# # from models import mymodel
# from models_curriculum import mymodel_curr
# # from train import train_net
# from train_curriculum import train_curr_net
# from test import test_net
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int)
    parser.add_argument('--latent_dim_t', type=int)
    parser.add_argument('--chkfiles')
    args = parser.parse_args()
    print args.latent_dim
    print args.latent_dim_t
    print args.chkfiles
    a = args.latent_dim
    b = args.latent_dim_t
    c = a+b
    print c

    print "chkfiles_{}".format(args.chkfiles)

if __name__ == '__main__':
    main()
