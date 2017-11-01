#!/usr/bin/env python
from misc_util import set_global_seeds
import tf_util as U
import os.path as osp
import sys
from misc_util import set_global_seeds, read_dataset, get_cur_dir, header
# from models import mymodel
# from models_2dshapes import mymodel
# from models_celeba import mymodel
# from models_curriculum import mymodel_curr
# from train import train_net
# from train_dsprites import train_net
# from train_curriculum import train_curr_net
# from test import test_net
import argparse
import numpy as np
from data_manager import DataManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset') # chairs, celeba, dsprites
    parser.add_argument('--mode') # train, test

    parser.add_argument('--disentangled_feat', type=int)
    parser.add_argument('--chkfiles')
    parser.add_argument('--logfiles')
    parser.add_argument('--validatefiles')
    args = parser.parse_args()

    # Important!!! : If we don't use single threaded session, then we need to change this!!!

    dataset = args.dataset
    mode = args.mode
    disentangled_feat = args.disentangled_feat
    chkfile_name = args.chkfiles
    logfile_name = args.logfiles
    validatefile_name = args.validatefiles

    # if model == 'dsprites':       
    #     header("Loading Dataset")
    #     dir_name = "dsprites-dataset"
    #     if dir_name == "dsprites-dataset":
    #         manager = DataManager()
    #         manager.load()

    #     header("Loading Datasetn Done")

    if dataset == 'chairs':
        dir_name = '/dataset/chairs/training_img'
    elif dataset == 'celeba':
        dir_name = 'temporarily not available'
    elif dataset == 'dsprites':
        dir_name = '/dataset/dsprites'
    else:
        header("Unknown dataset name")
        break

    cur_dir = get_cur_dir()
    img_dir = osp.join(cur_dir, dir_name)


    latent_dim = 10
    entangled_feat = latent_dim - disentangled_feat

    sess = U.single_threaded_session()
    sess.__enter__()
    set_global_seeds(0)

    # Model Setting

    import models

    if dataset == 'chairs':
        mynet = models.mymodel(name="mynet", img_shape = [64, 64, 1], latent_dim = latent_dim, disentangled_feat = disentangled_feat, mode = mode)
    elif dataset == 'celeba':
        mynet = models.mymodel(name="mynet", img_shape = [64, 64, 3], latent_dim = latent_dim, disentangled_feat = disentangled_feat, mode = mode)
    elif dataset == 'dsprites':
        mynet = models.mymodel(name="mynet", img_shape = [64, 64, 1], latent_dim = latent_dim, disentangled_feat = disentangled_feat, mode = mode)
    else:
        header("Unknown model name")
        break


    # Traing or Test

    if mode == 'train':
        train_net(model = mynet, mode = mode, img_dir = img_dir, dataset = dataset, chkfile_name = chkfile_name, logfile_name = logfile_name, validatefile_name = validatefile_name, entangled_feat = entangled_feat)

    elif mode == 'test':
        header("Not yet implemented")
    else:
        header("Unknown mode name")
        break


    # mynet = mymodel_curr(name="mynet", img_shape = [64, 64, 1], latent_dim = 32)
    # mynet = mymodel(name="mynet", img_shape = [64, 64, 1], latent_dim = latent_dim, disentangled_feat = disentangled_feat)
    # train_curr_net(model = mynet, img_dir = img_dir)
    # train_net(model = mynet, img_dir = img_dir, chkfile_name = chkfile_name, logfile_name = logfile_name, validatefile_name = validatefile_name, entangled_feat = entangled_feat)
    # train_net(model = mynet, manager = manager, chkfile_name = chkfile_name, logfile_name = logfile_name, validatefile_name = validatefile_name, entangled_feat = entangled_feat)
    # test_net(model = mynet, img_dir = img_dir)

if __name__ == '__main__':
    main()

