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
    parser.add_argument('--model') # chairs, celeba, dsprites
    parser.add_argument('--mode') # train, test

    parser.add_argument('--disentangled_feat', type=int)
    parser.add_argument('--chkfiles')
    parser.add_argument('--logfiles')
    parser.add_argument('--validatefiles')
    args = parser.parse_args()

    # Important!!! : If we don't use single threaded session, then we need to change this!!!

    model = args.model
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

    if model == 'chairs':
        dir_name = ''
    elif model == 'celeba':
        dir_name = ''
    elif model == 'dsprites':
        dir_name = ''
    else:
        header("Unknown model name")
        break

    cur_dir = get_cur_dir()
    img_dir = osp.join(cur_dir, dir_name)

    header("Load model")

    latent_dim = 10
    entangled_feat = latent_dim - disentangled_feat

    sess = U.single_threaded_session()
    sess.__enter__()
    set_global_seeds(0)

    if model == 'chairs':
        import models
        from train import train_net
        mynet = models.mymodel(name="mynet", img_shape = [64, 64, 1], latent_dim = latent_dim, disentangled_feat = disentangled_feat, mode = mode, batch_size = 512)
    elif model == 'celeb':
        import models_celeba
        from train import train_net
        mynet = models_celeba.mymodel(name="mynet", img_shape = [64, 64, 3], latent_dim = latent_dim, disentangled_feat = disentangled_feat, mode = mode, batch_size = 512)
    elif model == 'dsprites':
        import models_2dshapes
        from train_dsprites import train_net
        mynet = models_2dshapes.mymodel(name="mynet", img_shape = [64, 64, 1], latent_dim = latent_dim, disentangled_feat = disentangled_feat, mode = mode, batch_size = 512)
    else:
        header("Unknown model name")
        break

    if mode == 'train':
        train_net(model = mynet, img_dir = img_dir, chkfile_name = chkfile_name, logfile_name = logfile_name, validatefile_name = validatefile_name, entangled_feat = entangled_feat)

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

