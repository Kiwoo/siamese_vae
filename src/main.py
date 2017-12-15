#!/usr/bin/env python
#import numpy as np

import tensorflow as tf
import tf_util as U
import os.path as osp
import sys
from misc_util import set_global_seeds, read_dataset, get_cur_dir, header, warn

# from models_2dshapes import mymodel
# from models_celeba import mymodel
# from models_curriculum import mymodel_curr
from train import train_net, mgpu_train_net, mgpu_classifier_train_net
# from train_dsprites import train_net
# from train_curriculum import train_curr_net
# from test import test_net
import argparse
import numpy as np
from data_manager import DataManager


def main():

    # Base: https://openreview.net/pdf?id=Sy2fzU9gl

    # (1) parse arguments


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset') # chairs, celeba, dsprites
    parser.add_argument('--mode') # train, test
    parser.add_argument('--disentangled_feat', type=int)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()


    dataset = args.dataset
    mode = args.mode
    disentangled_feat = args.disentangled_feat
    chkfile_name = "chk_{}_{}".format(dataset, disentangled_feat)
    logfile_name = "log_{}_{}".format(dataset, disentangled_feat)
    validatefile_name = "val_{}_{}".format(dataset, disentangled_feat)

    # (2) Dataset

    if dataset == 'chairs':
        dir_name = "/dataset/chairs/training_img"
    elif dataset == 'celeba':
        dir_name = 'temporarily not available'
    elif dataset == 'dsprites':
        dir_name = '/dataset/dsprites' # This is dummy, for dsprites dataset, we are using data_manager
    else:
        header("Unknown dataset name")

    cur_dir = get_cur_dir()
    cur_dir = osp.join(cur_dir, 'dataset')
    cur_dir = osp.join(cur_dir, 'chairs')
    img_dir = osp.join(cur_dir, 'training_img') # This is for chairs


    # (3) Set experiment configuration, and disentangled_feat, according to beta-VAE( https://openreview.net/pdf?id=Sy2fzU9gl )

    if dataset == 'chairs':
        latent_dim = 32
        loss_weight = {'siam': 50000.0, 'kl': 30000.0}
        batch_size = 32
        max_epoch = 300
        lr = 0.0001
    elif dataset == 'celeba':
        latent_dim = 32
        loss_weight = {'siam': 1000.0, 'kl': 30000.0}
        batch_size = 512
        max_epoch = 300
        lr = 0.0001
    elif dataset == 'dsprites':
        latent_dim = 10
        loss_weight = {'siam': 1.0, 'kl': 4.0}
        batch_size = 64
        max_epoch = 300
        lr = 0.001
        feat_size = 5 # shape, rotation, size, x, y => Don't know why there are only 4 features in paper p6. Need to check more about it.
        batch_per_gpu = 15
        L = 20

    entangled_feat = latent_dim - disentangled_feat

    # (4) Open Tensorflow session, Need to find optimal configuration because we don't need to use single thread session
    # Important!!! : If we don't use single threaded session, then we need to change this!!!

    # sess = U.single_threaded_session()
    sess = U.mgpu_session()
    sess.__enter__()
    set_global_seeds(0)

    num_gpus = args.num_gpus

    # Model Setting

    # (5) Import model, merged into models.py
    # only celeba has RGB channel, other has black and white.

    if dataset == 'chairs':
        import models
        mynet = models.mymodel(name="mynet", img_shape = [64, 64, 1], latent_dim = latent_dim, disentangled_feat = disentangled_feat, mode = mode, loss_weight= loss_weight)
    elif dataset == 'celeba':
        import models
        mynet = models.mymodel(name="mynet", img_shape = [64, 64, 3], latent_dim = latent_dim, disentangled_feat = disentangled_feat, mode = mode, loss_weight= loss_weight)
    elif dataset == 'dsprites':
        import models

        img_shape = [None, 64, 64, 1]
        img1 = U.get_placeholder(name="img1", dtype=tf.float32, shape=img_shape)
        img2 = U.get_placeholder(name="img2", dtype=tf.float32, shape=img_shape)

        feat_cls = U.get_placeholder(name="feat_cls", dtype=tf.int32, shape=None)

        tf.assert_equal(tf.shape(img1)[0], tf.shape(img2)[0])
        tf.assert_equal(tf.floormod(tf.shape(img1)[0], num_gpus), 0)

        tf.assert_equal(tf.floormod(tf.shape(feat_cls)[0], num_gpus), 0)

        img1splits = tf.split(img1, num_gpus, 0)
        img2splits = tf.split(img2, num_gpus, 0)

        feat_cls_splits = tf.split(feat_cls, num_gpus, 0)

        mynets = []
        with tf.variable_scope(tf.get_variable_scope()):
            for gid in range(num_gpus):
                with tf.name_scope('gpu%d' % gid) as scope:
                    with tf.device('/gpu:%d' % gid):
                        mynet = models.mymodel(name="mynet", img1=img1splits[gid], img2=img2splits[gid],
                                               img_shape=img_shape[1:], latent_dim=latent_dim,
                                               disentangled_feat=disentangled_feat, mode=mode, loss_weight=loss_weight, feat_cls = feat_cls_splits[gid], feat_size = feat_size, L = L, batch_per_gpu = batch_per_gpu)
                        mynets.append(mynet)
                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

    else:
        header("Unknown model name")

    # (6) Train or test the model
    # Testing by adding noise on latent feature is not merged yet. Will be finished soon.

    if mode == 'train':
        mgpu_train_net(models=mynets, num_gpus = num_gpus, mode = mode, img_dir = img_dir, dataset = dataset, chkfile_name = chkfile_name, logfile_name = logfile_name, validatefile_name = validatefile_name, entangled_feat = entangled_feat, max_epoch = max_epoch, batch_size = batch_size, lr = lr)
        # train_net(model=mynets[0], mode = mode, img_dir = img_dir, dataset = dataset, chkfile_name = chkfile_name, logfile_name = logfile_name, validatefile_name = validatefile_name, entangled_feat = entangled_feat, max_epoch = max_epoch, batch_size = batch_size, lr = lr)
    elif mode == 'classifier_train':
        warn("Classifier Train")
        mgpu_classifier_train_net(models=mynets, num_gpus = num_gpus, mode = mode, img_dir = img_dir, dataset = dataset, chkfile_name = chkfile_name, logfile_name = logfile_name, validatefile_name = validatefile_name, entangled_feat = entangled_feat, max_epoch = max_epoch, batch_size = batch_size, lr = lr)



    elif mode == 'test':
        header("Need to be merged")
    else:
        header("Unknown mode name")

    # mynet = mymodel_curr(name="mynet", img_shape = [64, 64, 1], latent_dim = 32)
    # mynet = mymodel(name="mynet", img_shape = [64, 64, 1], latent_dim = latent_dim, disentangled_feat = disentangled_feat)
    # train_curr_net(model = mynet, img_dir = img_dir)
    # train_net(model = mynet, img_dir = img_dir, chkfile_name = chkfile_name, logfile_name = logfile_name, validatefile_name = validatefile_name, entangled_feat = entangled_feat)
    # train_net(model = mynet, manager = manager, chkfile_name = chkfile_name, logfile_name = logfile_name, validatefile_name = validatefile_name, entangled_feat = entangled_feat)
    # test_net(model = mynet, img_dir = img_dir)

if __name__ == '__main__':
    main()
