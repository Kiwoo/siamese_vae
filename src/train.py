import tf_util as U
import tensorflow as tf
import os
import sys
from misc_util import set_global_seeds, read_dataset, warn, mkdir_p, failure, header, get_cur_dir, load_image, Img_Saver, load_single_img, BW_Img_Saver
import argparse
import matplotlib.pyplot as plt
# from skimage.io import imsave
import h5py
import pandas as pd
from PIL import Image
import numpy as np
import random
from data_manager import DataManager
import os.path as osp
import time

slim = tf.contrib.slim
from tensorflow.python.ops import control_flow_ops


def mgpu_train_net(models, mode, img_dir, dataset, chkfile_name, logfile_name, validatefile_name, entangled_feat, max_epoch = 300, check_every_n = 500, loss_check_n = 10, save_model_freq = 5, batch_size = 512, lr = 0.001):
    img1 = U.get_placeholder_cached(name="img1")
    img2 = U.get_placeholder_cached(name="img2")

    # batch size must be multiples of ntowers (# of GPUs)
    ntowers = len(models)
    tf.assert_equal(tf.shape(img1)[0], tf.shape(img2)[0])
    tf.assert_equal(tf.floormod(tf.shape(img1)[0], ntowers), 0)

    img1splits = tf.split(img1, ntowers, 0)
    img2splits = tf.split(img2, ntowers, 0)

    tower_vae_loss = []
    tower_latent_z1_tp = []
    tower_latent_z2_tp = []
    tower_losses = []
    tower_siam_max = []
    tower_reconst1 = []
    tower_reconst2 = []
    for gid, model in enumerate(models):
        with tf.name_scope('gpu%d' % gid) as scope:
            with tf.device('/gpu:%d' % gid):

                vae_loss = U.mean(model.vaeloss)
                latent_z1_tp = model.latent_z1
                latent_z2_tp = model.latent_z2
                losses = [U.mean(model.vaeloss),
                          U.mean(model.siam_loss),
                          U.mean(model.kl_loss1),
                          U.mean(model.kl_loss2),
                          U.mean(model.reconst_error1),
                          U.mean(model.reconst_error2),
                          ]
                siam_max = U.mean(model.max_siam_loss)

                tower_vae_loss.append(vae_loss)
                tower_latent_z1_tp.append(latent_z1_tp)
                tower_latent_z2_tp.append(latent_z2_tp)
                tower_losses.append(losses)
                tower_siam_max.append(siam_max)
                tower_reconst1.append(model.reconst1)
                tower_reconst2.append(model.reconst2)

                tf.summary.scalar('Total Loss', losses[0])
                tf.summary.scalar('Siam Loss', losses[1])
                tf.summary.scalar('kl1_loss', losses[2])
                tf.summary.scalar('kl2_loss', losses[3])
                tf.summary.scalar('reconst_err1', losses[4])
                tf.summary.scalar('reconst_err2', losses[5])
                tf.summary.scalar('Siam Max', siam_max)

    vae_loss = U.mean(tower_vae_loss)
    siam_max = U.mean(tower_siam_max)
    latent_z1_tp = tf.concat(tower_latent_z1_tp, 0)
    latent_z2_tp = tf.concat(tower_latent_z2_tp, 0)
    model_reconst1 = tf.concat(tower_reconst1, 0)
    model_reconst2 = tf.concat(tower_reconst2, 0)

    losses = [[] for _ in range(len(losses))]
    for tl in tower_losses:
        for i, l in enumerate(tl):
            losses[i].append(l)

    losses = [U.mean(l) for l in losses]
    siam_normal = losses[1] / entangled_feat

    tf.summary.scalar('total/Total Loss', losses[0])
    tf.summary.scalar('total/Siam Loss', losses[1])
    tf.summary.scalar('total/kl1_loss', losses[2])
    tf.summary.scalar('total/kl2_loss', losses[3])
    tf.summary.scalar('total/reconst_err1', losses[4])
    tf.summary.scalar('total/reconst_err2', losses[5])
    tf.summary.scalar('total/Siam Normal', siam_normal)
    tf.summary.scalar('total/Siam Max', siam_max)

    compute_losses = U.function([img1, img2], vae_loss)

    all_var_list = model.get_trainable_variables()
    img1_var_list = all_var_list

    # with tf.device('/cpu:0'):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon = 0.01/batch_size)
    optimize_expr1 = optimizer.minimize(vae_loss, var_list=img1_var_list)

    merged = tf.summary.merge_all()
    train = U.function([img1, img2],
                        [losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], latent_z1_tp, latent_z2_tp, merged], updates = [optimize_expr1])

    get_reconst_img = U.function([img1, img2], [model_reconst1, model_reconst2, latent_z1_tp, latent_z2_tp])
    get_latent_var = U.function([img1, img2], [latent_z1_tp, latent_z2_tp])

    cur_dir = get_cur_dir()
    chk_save_dir = os.path.join(cur_dir, chkfile_name)
    log_save_dir = os.path.join(cur_dir, logfile_name)
    validate_img_saver_dir = os.path.join(cur_dir, validatefile_name)
    if dataset == 'chairs' or dataset == 'celeba':
        test_img_saver_dir = os.path.join(cur_dir, "test_images")
        testing_img_dir = os.path.join(cur_dir, "dataset/{}/test_img".format(dataset))

    train_writer = U.summary_writer(dir = log_save_dir)

    U.initialize()

    saver, chk_file_epoch_num = U.load_checkpoints(load_requested = True, checkpoint_dir = chk_save_dir)
    if dataset == 'chairs' or dataset == 'celeba':
        validate_img_saver = Img_Saver(Img_dir = validate_img_saver_dir)
    elif dataset == 'dsprites':
        validate_img_saver = BW_Img_Saver(Img_dir = validate_img_saver_dir) # Black and White, temporary usage
    else:
        warn("Unknown dataset Error")
        # break

    warn(img_dir)
    if dataset == 'chairs' or dataset == 'celeba':
        training_images_list = read_dataset(img_dir)
        n_total_train_data = len(training_images_list)
        testing_images_list = read_dataset(testing_img_dir)
        n_total_testing_data = len(testing_images_list)
    elif dataset == 'dsprites':
        cur_dir = osp.join(cur_dir, 'dataset')
        cur_dir = osp.join(cur_dir, 'dsprites')
        img_dir = osp.join(cur_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        manager = DataManager(img_dir, batch_size)
    else:
        warn("Unknown dataset Error")
        # break

    meta_saved = False

    if mode == 'train':
        for epoch_idx in range(chk_file_epoch_num+1, max_epoch):
            t_epoch_start = time.time()
            num_batch = manager.get_len()

            for batch_idx in range(num_batch):
                if dataset == 'chairs' or dataset == 'celeba':
                    idx = random.sample(range(n_total_train_data), 2*batch_size)
                    batch_files = [training_images_list[i] for i in idx]
                    [images1, images2] = load_image(dir_name = img_dir, img_names = batch_files)
                elif dataset == 'dsprites':
                    [images1, images2] = manager.get_next()
                img1, img2 = images1, images2
                [l1, l2, _, _] = get_reconst_img(img1, img2)

                [loss0, loss1, loss2, loss3, loss4, loss5, latent1, latent2, summary] = train(img1, img2)

                if batch_idx % 50 == 1:
                    header("******* epoch: {}/{} batch: {}/{} *******".format(epoch_idx, max_epoch, batch_idx, num_batch))
                    warn("Total Loss: {}".format(loss0))
                    warn("Siam loss: {}".format(loss1))
                    warn("kl1_loss: {}".format(loss2))
                    warn("kl2_loss: {}".format(loss3))
                    warn("reconst_err1: {}".format(loss4))
                    warn("reconst_err2: {}".format(loss5))

                if batch_idx % check_every_n == 1:
                    if dataset == 'chairs' or dataset == 'celeba':
                        idx = random.sample(range(len(training_images_list)), 2*5)
                        validate_batch_files = [training_images_list[i] for i in idx]
                        [images1, images2] = load_image(dir_name = img_dir, img_names = validate_batch_files)
                    elif dataset == 'dsprites':
                        [images1, images2] = manager.get_next()

                    [reconst1, reconst2, _, _] = get_reconst_img(images1, images2)

                    if dataset == 'chairs':
                        for img_idx in range(len(images1)):
                            sub_dir = "iter_{}".format(batch_idx)

                            save_img = np.squeeze(images1[img_idx])
                            save_img = Image.fromarray(save_img)
                            img_file_name = "{}_ori.png".format(validate_batch_files[img_idx].split('.')[0])
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

                            save_img = np.squeeze(reconst1[img_idx])
                            save_img = Image.fromarray(save_img)
                            img_file_name = "{}_rec.png".format(validate_batch_files[img_idx].split('.')[0])
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)
                    elif dataset == 'celeba':
                        for img_idx in range(len(images1)):
                            sub_dir = "iter_{}".format(batch_idx)

                            save_img = np.squeeze(images1[img_idx])
                            save_img = Image.fromarray(save_img, 'RGB')
                            img_file_name = "{}_ori.png".format(validate_batch_files[img_idx].split('.')[0])
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

                            save_img = np.squeeze(reconst1[img_idx])
                            save_img = Image.fromarray(save_img, 'RGB')
                            img_file_name = "{}_rec.png".format(validate_batch_files[img_idx].split('.')[0])
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)
                    elif dataset == 'dsprites':
                        for img_idx in range(len(images1)):
                            sub_dir = "iter_{}".format(batch_idx)

                            # save_img = images1[img_idx].reshape(64, 64)
                            save_img = np.squeeze(images1[img_idx])
                            save_img = save_img.astype(np.float32)
                            img_file_name = "{}_ori.jpg".format(img_idx)
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

                            # save_img = reconst1[img_idx].reshape(64, 64)
                            save_img = np.squeeze(reconst1[img_idx])
                            save_img = save_img.astype(np.float32)
                            img_file_name = "{}_rec.jpg".format(img_idx)
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

                if batch_idx % loss_check_n == 1:
                    train_writer.add_summary(summary, batch_idx)

            t_epoch_end = time.time()
            t_epoch_run = t_epoch_end - t_epoch_start
            if dataset == 'dsprites':
                t_check = manager.sample_size / t_epoch_run

                warn("==========================================")
                warn("Run {} th epoch in {} sec: {} images / sec".format(epoch_idx+1, t_epoch_run, t_check))
                warn("==========================================")

            # if epoch_idx % save_model_freq == 0:
            if meta_saved == True:
                saver.save(U.get_session(), chk_save_dir + '/' + 'checkpoint', global_step = epoch_idx, write_meta_graph = False)
            else:
                print "Save  meta graph"
                saver.save(U.get_session(), chk_save_dir + '/' + 'checkpoint', global_step = epoch_idx, write_meta_graph = True)
                meta_saved = True



def train_net(model, mode, img_dir, dataset, chkfile_name, logfile_name, validatefile_name, entangled_feat, max_epoch = 300, check_every_n = 500, loss_check_n = 10, save_model_freq = 5, batch_size = 512, lr = 0.001):
    img1 = U.get_placeholder_cached(name="img1")
    img2 = U.get_placeholder_cached(name="img2")

    vae_loss = U.mean(model.vaeloss)

    latent_z1_tp = model.latent_z1
    latent_z2_tp = model.latent_z2

    losses = [U.mean(model.vaeloss),
            U.mean(model.siam_loss),
            U.mean(model.kl_loss1),
            U.mean(model.kl_loss2),
            U.mean(model.reconst_error1),
            U.mean(model.reconst_error2),
            ]

    siam_normal = losses[1]/entangled_feat
    siam_max = U.mean(model.max_siam_loss)

    tf.summary.scalar('Total Loss', losses[0])
    tf.summary.scalar('Siam Loss', losses[1])
    tf.summary.scalar('kl1_loss', losses[2])
    tf.summary.scalar('kl2_loss', losses[3])
    tf.summary.scalar('reconst_err1', losses[4])
    tf.summary.scalar('reconst_err2', losses[5])
    tf.summary.scalar('Siam Normal', siam_normal)
    tf.summary.scalar('Siam Max', siam_max)



    compute_losses = U.function([img1, img2], vae_loss)
    optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon = 0.01/batch_size)

    all_var_list = model.get_trainable_variables()


    img1_var_list = all_var_list
    optimize_expr1 = optimizer.minimize(vae_loss, var_list=img1_var_list)
    merged = tf.summary.merge_all()
    train = U.function([img1, img2],
                        [losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], latent_z1_tp, latent_z2_tp, merged], updates = [optimize_expr1])
    get_reconst_img = U.function([img1, img2], [model.reconst1, model.reconst2, latent_z1_tp, latent_z2_tp])
    get_latent_var = U.function([img1, img2], [latent_z1_tp, latent_z2_tp])

    cur_dir = get_cur_dir()
    chk_save_dir = os.path.join(cur_dir, chkfile_name)
    log_save_dir = os.path.join(cur_dir, logfile_name)
    validate_img_saver_dir = os.path.join(cur_dir, validatefile_name)
    if dataset == 'chairs' or dataset == 'celeba':
        test_img_saver_dir = os.path.join(cur_dir, "test_images")
        testing_img_dir = os.path.join(cur_dir, "dataset/{}/test_img".format(dataset))

    train_writer = U.summary_writer(dir = log_save_dir)

    U.initialize()

    saver, chk_file_epoch_num = U.load_checkpoints(load_requested = True, checkpoint_dir = chk_save_dir)
    if dataset == 'chairs' or dataset == 'celeba':
        validate_img_saver = Img_Saver(Img_dir = validate_img_saver_dir)
    elif dataset == 'dsprites':
        validate_img_saver = BW_Img_Saver(Img_dir = validate_img_saver_dir) # Black and White, temporary usage
    else:
        warn("Unknown dataset Error")
        # break

    warn(img_dir)
    if dataset == 'chairs' or dataset == 'celeba':
        training_images_list = read_dataset(img_dir)
        n_total_train_data = len(training_images_list)
        testing_images_list = read_dataset(testing_img_dir)
        n_total_testing_data = len(testing_images_list)
    elif dataset == 'dsprites':
        cur_dir = osp.join(cur_dir, 'dataset')
        cur_dir = osp.join(cur_dir, 'dsprites')
        img_dir = osp.join(cur_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        manager = DataManager(img_dir, batch_size)
    else:
        warn("Unknown dataset Error")
        # break

    meta_saved = False

    if mode == 'train':
        for epoch_idx in range(chk_file_epoch_num+1, max_epoch):
            t_epoch_start = time.time()
            num_batch = manager.get_len()

            for batch_idx in range(num_batch):
                if dataset == 'chairs' or dataset == 'celeba':
                    idx = random.sample(range(n_total_train_data), 2*batch_size)
                    batch_files = [training_images_list[i] for i in idx]
                    [images1, images2] = load_image(dir_name = img_dir, img_names = batch_files)
                elif dataset == 'dsprites':
                    [images1, images2] = manager.get_next()
                img1, img2 = images1, images2
                [l1, l2, _, _] = get_reconst_img(img1, img2)

                [loss0, loss1, loss2, loss3, loss4, loss5, latent1, latent2, summary] = train(img1, img2)

                if batch_idx % 50 == 1:
                    header("******* epoch: {}/{} batch: {}/{} *******".format(epoch_idx, max_epoch, batch_idx, num_batch))
                    warn("Total Loss: {}".format(loss0))
                    warn("Siam loss: {}".format(loss1))
                    warn("kl1_loss: {}".format(loss2))
                    warn("kl2_loss: {}".format(loss3))
                    warn("reconst_err1: {}".format(loss4))
                    warn("reconst_err2: {}".format(loss5))

                if batch_idx % check_every_n == 1:
                    if dataset == 'chairs' or dataset == 'celeba':
                        idx = random.sample(range(len(training_images_list)), 2*5)
                        validate_batch_files = [training_images_list[i] for i in idx]
                        [images1, images2] = load_image(dir_name = img_dir, img_names = validate_batch_files)
                    elif dataset == 'dsprites':
                        [images1, images2] = manager.get_next()

                    [reconst1, reconst2, _, _] = get_reconst_img(images1, images2)

                    if dataset == 'chairs':
                        for img_idx in range(len(images1)):
                            sub_dir = "iter_{}".format(batch_idx)

                            save_img = np.squeeze(images1[img_idx])
                            save_img = Image.fromarray(save_img)
                            img_file_name = "{}_ori.png".format(validate_batch_files[img_idx].split('.')[0])
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

                            save_img = np.squeeze(reconst1[img_idx])
                            save_img = Image.fromarray(save_img)
                            img_file_name = "{}_rec.png".format(validate_batch_files[img_idx].split('.')[0])
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)
                    elif dataset == 'celeba':
                        for img_idx in range(len(images1)):
                            sub_dir = "iter_{}".format(batch_idx)

                            save_img = np.squeeze(images1[img_idx])
                            save_img = Image.fromarray(save_img, 'RGB')
                            img_file_name = "{}_ori.png".format(validate_batch_files[img_idx].split('.')[0])
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

                            save_img = np.squeeze(reconst1[img_idx])
                            save_img = Image.fromarray(save_img, 'RGB')
                            img_file_name = "{}_rec.png".format(validate_batch_files[img_idx].split('.')[0])
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)
                    elif dataset == 'dsprites':
                        for img_idx in range(len(images1)):
                            sub_dir = "iter_{}".format(batch_idx)

                            # save_img = images1[img_idx].reshape(64, 64)
                            save_img = np.squeeze(images1[img_idx])
                            save_img = save_img.astype(np.float32)
                            img_file_name = "{}_ori.jpg".format(img_idx)
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

                            # save_img = reconst1[img_idx].reshape(64, 64)
                            save_img = np.squeeze(reconst1[img_idx])
                            save_img = save_img.astype(np.float32)
                            img_file_name = "{}_rec.jpg".format(img_idx)
                            validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

                if batch_idx % loss_check_n == 1:
                    train_writer.add_summary(summary, batch_idx)

            t_epoch_end = time.time()
            t_epoch_run = t_epoch_end - t_epoch_start
            if dataset == 'dsprites':
                t_check = manager.sample_size / t_epoch_run

                warn("==========================================")
                warn("Run {} th epoch in {} sec: {} images / sec".format(epoch_idx+1, t_epoch_run, t_check))
                warn("==========================================")

            # if epoch_idx % save_model_freq == 0:
            if meta_saved == True:
                saver.save(U.get_session(), chk_save_dir + '/' + 'checkpoint', global_step = epoch_idx, write_meta_graph = False)
            else:
                print "Save  meta graph"
                saver.save(U.get_session(), chk_save_dir + '/' + 'checkpoint', global_step = epoch_idx, write_meta_graph = True)
                meta_saved = True

    # Testing
    elif mode == 'test':
        test_file_name = testing_images_list[0]
        test_img = load_single_img(dir_name = testing_img_dir, img_name = test_file_name)
        test_feature = 31
        test_variation = np.arange(-5, 5, 0.1)

        z = test(test_img)
        for idx in range(len(test_variation)):
            z_test = np.copy(z)
            z_test[0, test_feature] = z_test[0, test_feature] + test_variation[idx]
            reconst_test = test_reconst(z_test)
            test_save_img = np.squeeze(reconst_test[0])
            test_save_img = Image.fromarray(test_save_img)
            img_file_name = "test_feat_{}_var_({}).png".format(test_feature, test_variation[idx])
            test_img_saver.save(test_save_img, img_file_name, sub_dir = None)
        reconst_test = test_reconst(z)
        test_save_img = np.squeeze(reconst_test[0])
        test_save_img = Image.fromarray(test_save_img)
        img_file_name = "test_feat_{}_var_original.png".format(test_feature)
        test_img_saver.save(test_save_img, img_file_name, sub_dir = None)
