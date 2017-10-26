import tf_util as U
import tensorflow as tf
import os
import sys
from misc_util import set_global_seeds, read_dataset, warn, mkdir_p, failure, header, get_cur_dir, load_image, Img_Saver, load_single_img
import argparse
import matplotlib.pyplot as plt
# from skimage.io import imsave
import h5py
import pandas as pd
from PIL import Image
import numpy as np
import random

def train_net(model, img_dir, chkfile_name, logfile_name, validatefile_name, entangled_feat, max_iter = 50000, check_every_n = 500, loss_check_n = 10, save_model_freq = 1000, batch_size = 64):
	img1 = U.get_placeholder_cached(name="img1")
	img2 = U.get_placeholder_cached(name="img2")


	# Testing
	# img_test = U.get_placeholder_cached(name="img_test")
	# reconst_tp = U.get_placeholder_cached(name="reconst_tp")


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

	decoded_img = [model.reconst1, model.reconst2]


	weight_loss = [1, 1, 1]

	compute_losses = U.function([img1, img2], vae_loss)
	lr = 0.0001
	optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon = 0.01/batch_size)

	all_var_list = model.get_trainable_variables()

	# print all_var_list
	img1_var_list = all_var_list
	#[v for v in all_var_list if v.name.split("/")[1].startswith("proj1") or v.name.split("/")[1].startswith("unproj1")]
	optimize_expr1 = optimizer.minimize(vae_loss, var_list=img1_var_list)
	merged = tf.summary.merge_all()
	train = U.function([img1, img2], 
						[losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], latent_z1_tp, latent_z2_tp, merged], updates = [optimize_expr1])
	get_reconst_img = U.function([img1, img2], [model.reconst1, model.reconst2, latent_z1_tp, latent_z2_tp])
	get_latent_var = U.function([img1, img2], [latent_z1_tp, latent_z2_tp])


	# testing
	# test = U.function([img_test], model.latent_z_test)
	# test_reconst = U.function([reconst_tp], [model.reconst_test])

	cur_dir = get_cur_dir()
	chk_save_dir = os.path.join(cur_dir, chkfile_name)
	log_save_dir = os.path.join(cur_dir, logfile_name)
	validate_img_saver_dir = os.path.join(cur_dir, validatefile_name)
	test_img_saver_dir = os.path.join(cur_dir, "test_images")
	testing_img_dir = os.path.join(cur_dir, "dataset/test_img")
	
	train_writer = U.summary_writer(dir = log_save_dir)


	U.initialize()

	saver, chk_file_num = U.load_checkpoints(load_requested = True, checkpoint_dir = chk_save_dir)
	validate_img_saver = Img_Saver(validate_img_saver_dir)

	# testing
	# test_img_saver = Img_Saver(test_img_saver_dir)

	meta_saved = False

	iter_log = []
	loss1_log = []
	loss2_log = []

	loss3_log = []

	training_images_list = read_dataset(img_dir)
	n_total_train_data = len(training_images_list)

	testing_images_list = read_dataset(testing_img_dir)
	n_total_testing_data = len(testing_images_list)

	training = True
	testing = False

	if training == True:
		for num_iter in range(chk_file_num+1, max_iter):
			header("******* {}th iter: *******".format(num_iter))

			idx = random.sample(range(n_total_train_data), 2*batch_size)
			batch_files = [training_images_list[i] for i in idx]
			# print batch_files
			[images1, images2] = load_image(dir_name = img_dir, img_names = batch_files)
			img1, img2 = images1, images2
			[l1, l2, _, _] = get_reconst_img(img1, img2)

			[loss0, loss1, loss2, loss3, loss4, loss5, latent1, latent2, summary] = train(img1, img2)	

			warn("Total Loss: {}".format(loss0))
			warn("Siam loss: {}".format(loss1))
			warn("kl1_loss: {}".format(loss2))
			warn("kl2_loss: {}".format(loss3))
			warn("reconst_err1: {}".format(loss4))
			warn("reconst_err2: {}".format(loss5))

			# warn("num_iter: {} check: {}".format(num_iter, check_every_n))
			# warn("Total Loss: {}".format(loss6))
			if num_iter % check_every_n == 1:
				header("******* {}th iter: *******".format(num_iter))
				idx = random.sample(range(len(training_images_list)), 2*5)
				validate_batch_files = [training_images_list[i] for i in idx]
				[images1, images2] = load_image(dir_name = img_dir, img_names = validate_batch_files)
				[reconst1, reconst2, _, _] = get_reconst_img(images1, images2)
				# for i in range(len(latent1[0])):
				# 	print "{} th: {:.2f}".format(i, np.mean(np.abs(latent1[:, i] - latent2[:, i])))
				for img_idx in range(len(images1)):
					sub_dir = "iter_{}".format(num_iter)

					save_img = np.squeeze(images1[img_idx])
					save_img = Image.fromarray(save_img)
					img_file_name = "{}_ori.png".format(validate_batch_files[img_idx].split('.')[0])				
					validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

					save_img = np.squeeze(reconst1[img_idx])
					save_img = Image.fromarray(save_img)
					img_file_name = "{}_rec.png".format(validate_batch_files[img_idx].split('.')[0])				
					validate_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

			if num_iter % loss_check_n == 1:
				train_writer.add_summary(summary, num_iter)

			if num_iter > 11 and num_iter % save_model_freq == 1:
				if meta_saved == True:
					saver.save(U.get_session(), chk_save_dir + '/' + 'checkpoint', global_step = num_iter, write_meta_graph = False)
				else:
					print "Save  meta graph"
					saver.save(U.get_session(), chk_save_dir + '/' + 'checkpoint', global_step = num_iter, write_meta_graph = True)
					meta_saved = True

	# Testing
	# if testing == True:
	# 	test_file_name = testing_images_list[0]
	# 	print test_file_name
	# 	test_img = load_single_img(dir_name = testing_img_dir, img_name = test_file_name)
	# 	test_feature = 31
	# 	test_variation = np.arange(-5, 5, 0.1)

	# 	z = test(test_img)
	# 	print np.shape(z)
	# 	print z
	# 	for idx in range(len(test_variation)):
	# 		z_test = np.copy(z)
	# 		z_test[0, test_feature] = z_test[0, test_feature] + test_variation[idx]
	# 		reconst_test = test_reconst(z_test)
	# 		test_save_img = np.squeeze(reconst_test[0])
	# 		test_save_img = Image.fromarray(test_save_img)
	# 		img_file_name = "test_feat_{}_var_({}).png".format(test_feature, test_variation[idx])
	# 		test_img_saver.save(test_save_img, img_file_name, sub_dir = None)
	# 	reconst_test = test_reconst(z)
	# 	test_save_img = np.squeeze(reconst_test[0])
	# 	test_save_img = Image.fromarray(test_save_img)
	# 	img_file_name = "test_feat_{}_var_original.png".format(test_feature)
	# 	test_img_saver.save(test_save_img, img_file_name, sub_dir = None)
