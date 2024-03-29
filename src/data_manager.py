import numpy as np
import os.path as osp
import random
from misc_util import warn

class DataManager(object):
	def __init__(self, img_dir, batch_size = 64):
		self.img_dir = img_dir
		self.batch_size= batch_size
		dataset_zip = np.load(img_dir)
		# print('Keys in the dataset:', dataset_zip.keys())
		#  ['metadata', 'imgs', 'latents_classes', 'latents_values']
		self.imgs       = dataset_zip['imgs']
		latents_values  = dataset_zip['latents_values']
		latents_classes = dataset_zip['latents_classes']
		metadata        = dataset_zip['metadata'][()]

		# Define number of values per latents and functions to convert to indices
		latents_sizes = metadata['latents_sizes']
		self.latents_sizes = latents_sizes
		# [ 1,  3,  6, 40, 32, 32]
		# color, shape, scale, orientation, posX, posY

		self.n_samples = latents_sizes[::-1].cumprod()[-1]
		# 737280

		self.latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
		# [737280, 245760, 40960, 1024, 32, 1]
		self.imglist = range(self.n_samples)
		random.shuffle(self.imglist)

		self.max_batch = self.n_samples / (self.batch_size * 2)		
		self.cur_batch_idx = 0



	@property
	def sample_size(self):
		return self.n_samples

	def get_len(self):
		return self.max_batch

	def get_image_fixed_feat_batch(self, feats, num_img_pair):
		# k in beta-VAE paper		
		# color: latents_sizes[0] is fixed as 0
		# feat starts from 0 to 4. 0: shape, 1: scale, ... 

        # L = 10
        # batch_per_gpu = 5
        # num_img_pair = L * num_gpus * batch_per_gpu
        # feat = np.random.randint(manager.latents_sizes-1, num_gpus * batch_per_gpu)

		images1 = []
		images2 = []
		feature = np.zeros(len(self.latents_sizes)-1)

		L = num_img_pair / len(feats)
		for feat in feats:
			for l in range(2*L):
				for i in range(len(self.latents_sizes)-1):
					feature[i] = np.random.randint(self.latents_sizes[i+1])
				if l % 2 == 0:
					fixed_feat_value = feature[feat]
				elif l % 2 == 1:
					feature[feat] = fixed_feat_value

				if l % 2 == 0:
					# warn("features: {}".format(feature))
					images1.append(self.get_image(feature[0], feature[1], feature[2], feature[3], feature[4]))
				elif l % 2 == 1:
					images2.append(self.get_image(feature[0], feature[1], feature[2], feature[3], feature[4]))

		return images1, images2


	def get_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
		latents = [0, shape, scale, orientation, x, y]
		index = np.dot(latents, self.latents_bases).astype(int)
		img = self.imgs[index]
		img = np.expand_dims(img, axis = 2)
		return img
		# return self.get_images([index])[0]

	def get_images_single(self, indices):
		images = []
		for index in indices:
			img = self.imgs[index]
			img = np.expand_dims(img, axis = 2)
			# img = img.reshape(4096)
			images.append(img)
		return images

	def get_next(self):
		cur_idx = self.cur_batch_idx
		sz = self.batch_size
		indices = self.imglist[cur_idx * sz: (cur_idx+2) * sz] # Because we sample set of 2 images for one batch
		images = []
		for index in indices:
			img = self.imgs[index]
			img = np.expand_dims(img, axis = 2)
			images.append(img)
		self.cur_batch_idx += 2 # Because we sample set of 2 images for one batch
		self.cur_batch_idx = self.cur_batch_idx % self.max_batch
		return images[0:len(images)/2], images[len(images)/2:]

	def get_images(self, indices):
		images = []
		for index in indices:
			img = self.imgs[index]
			img = np.expand_dims(img, axis = 2)
			# img = img.reshape(4096)
			images.append(img)
		return images[0:len(images)/2], images[len(images)/2:]

	def get_random_images(self, size):
		indices = [np.random.randint(self.n_samples) for i in range(size)]
		return self.get_images(indices)
