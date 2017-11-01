from PIL import Image
import numpy as np
from misc_util import get_cur_dir
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
	resize_size = (64, 64)

	idx = 6632
	img_set = np.load("imgs.npy")
	print img_set[idx][32]
	for i in range(len(img_set[idx][0])):
		img_set[idx][32][i] = 1
	print img_set[idx][32]
	plt.imsave('test.png', img_set[idx], cmap=cm.gray)
	# print len(img_set)
	# test = Image.fromarray(img_set[0], 'L')
	# for i in range(len(img_set[0][0])):
	# 	test[32][i] = 244
	# test.show()
	# print img_set[0]
	# for i in range(len(img_set[0][0])):
	# 	print img_set[0][i]
	# plt.imshow(test, cmap='Greys_r')
	# test.imshow()
	# cur_dir = get_cur_dir()
	# dataset_dir = os.path.join(cur_dir, "dsprites_dataset")
	# original_data_dir = os.path.join(dataset_dir, "img_align_celeba")
	# img_save_dir = os.path.join(dataset_dir, "celebA_training_img")

	# all_subdir = [f for f in os.listdir(original_data_dir) if os.path.isdir(os.path.join(original_data_dir, f))]	
	# # print all_subdir

	# idx = 0
	# all_idx = len(all_subdir)
	# n_files = 0
	# # for subdir in all_subdir:
	# # print "Processing... {} / {}".format(idx, all_idx)	
	# # subdir_dir = os.path.join(original_data_dir, subdir)
	# # render_data_dir = os.path.join(subdir_dir, "renders")
	# img_files = [f for f in os.listdir(original_data_dir) if os.path.isfile(os.path.join(original_data_dir, f))]	
	# total_f_num = len(img_files)
	# for file_name in img_files:
	# 	img_file_path = os.path.join(original_data_dir, file_name)
	# 	im = Image.open(img_file_path)
	# 	width, height = im.size   # Get dimensions

	# 	# print width, height

	# 	left = (width - 120)/2
	# 	top = (height - 120)/2
	# 	right = (width + 120)/2
	# 	bottom = (height + 120)/2

	# 	im = im.crop((left, top, right, bottom))

	# 	im_resized = im.resize(resize_size, Image.ANTIALIAS)
	# 	file_name = "img_{}.jpg".format(n_files)
	# 	img_save_file_path = os.path.join(img_save_dir, file_name)
	# 	# im_resized = im_resized.convert("L")
	# 	img_data = np.asarray(im_resized)
	# 	test = Image.fromarray(img_data)
	# 	test.save(img_save_file_path)
	# 	n_files = n_files + 1
	# 	if n_files % 100 == 0:
	# 		print "{} \ {}".format(n_files, total_f_num)
	# idx = idx + 1
	# print "{} files processed".format(n_files)
	# print "Total Files : {}".format(n_files)

if __name__ == '__main__':
    main()
