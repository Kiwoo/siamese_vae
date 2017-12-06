import tf_util as U
import tensorflow as tf
import numpy as np

# Check max-pooling on proj, deconv net

class mymodel(object):
	def __init__(self, name, *args, **kwargs):
		with tf.variable_scope(name):
			self._init(*args, **kwargs)
			self.scope = tf.get_variable_scope().name

	def _init(self, img1, img2, img_shape, latent_dim, disentangled_feat, mode, loss_weight):

		# (1) batch size
		sequence_length = None

		# (2) input images, for siamese network, use two image input
		# ywpaul [change] inputs are given (for multi-GPU)
		# img1 = U.get_placeholder(name="img1", dtype=tf.float32, shape=[sequence_length, img_shape[0], img_shape[1], img_shape[2]])
		# img2 = U.get_placeholder(name="img2", dtype=tf.float32, shape=[sequence_length, img_shape[0], img_shape[1], img_shape[2]])

		self.channel_img = img_shape[2]

		# (3) # of disentangled or entangled features,
		disentangle_feat_sz = disentangled_feat
		entangle_feat_sz = latent_dim - disentangle_feat_sz

		# (4) this code is used only when test mode.
		# [testing -> ]
		if mode == 'test':
			img_test = U.get_placeholder(name="img_test", dtype=tf.float32, shape=[sequence_length, img_shape[0], img_shape[1], img_shape[2]])
			reconst_tp = U.get_placeholder(name="reconst_tp", dtype=tf.float32, shape=[sequence_length, latent_dim])
		# [testing <- ]

		# (5) If we need re-scale the original images. Currently not used.
		img1_scaled = img1
		img2_scaled = img2

		# (6) Encoder

		[mu1, logvar1, mu2, logvar2] = self.siamese_encoder(img1_scaled, img2_scaled, latent_dim)

		# (7) Sampling with mean and log-variance

		latent_z1 = self.sample_latent_var(mu1, logvar1)
		latent_z2 = self.sample_latent_var(mu2, logvar2)

		# (8) Decoder: reconst1, 2 are reconstructed images

		[reconst1, reconst2] = self.siamese_decoder(latent_z1, latent_z2)

		# (9) To use outside of model
		self.latent_z1 = latent_z1
		self.latent_z2 = latent_z2
		self.reconst1 = reconst1
		self.reconst2 = reconst2

		# [testing -> ]
		if mode == 'test':
			selected_feature = 28
			latent_v_range = np.arange(-1, 1, 0.5)

			[mu_test, logvar_test] = self.encoder(img_test, latent_dim)
			self.latent_z_test = self.sample_latent_var(mu_test, logvar_test)
			print ("----")
			print np.shape(self.latent_z_test)
			self.reconst_test = self.decoder(reconst_tp)
		# [testing <- ]

		# (10) Preparing Siamese Loss(self.siam_loss), Maximum Wasserstein distance(self.max_siam_loss) in Entangled Features
		# Used Wasserstein Metric
		# Refer: http://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/

		sh_mu1 = mu1[:, 0:entangle_feat_sz]
		sh_logvar1 = logvar1[:, 0:entangle_feat_sz]

		sh_mu2 = mu2[:, 0:entangle_feat_sz]
		sh_logvar2 = logvar2[:, 0:entangle_feat_sz]

		self.siam_loss = U.sum(tf.square(sh_mu1 - sh_mu2), axis = 1) + U.sum(tf.square(tf.sqrt(tf.exp(sh_logvar1)) - tf.sqrt(tf.exp(sh_logvar2))), axis = 1)

		self.max_siam_loss = U.max(tf.square(sh_mu1 - sh_mu2) + tf.square(tf.sqrt(tf.exp(sh_logvar1)) - tf.sqrt(tf.exp(sh_logvar2))), axis = 1)

		# (11) Preparing KL Loss between Gaussian(mu, logvar) and N(0, 1) following original VAE paper
		# refer: https://arxiv.org/pdf/1312.6114.pdf    (:page 5)

		self.kl_loss1 = 0.5 * U.sum((tf.exp(logvar1) + mu1**2 - 1. - logvar1), axis = 1)
		self.kl_loss2 = 0.5 * U.sum((tf.exp(logvar2) + mu2**2 - 1. - logvar2), axis = 1)

		# (12) Reconstruction error from both two images, used L2 loss

		reconst_error1 = tf.square(reconst1 - img1_scaled)
		self.reconst_error1 = tf.reduce_sum(reconst_error1, [1, 2, 3])
		reconst_error2 = tf.square(reconst2 - img2_scaled)
		self.reconst_error2 = tf.reduce_sum(reconst_error2, [1, 2, 3])

		# (13) Total loss is weighted sum of all losses above.

		self.vaeloss = loss_weight['siam']*self.siam_loss + loss_weight['kl']*self.kl_loss1 + loss_weight['kl']*self.kl_loss2 + self.reconst_error1 + self.reconst_error2



	def encoder_net(self, img, latent_dim):
		x = img
		x = tf.nn.relu(U.conv2d(x, 32, "c1", [4, 4], [2, 2], pad = "SAME")) # [32, 32, 32]
		x = tf.nn.relu(U.conv2d(x, 32, "c2", [4, 4], [2, 2], pad = "SAME")) # [16, 16, 32]
		x = tf.nn.relu(U.conv2d(x, 64, "c3", [4, 4], [2, 2], pad = "SAME")) # [8, 8, 64]
		x = tf.nn.relu(U.conv2d(x, 64, "c4", [4, 4], [2, 2], pad = "SAME")) # [4, 4, 64]
		x = U.flattenallbut0(x) # [1024]
		x = tf.nn.relu(U.dense(x, 256, 'l1', U.normc_initializer(1.0))) # 1024
		mu = U.dense(x, latent_dim, 'l1_1', U.normc_initializer(1.0)) # 32
		logvar = U.dense(x, latent_dim, 'l1_2', U.normc_initializer(1.0)) # 32
		return mu, logvar

	def decoder_net(self, latent_variable):
		x = latent_variable
		x = U.dense(x, 256, 'l2', U.normc_initializer(1.0))
		x = tf.nn.relu(U.dense(x, 1024, 'l3', U.normc_initializer(1.0)))
		x = tf.reshape(x, [tf.shape(x)[0], 4,4,64]) # Unflatten [4, 4, 64]
		x = tf.nn.relu(U.conv2d_transpose(x, [4,4,64,64], [tf.shape(x)[0], 8,8,64], "uc1", [2, 2], pad="SAME")) # [8, 8, 64]
		x = tf.nn.relu(U.conv2d_transpose(x, [4,4,32,64], [tf.shape(x)[0], 16,16,32], "uc2", [2, 2], pad="SAME")) # [16, 16, 32]
		x = tf.nn.relu(U.conv2d_transpose(x, [4,4,32,32], [tf.shape(x)[0], 32,32,32], "uc3", [2, 2], pad="SAME")) # [32, 32, 32]
		x = U.conv2d_transpose(x, [4,4,self.channel_img, 32], [tf.shape(x)[0], 64,64,self.channel_img], "uc4", [2, 2], pad="SAME") # [64, 64, channel_img = 1 or 3]
		return x

	def sample_latent_var(self, mu, logvar):
		eps = tf.random_normal(shape=tf.shape(mu))
		return mu + tf.exp(logvar / 2) * eps

	def siamese_encoder(self, s1, s2, latent_dim):
		with tf.variable_scope("siamese_encoder") as scope:
			[mu1, logvar1] = self.encoder_net(img = s1, latent_dim = latent_dim)
			scope.reuse_variables()
			[mu2, logvar2] = self.encoder_net(img = s2, latent_dim = latent_dim)
		return [mu1, logvar1, mu2, logvar2]

	# [testing -> ]
	# def encoder(self, s1, latent_dim):
	# 	with tf.variable_scope("siamese_encoder") as scope:
	# 		scope.reuse_variables()
	# 		[mu1, logvar1] = self.encoder_net(img = s1, latent_dim = latent_dim)
	# 	return [mu1, logvar1]

	# def decoder(self, l1):
	# 	with tf.variable_scope("siamese_decoder") as scope:
	# 		scope.reuse_variables()
	# 		reconst_test = self.decoder_net(latent_variable = l1)
	# 	return reconst_test
	# [testing <- ]

	def siamese_decoder(self, l1, l2):
		with tf.variable_scope("siamese_decoder") as scope:
			reconst1 = self.decoder_net(latent_variable = l1)
			scope.reuse_variables()
			reconst2 = self.decoder_net(latent_variable = l2)
		return [reconst1, reconst2]

	def get_loss(self):
		return [self.siam_loss, self.kl_loss1, self.kl_loss2, self.reconst_error1, self.reconst_error2]

	def get_variables(self):
		return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
	def get_initial_state(self):
		return []